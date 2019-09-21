import sys
import os
import argparse
import logging
import json

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, GRU, Activation, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.models import Model, save_model, load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model

from data import *
from keras_components import *
from sklearn.utils import resample

from typing import *

logger = logging.getLogger()


def build_model(max_length: int,
                embedding_matrix: Union[np.ndarray, Tuple[int, int]],
                filters: List[int],
                kernel_size: List[int],
                pool_size: List[int],
                conv_padding: str,
                pool_padding: str,
                cell_type: str,
                cell_size: int,
                cell_stack_size: int,
                dense_size: List[int],
                embedding_dropout: float = 0.6,
                embedding_not_trainable: bool = False,
                conv_dropout: float = 0.1,
                dense_dropout: Union[float, List[float]] = 0.3,
                classifier_dropout: float = 0.1,
                rnn_first: bool = False) -> Model:

    if not (len(filters) > 0 and len(kernel_size) > 0 and len(pool_size) > 0):
        logger.error("There are no filters, kernel sizes or pool sizes specified for the CNN.")
        raise ValueError("There are no filters, kernel sizes or pool sizes specified for the CNN.")

    if type(dense_dropout) != list:
        dense_dropout = [dense_dropout]

    if len(dense_size) > 0 and len(dense_size) != len(dense_dropout):
        max_list_length = max([len(dense_size), len(dense_dropout)])
        new_dense_size = []
        new_dense_dropout = []
        for i in range(max_list_length):
            new_dense_size.append(dense_size[i] if i < len(dense_size) else dense_size[-1])
            new_dense_dropout.append(dense_dropout[i] if i < len(dense_dropout) else dense_dropout[-1])
        dense_size = new_dense_size
        dense_dropout = new_dense_dropout
        logger.warning("Lists given for dense layer sizes and dense layer dropout rates are not the same length. "
                       "The shorter lists are padded using the last value to match the length of the longest.")

    if len(filters) != len(kernel_size) or len(filters) != len(pool_size) or len(kernel_size) != len(pool_size):
        max_list_length = max([len(filters), len(kernel_size), len(pool_size)])
        new_filters = []
        new_kernel_size = []
        new_pool_size = []
        for i in range(max_list_length):
            new_filters.append(filters[i] if i < len(filters) else filters[-1])
            new_kernel_size.append(kernel_size[i] if i < len(kernel_size) else kernel_size[-1])
            new_pool_size.append(pool_size[i] if i < len(pool_size) else pool_size[-1])
        filters = new_filters
        kernel_size = new_kernel_size
        pool_size = new_pool_size
        logger.warning("Lists given for convolutional filters, kernel sizes and pooling sizes had different lengths. "
                       "The shorter lists are padded using the last value to match the length of the longest.")

    cell_type_name = cell_type.lower()
    if cell_type_name == "lstm":
        cell_type = LSTM
    elif cell_type_name == "gru":
        cell_type = GRU

    # input 1: word indices, input 2: handcrafted_features
    raw_input = Input(shape=(max_length,), name="word_input")

    # embedding layer
    embedding_layer = Embedding(
        input_dim=(embedding_matrix[0] if type(embedding_matrix) == tuple else embedding_matrix.shape[0]),
        output_dim=(embedding_matrix[1] if type(embedding_matrix) == tuple else embedding_matrix.shape[1]),
        input_length=max_length,
        name="word_embedding",
        weights=(None if type(embedding_matrix) == tuple else [embedding_matrix]),
        trainable=(not embedding_not_trainable))
    embedding_dropout = Dropout(embedding_dropout, name="embedding_dropout")

    # convolutional layer(s)
    conv_layers = []
    for i in range(len(filters)):
        conv_layer_name = "conv_{}".format(i)
        convolution = Conv1D(filters[i], kernel_size[i], padding=conv_padding, activation="relu", name=conv_layer_name)
        pooling = MaxPooling1D(pool_size[i], padding=pool_padding, name="max_pool_{}".format(i))
        conv_layers.append((convolution, pooling))
    conv_dropout = Dropout(conv_dropout, name="conv_dropout")

    # bidirectional RNN
    rnn_cells = []
    for i in range(cell_stack_size - 1):
        cell_name = "{}_{}".format(cell_type_name, i)
        cell = Bidirectional(cell_type(cell_size, return_sequences=True, name=cell_name),
                             name="bidirectional_{}".format(cell_name))
        rnn_cells.append(cell)

    # last cell
    cell_name = "{}_{}".format(cell_type_name, cell_stack_size - 1)
    cell = Bidirectional(cell_type(cell_size, return_sequences=rnn_first, name=cell_name))
    rnn_cells.append(cell)

    # dense layer(s)
    dense_layers = []
    for i in range(len(dense_size)):
        dropout = Dropout(rate=dense_dropout[i], name="dense_dropout_{}".format(i))
        dense_layer_name = "dense_{}".format(i)
        dense = Dense(dense_size[i], name=dense_layer_name)
        dense_layers.append((dropout, dense))

    # classification layer
    classifier_dropout = Dropout(classifier_dropout, name="classifier_dropout")
    classifier = Dense(1, name="classifier")
    classifier_prediction = Activation("sigmoid", name="classifier_prediction")

    # build the actual model
    output = embedding_dropout(embedding_layer(raw_input))
    if rnn_first:
        for c in rnn_cells:
            output = c(output)
        for l in conv_layers:
            output = l[1](l[0](conv_dropout(output)))
        output = Flatten(name="flatten")(output)
    else:
        for l in conv_layers:
            output = conv_dropout(l[1](l[0](output)))
        for c in rnn_cells:
            output = c(output)
    for l in dense_layers:
        output = l[1](l[0](output))
    output = classifier_prediction(classifier(classifier_dropout(output)))
    model = Model(inputs=raw_input, outputs=output)

    return model


def build_and_train(args,
                    checkpoint_template: str,
                    log_dir: str,
                    embedding: Union[np.ndarray, Tuple[int]],
                    data_train: Union[np.ndarray, List[np.ndarray]],
                    targets_train: Union[np.ndarray, List[np.ndarray]],
                    data_val: Union[np.ndarray, List[np.ndarray]] = None,
                    targets_val: Union[np.ndarray, List[np.ndarray]] = None,
                    cv_fold: int = None) -> Model:

    validation_data = None
    if data_val is not None and targets_val is not None:
        validation_data = (data_val, targets_val)

    # create the model
    optimizer = Adam(lr=args.learning_rate)

    # load or build the model
    if args.model_load_file:
        model = load_model(args.model_load_file)
    else:
        model = build_model(max_length=args.max_length,
                            embedding_matrix=embedding,
                            filters=args.filters,
                            kernel_size=args.kernel_size,
                            pool_size=args.pool_size,
                            conv_padding=args.conv_padding,
                            pool_padding=args.pool_padding,
                            cell_type=args.cell_type,
                            cell_size=args.cell_size,
                            cell_stack_size=args.cell_stack_size,
                            dense_size=args.dense_size,
                            embedding_dropout=args.embedding_dropout,
                            embedding_not_trainable=args.embedding_not_trainable,
                            conv_dropout=args.conv_dropout,
                            dense_dropout=args.dense_dropout,
                            classifier_dropout=args.classifier_dropout,
                            rnn_first=args.rnn_first)

    if args.gpus > 1:
        logger.warning("Using {} GPUs. This means that batches are divided up between the GPUs!".format(args.gpus))
        model = multi_gpu_model(model, gpus=args.gpus)

    # losses depend on what model is loaded
    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=["accuracy"])
    if 2 != args.verbosity > 0:
        model.summary()

    # use different learning rate schedulers for the different models
    cb = []
    lr_scheduler = LearningRateScheduler(CosineLRSchedule(lr_high=args.learning_rate,
                                                          lr_low=args.learning_rate / args.batch_size,
                                                          initial_period=args.epochs), verbose=0)
    cb.append(lr_scheduler)

    # model config saved separately from weights
    directory = os.path.dirname(os.path.abspath(checkpoint_template))
    if not os.path.exists(directory):
        os.makedirs(directory)
    config = model.get_config()
    config_save_file = os.path.join(directory, "config.json")
    with open(config_save_file, "w") as f:
        json.dump(config, f)

    # checkpoints for model weights
    period = args.save_every if args.save_every else max(int(args.epochs / 10), 1)
    checkpoint = ModelCheckpoint(checkpoint_template, monitor="loss", verbose=1,
                                 save_best_only=False, mode="min", period=period)

    # tensorboard for monitoring progress
    tensorboard_dir = log_dir
    if cv_fold is not None:
        tensorboard_dir = os.path.join(tensorboard_dir, "cv_fold_{}".format(cv_fold))
    directory = os.path.dirname(os.path.abspath(tensorboard_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    tensorboard = TensorBoard(tensorboard_dir, write_graph=False, update_freq="epoch")

    if not args.cross_validation:
        cb.insert(0, checkpoint)
    cb.append(tensorboard)

    # train model
    model.fit(x=data_train, y=targets_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=validation_data, shuffle=True, verbose=args.verbosity, callbacks=cb)

    return model


def build_and_train_ensemble(args,
                             embedding: Union[np.ndarray, Tuple[int]],
                             data_train: Union[np.ndarray, List[np.ndarray]],
                             targets_train: Union[np.ndarray, List[np.ndarray]],
                             data_val: Union[np.ndarray, List[np.ndarray]] = None,
                             targets_val: Union[np.ndarray, List[np.ndarray]] = None) -> List[Model]:

    validation_data = None
    if data_val is not None and targets_val is not None:
        validation_data = (data_val, targets_val)

    # since we want to save each model once it is trained, create the model save directory here
    model_save_dir = os.path.abspath(args.model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # the optimiser is the same for all models
    optimizer = Adam(lr=args.learning_rate)

    ensemble = []
    for i in range(args.ensemble_count):
        logger.info("Training ensemble classifier {:02d}/{:02d}...".format(i + 1, args.ensemble_count))

        # sample from the data
        n = len(data_train[0] if type(data_train) == list else data_train)
        indices = np.array(range(n))
        indices = resample(indices, n_samples=int(n * args.ensemble_sample_pct), random_state=args.random_seed)
        data, _, targets, _ = split_by_index(indices, indices, data_train, targets_train)

        # load or build the model
        if args.model_config_file:
            model = load_model(args.model_load_file)
        else:
            model = build_model(max_length=args.max_length,
                                embedding_matrix=embedding,
                                filters=args.filters,
                                kernel_size=args.kernel_size,
                                pool_size=args.pool_size,
                                conv_padding=args.conv_padding,
                                pool_padding=args.pool_padding,
                                cell_type=args.cell_type,
                                cell_size=args.cell_size,
                                cell_stack_size=args.cell_stack_size,
                                dense_size=args.dense_size,
                                embedding_dropout=args.embedding_dropout,
                                embedding_not_trainable=args.embedding_not_trainable,
                                conv_dropout=args.conv_dropout,
                                dense_dropout=args.dense_dropout,
                                classifier_dropout=args.classifier_dropout,
                                rnn_first=args.rnn_first)

        # complete by compiling the model
        model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=["accuracy"])
        if i == 0 and 2 != args.verbosity > 0:
            model.summary()

        # only use the LR scheduler callback for now, no tensorboard or model checkpoints
        cb = [LearningRateScheduler(CosineLRSchedule(lr_high=args.learning_rate,
                                                     lr_low=args.learning_rate / args.batch_size,
                                                     initial_period=args.epochs), verbose=0)]

        # train model (and also evaluate it on validation data just to see how individual classifiers perform)
        model.fit(x=data, y=targets, batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=validation_data, shuffle=True, verbose=args.verbosity, callbacks=cb)

        # save the config
        if i == 0:
            config = model.get_config()
            config_save_file = os.path.join(model_save_dir, "config.json")
            with open(config_save_file, "w") as f:
                json.dump(config, f)

        # save the model
        if not args.cross_validation:
            model_save_file = os.path.join(model_save_dir, f"ensemble_model-{i:02d}-{args.epochs:03d}.hdf5")
            save_model(model, model_save_file)

        # add model to the ensemble
        ensemble.append(model)
        logger.info("Finished training ensemble classifier {:02d}/{:02d}.".format(i + 1, args.ensemble_count))

    logger.info("Finished training {} ensemble classifiers.".format(args.ensemble_count))

    return ensemble


def train(args, log_dir: str):
    if args.model_save_dir:
        directory = os.path.abspath(args.model_save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        json_file = os.path.join(args.model_save_dir, args.model_name + ".json")
        args_dict = vars(args)
        with open(json_file, "w") as f:
            json.dump(args_dict, f)

    if args.model_weights_file and not args.model_config_file:
        logger.error("Cannot load a model from weights only, a config file needs to be provided.")
        raise ValueError("Cannot load a model from weights only, a config file needs to be provided.")

    # make checkpoint template
    checkpoint_template = os.path.join(args.checkpoint_dir, args.model_name, "model-{epoch:03d}.hdf5")

    # load data
    data, targets = load_data(args.data_path, split=(not args.use_bpe))

    # construct vocabulary from data
    vocabulary = construct_vocabulary(args.vocabulary_source_file or data, vocabulary_size=args.vocabulary_size,
                                      use_bpe=args.use_bpe, bpe_percentage=args.bpe_percentage,
                                      vocabulary_save_file=args.vocabulary_save_file)

    # maybe load embedding vectors and construct matrix
    if not args.use_bpe:
        embedding_vectors = load_embedding_vectors(vocabulary, embedding_file=args.embedding_file)
        embedding = create_embedding_matrix(vocabulary, embedding_vectors)
    else:
        embedding = (args.vocabulary_size, args.embedding_size)

    # encode data (don't need LM targets here)
    data = encode_data(data, vocabulary, max_length=args.max_length, use_bpe=args.use_bpe)
    if not args.max_length:
        args.max_length = data.shape[1]
        logger.debug("Maximum length of the encoded data is {}.".format(args.max_length))

    # split data
    data_train = data_val = targets_train = targets_val = split = None
    if args.cross_validation or args.validation_data is None:
        split = split_data(data, targets,
                           cross_validation=args.cross_validation,
                           train_val_split=args.train_val_split,
                           cv_folds=args.cross_validation_folds,
                           random_seed=args.random_seed)
        if not args.cross_validation:
            data_train, data_val, targets_train, targets_val = split
    else:
        logger.info("Using validation set instead of splitting training data.")

        data_train = data
        targets_train = targets

        # load validation data
        data_val, labels_val = load_data(args.validation_data, split=(not args.use_bpe))

        # encode validation data
        data_val, targets_val = encode_data(data_val, vocabulary, labels=labels_val, max_length=args.max_length,
                                            use_bpe=args.use_bpe, for_classification=args.fine_tune_model)

    # potentially sample from the training set to reduce its size
    if args.train_on_subset < 1.0:
        data_train, _, targets_train, _ = split_data(data_train, targets_train, train_val_split=args.train_on_subset,
                                                     random_seed=args.random_seed)
    if args.validate_on_subset < 1.0:
        data_val, _, targets_val, _ = split_data(data_val, targets_val, train_val_split=args.validate_on_subset,
                                                 random_seed=args.random_seed)

    # create the model
    model = None
    if not args.cross_validation:
        logger.info("Starting training...")
        model = build_and_train(args, checkpoint_template, log_dir, embedding,
                                data_train, targets_train, data_val, targets_val)
        logger.info("Finished training.")
    else:
        logger.info("Starting cross validation...")
        cross_validation_scores = []
        x = data[0] if type(data) == list else data
        y = targets[1] if type(targets) == list else targets
        for i, (train_ind, val_ind) in enumerate(split.split(x, y)):
            # get the data
            data_train, data_val, targets_train, targets_val = split_by_index(train_ind, val_ind, data, targets)

            # train the model on the current folds
            model = build_and_train(args, checkpoint_template, log_dir, embedding,
                                    data_train, targets_train, data_val, targets_val, i)

            # evaluate model
            scores = model.evaluate(x=data_val, y=targets_val, batch_size=args.batch_size, verbose=args.verbosity)
            accuracy_index = model.metrics_names.index("classifier_prediction_acc")
            cross_validation_scores.append(scores[accuracy_index] * 100.0)

            # print current score
            logger.info("Fold {} accuracy: {:.2f}%".format(i, cross_validation_scores[-1]))
        logger.info("SCORES OVER {} FOLDS: {:.2f}% (+/- {:.2f}%)".format(args.cross_validation_folds,
                                                                         np.mean(cross_validation_scores),
                                                                         np.std(cross_validation_scores)))

    # save model
    if not args.cross_validation and args.model_save_dir:
        model_save_file = os.path.join(args.model_save_dir, args.model_name + ".hdf5")
        log_string = "Saving model to \"{}\".".format(model_save_file)
        if args.model_load_file:
            log_string += " Model was previously loaded from \"{}\".".format(args.model_load_file)
        logger.info(log_string)
        save_model(model, model_save_file)


def evaluate_ensemble(args,
                      ensemble: List[Model],
                      data_val: Union[np.ndarray, List[np.ndarray]],
                      targets_val: Union[np.ndarray, List[np.ndarray]]) -> Tuple[float, float]:

    # get predictions
    predictions_hard = []
    predictions_soft = []
    for model_idx, model in enumerate(ensemble):
        logger.info("Evaluating ensemble classifier {:02d}/{:02d}...".format(model_idx + 1, args.ensemble_count))

        predictions = model.predict(data_val, batch_size=args.batch_size, verbose=1)
        predictions = predictions.reshape(-1)
        predictions_soft.append(predictions)
        predictions = np.round(predictions).astype(int)
        predictions_hard.append(predictions)

    predictions_hard = np.array(predictions_hard)
    predictions_soft = np.array(predictions_soft)

    # get single prediction using hard voting and soft averaging
    predictions_hard = np.round(np.mean(predictions_hard, axis=0))
    predictions_soft = np.round(np.mean(predictions_soft, axis=0))

    # check how many were correct
    correct = targets_val[1] if type(targets_val) == list else targets_val
    accuracy_hard = (predictions_hard == correct).mean() * 100
    accuracy_soft = (predictions_soft == correct).mean() * 100

    logger.info("Finished evaluating {} ensemble classifiers.".format(args.ensemble_count))
    return accuracy_hard, accuracy_soft


def train_ensemble(args):
    if args.model_save_dir:
        directory = os.path.abspath(args.model_save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        json_file = os.path.join(args.model_save_dir, args.model_name + ".json")
        args_dict = vars(args)
        with open(json_file, "w") as f:
            json.dump(args_dict, f)
    else:
        logger.warning("No model save directory specified. During ensemble training, no checkpoints are saved. "
                       "Instead, the checkpoint directory will be used as the model save directory.")
        args.model_save_dir = os.path.join(args.checkpoint_dir, args.model_name)

    # load data
    data, targets = load_data(args.data_path, split=(not args.use_bpe))

    # construct vocabulary from data
    vocabulary = construct_vocabulary(args.vocabulary_source_file or data, vocabulary_size=args.vocabulary_size,
                                      use_bpe=args.use_bpe, bpe_percentage=args.bpe_percentage,
                                      vocabulary_save_file=args.vocabulary_save_file)

    # maybe load embedding vectors and construct matrix
    if not args.use_bpe:
        embedding_vectors = load_embedding_vectors(vocabulary, embedding_file=args.embedding_file)
        embedding = create_embedding_matrix(vocabulary, embedding_vectors)
    else:
        embedding = (args.vocabulary_size, args.embedding_size)

    # encode data (don't need LM targets here)
    data = encode_data(data, vocabulary, max_length=args.max_length, use_bpe=args.use_bpe)
    if not args.max_length:
        args.max_length = data.shape[1]
        logger.debug("Maximum length of the encoded data is {}.".format(args.max_length))

    # split data
    data_train = data_val = targets_train = targets_val = split = None
    if args.cross_validation or args.validation_data is None:
        split = split_data(data, targets,
                           cross_validation=args.cross_validation,
                           train_val_split=args.train_val_split,
                           cv_folds=args.cross_validation_folds,
                           random_seed=args.random_seed)
        if not args.cross_validation:
            data_train, data_val, targets_train, targets_val = split
        else:
            logger.warning("Performing cross validation for ensemble classifier. This may take a long time.")
    else:
        logger.info("Using validation set instead of splitting training data.")

        data_train = data
        targets_train = targets

        # load validation data
        data_val, labels_val = load_data(args.validation_data, split=(not args.use_bpe))

        # encode validation data
        data_val, targets_val = encode_data(data_val, vocabulary, labels=labels_val, max_length=args.max_length,
                                            use_bpe=args.use_bpe, for_classification=args.fine_tune_model)

    # potentially sample from the data set to reduce its size
    if args.train_on_subset < 1.0:
        data_train, _, targets_train, _ = split_data(data_train, targets_train,
                                                     train_val_split=args.train_on_subset,
                                                     random_seed=args.random_seed)
    if args.validate_on_subset < 1.0:
        data_val, _, targets_val, _ = split_data(data_val, targets_val, train_val_split=args.validate_on_subset,
                                                 random_seed=args.random_seed)

    if not args.cross_validation:
        logger.info("Starting training...")
        ensemble = build_and_train_ensemble(args, embedding, data_train, targets_train, data_val, targets_val)
        logger.info("Finished training.")

        # do final evaluation
        accuracy_hard, accuracy_soft = evaluate_ensemble(args, ensemble, data_val, targets_val)
        logger.info("Final validation accuracy: {:.2f}% (hard), {:.2f}% (soft)".format(accuracy_hard, accuracy_soft))
    else:
        logger.info("Starting cross validation...")
        scores_hard = []
        scores_soft = []
        x = data[0] if type(data) == list else data
        y = targets[1] if type(targets) == list else targets
        for i, (train_ind, val_ind) in enumerate(split.split(x, y)):
            # get the data
            data_train, data_val, targets_train, targets_val = split_by_index(train_ind, val_ind, data, targets)

            # train the model on the current fold
            ensemble = build_and_train_ensemble(args, embedding, data_train, targets_train, data_val, targets_val)

            # evaluate model
            accuracy_hard, accuracy_soft = evaluate_ensemble(args, ensemble, data_val, targets_val)
            scores_hard.append(accuracy_hard)
            scores_soft.append(accuracy_soft)

            # print current score
            logger.info("Fold {} accuracy: {:.2f}% (hard), {:.2f}% (soft)".format(i, accuracy_hard, accuracy_soft))
        logger.info("SCORES OVER {} FOLDS: {:.2f}% (+/- {:.2f}%) (hard), {:.2f}% (+/- {:.2f}%) (soft)".format(
            args.cross_validation_folds,
            np.mean(scores_hard), np.std(scores_hard),
            np.mean(scores_soft), np.std(scores_soft)))


def predict(args):
    # load data
    data_test = load_data(args.data_path, predict=True,  split=(not args.use_bpe))

    # load vocabulary
    vocabulary = construct_vocabulary(args.vocabulary_source_file, vocabulary_size=args.vocabulary_size,
                                      use_bpe=args.use_bpe, bpe_percentage=args.bpe_percentage,
                                      vocabulary_save_file=args.vocabulary_save_file)

    # encode the data, IMPORTANT: need max_length to be specified
    data_test = encode_data(data_test, vocabulary, max_length=args.max_length, use_bpe=args.use_bpe)

    # load the model
    model = load_model(args.model_load_file)

    # do the predictions:
    predictions = model.predict(data_test, batch_size=args.batch_size, verbose=1)
    prediction_probabilities = predictions.reshape(-1)
    predictions = np.round(predictions.reshape(-1)).astype(int)

    # save predictions to file
    if args.prediction_save_file:
        if os.path.isdir(args.prediction_save_file):
            file_name = os.path.join(args.prediction_save_file, "predictions.csv")
        else:
            file_name = args.prediction_save_file + ("" if ".csv" in args.prediction_save_file else ".csv")
    else:
        file_name = os.path.abspath(args.data_path)
        file_name = re.sub(r"\.csv", "", file_name)
        file_name = re.sub(r"\.tsv", "", file_name)
        file_name += "-predictions.csv"
    with open(file_name, "w") as f:
        f.write("Id,Prediction\n")
        for p_idx, p in enumerate(predictions):
            f.write(str(p_idx + 1))
            f.write(",")
            f.write(str(-1 if p == 0 else 1))
            f.write("\n")
    logger.info("Saved predictions to \"{}\".".format(file_name))

    if args.predict_probabilities:
        file_name = re.sub(r"\.csv", "", file_name)
        file_name += "-probabilities.csv"
        with open(file_name, "w") as f:
            f.write("Id,Prediction\n")
            for p_idx, p in enumerate(prediction_probabilities):
                f.write(str(p_idx + 1))
                f.write(",")
                f.write(str(p))
                f.write("\n")
        logger.info("Saved prediction probabilities to \"{}\".".format(file_name))


def predict_ensemble(args):
    # load data
    data_test = load_data(args.data_path, predict=True, split=(not args.use_bpe))

    # load vocabulary
    vocabulary = construct_vocabulary(args.vocabulary_source_file, vocabulary_size=args.vocabulary_size,
                                      use_bpe=args.use_bpe, bpe_percentage=args.bpe_percentage,
                                      vocabulary_save_file=args.vocabulary_save_file)

    # encode the data, IMPORTANT: need max_length to be specified
    data_test = encode_data(data_test, vocabulary, max_length=args.max_length, use_bpe=args.use_bpe)

    # load all ensemble models
    if os.path.isdir(args.model_load_file):
        model_load_dir = os.path.abspath(args.model_load_file)
    else:
        model_load_dir = os.path.dirname(os.path.abspath(args.model_load_file))
    ensemble = []
    for fn in os.listdir(model_load_dir):
        # might want to make this less restrictive?
        if re.match(r"ensemble_model-\d+-\d+.hdf5", fn):
            model = load_model(os.path.join(model_load_dir, fn))
            ensemble.append(model)
        else:
            print(fn)
    logger.info("Loaded {} ensemble classifiers.".format(len(ensemble)))

    # get predictions
    predictions_hard = []
    predictions_soft = []
    for model_idx, model in enumerate(ensemble):
        logger.info("Predicting using ensemble classifier {:02d}/{:02d}...".format(model_idx + 1, len(ensemble)))
        predictions = model.predict(data_test, batch_size=args.batch_size, verbose=1)
        predictions = predictions.reshape(-1)
        predictions_soft.append(predictions)
        predictions = np.round(predictions).astype(int)
        predictions_hard.append(predictions)

    predictions_hard = np.array(predictions_hard)
    predictions_soft = np.array(predictions_soft)

    # get single prediction using hard voting and soft averaging
    predictions_hard = np.round(np.mean(predictions_hard, axis=0))
    predictions_soft = np.round(np.mean(predictions_soft, axis=0))

    if args.ensemble_soft_voting:
        predictions = predictions_soft
    else:
        predictions = predictions_hard

    # save predictions to file
    if args.prediction_save_file:
        if os.path.isdir(args.prediction_save_file):
            file_name = os.path.join(args.prediction_save_file, "predictions.csv")
        else:
            file_name = args.prediction_save_file + ("" if ".csv" in args.prediction_save_file else ".csv")
    else:
        file_name = os.path.abspath(args.data_path)
        file_name = re.sub(r"\.csv", "", file_name)
        file_name = re.sub(r"\.tsv", "", file_name)
        file_name += "-predictions.csv"
    with open(file_name, "w") as f:
        f.write("Id,Prediction\n")
        for p_idx, p in enumerate(predictions):
            f.write(str(p_idx + 1))
            f.write(",")
            f.write(str(-1 if p == 0 else 1))
            f.write("\n")
    logger.info("Saved predictions to \"{}\".".format(file_name))


def main(args):
    # name stuff
    log_dir = os.path.join(args.log_dir, args.model_name)
    log_file = os.path.join(log_dir, args.model_name + ("_predict" if args.predict else "") + ".log")

    # set up the logger
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(filename)s, line %(lineno)4d] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # create logging file handler
    directory = os.path.dirname(os.path.abspath(log_file))
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    global logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # call the correct functions
    if args.predict:
        if os.path.isfile(args.model_load_file):
            predict(arguments)
        else:
            predict_ensemble(args)
    else:
        if args.ensemble_count <= 1:
            train(args, log_dir)
        else:
            train_ensemble(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # main input: path to the data
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("--validation_data", "-vd", type=str, default=None,
                        help="Data to use for validation (instead of splitting the training data).")

    # arguments related to the embedding layer
    parser.add_argument("--embedding_file", "-ef", type=str, default=None,
                        help="File path of the word embedding vectors to be used (e.g. from GLOVE).")
    parser.add_argument("--embedding_size", "-es", type=int, default=256,
                        help="Size of the BPE embedding matrix. Only used when --use_bpe is specified.")
    parser.add_argument("--embedding_not_trainable", "-ent", action="store_true", default=False,
                        help="If specified, do not train the embedding matrix. Should not be used with BPE.")
    parser.add_argument("--embedding_dropout", "-edo", type=float, default=0.6,
                        help="Dropout rate for the embedding layer.")

    # arguments related to BPE
    parser.add_argument("--use_bpe", "-bpe", action="store_true", default=False,
                        help="If specified, use byte pair encoding instead of pre-trained word embeddings.")
    parser.add_argument("--bpe_percentage", "-pct", type=float, default=0.2,
                        help="Percentage of vocabulary that is BPE. Only used when --use_bpe is specified.")

    # arguments related to the vocabulary and data encoding
    parser.add_argument("--vocabulary_size", "-vs", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--vocabulary_source_file", "-vso", type=str, default=None,
                        help="Vocabulary source file from which to construct or load the vocabulary.")
    parser.add_argument("--vocabulary_save_file", "-vsv", type=str, default=None,
                        help="Vocabulary save file to which to save a constructed vocabulary.")
    parser.add_argument("--max_length", "-ml", type=int, default=None,
                        help="Maximum length of sequences to be fed to the model. NOTE: depends on embedding type!")

    # arguments related to the model
    parser.add_argument("--model_name", "-mn", type=str, default="cnn_rnn",
                        help="Name for the model to be trained, which is used to name log files and checkpoints.")
    parser.add_argument("--model_load_file", "-mlf", type=str, default=None,
                        help="File to load model from for further training or prediction.")
    parser.add_argument("--model_weights_file", "-mwf", type=str, default=None,
                        help="File to load model weights from for further training or prediction.")
    parser.add_argument("--model_config_file", "-mcf", type=str, default=None,
                        help="File to load model config (architecture) from for further training or prediction.")
    parser.add_argument("--model_save_dir", "-msd", type=str, default=None,
                        help="Directory to which to save the final model if not doing CV.")

    # arguments related to the ensemble
    parser.add_argument("--ensemble_count", "-ec", type=int, default=1,
                        help="If more than 1, train an ensemble on the data.")
    parser.add_argument("--ensemble_sample_pct", "-esp", type=float, default=0.8,
                        help="What percentage of the data to draw with replacement for each ensemble classifier.")
    parser.add_argument("--ensemble_soft_voting", "-esv", action="store_true", default=False,
                        help="If specified, use soft instead of hard voting for prediction using an ensemble.")

    # arguments related to predictions
    parser.add_argument("--predict", "-p", action="store_true", default=False,
                        help="If specified, do predictions on the input data.")
    parser.add_argument("--prediction_save_file", "-psf", type=str, default=None,
                        help="File path at which to save predictions.")
    parser.add_argument("--predict_probabilities", "-ppr", action="store_true", default=False,
                        help="If specified, don't predict a label but instead save the probability of class 1.")

    # arguments related to cross validation
    parser.add_argument("--cross_validation", "-cv", action="store_true", default=False,
                        help="If specified, do cross validation instead of a train/validation split.")
    parser.add_argument("--cross_validation_folds", "-cvf", type=int, default=5,
                        help="Number of folds to use in cross validation.")

    # arguments related to training
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="The batch size.")
    parser.add_argument("--epochs", "-ep", type=int, default=10, help="The number of epochs to train the model.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0005, help="The learning rate.")

    # arguments related to the CNN part of the model
    parser.add_argument("--filters", "-f", type=int, nargs="+", default=[64],
                        help="The number of filters for each convolutional layer.")
    parser.add_argument("--kernel_size", "-ks", type=int, nargs="+", default=[5],
                        help="The kernel size for each convolutional layer.")
    parser.add_argument("--pool_size", "-ps", type=int, nargs="+", default=[2],
                        help="The pool size for each max pooling layer (following a convolutional layer).")
    parser.add_argument("--conv_padding", "-cvp", type=str, default="valid", choices=["valid", "same", "causal"],
                        help="The padding to use for convolutional layers.")
    parser.add_argument("--pool_padding", "-pp", type=str, default="valid", choices=["valid", "same", "causal"],
                        help="The padding to use for max pooling layers.")
    parser.add_argument("--conv_dropout", "-cvdo", type=float, default=0.1,
                        help="Dropout rate between the convolutional layers.")

    # arguments related to the RNN part of the model
    parser.add_argument("--cell_type", "-ct", type=str, default="LSTM", help="The type of RNN cell to use.")
    parser.add_argument("--cell_size", "-cs", type=int, default=128, help="The size of the used RNN cells.")
    parser.add_argument("--cell_stack_size", "-css", type=int, default=2, help="The number of RNN cells to stack.")
    parser.add_argument("--rnn_first", "-rf", action="store_true", default=False,
                        help="If specified, pass input through RNN first, then through CNN (default is CNN, then RNN).")
    parser.add_argument("--rnn_dropout", "-rdo", type=float, default=0.3, help="Dropout rate for the RNN output.")

    # arguments related to the dense layers (including the classifier)
    parser.add_argument("--dense_size", "-ds", type=int, nargs="+", default=[],
                        help="The sizes of the dense layers before the classifier.")
    parser.add_argument("--dense_dropout", "-ddo", type=float, nargs="+", default=[0.3],
                        help="Dropout rates for the dense layers.")
    parser.add_argument("--classifier_dropout", "-cldo", type=float, default=0.1,
                        help="Dropout rate for the classifier.")

    # other meta-level arguments
    parser.add_argument("--train_val_split", "-tvs", type=float, default=0.8,
                        help="Fraction of data to be used for training if not doing CV (the rest used for validation).")
    parser.add_argument("--train_on_subset", "-tos", type=float, default=1.0,
                        help="Fraction of data to be used for training. Mostly relevant when using a large data set")
    parser.add_argument("--validate_on_subset", "-vos", type=float, default=1.0,
                        help="Fraction of data to be used for validation. Mostly relevant when using a large data set")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="The number of GPUs to train on.")
    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed to use for splitting data.")
    parser.add_argument("--save_every", "-se", type=int, default=None,
                        help="Frequency at which to save models to checkpoints (default is 10 total checkpoints).")
    parser.add_argument("--checkpoint_dir", "-cpd", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../../checkpoints/"),
                        help="Directory to which checkpoints should be saved during training.")
    parser.add_argument("--log_dir", "-ld", type=str, default=os.path.join(os.path.dirname(__file__), "../../logs"),
                        help="Directory to which both TensorBoard and custom logs should be written.")
    parser.add_argument("--verbosity", "-v", type=int, default=1,
                        help="Verbosity of the output of model training and evaluation (same as Keras verbosity).")

    arguments = parser.parse_args()
    main(arguments)

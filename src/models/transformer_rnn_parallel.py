import sys
import os
import argparse
import logging
import json
import re

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Dropout, Softmax, Conv1D, MaxPooling1D, Activation, Flatten, Bidirectional, LSTM, GRU, Concatenate
from keras.models import Model, save_model, load_model
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model
from keras_transformer.transformer import TransformerBlock
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding

from data import *
from keras_components import *

from typing import *

logger = logging.getLogger()


def build_model(max_length: int,
                embedding_matrix: Union[np.ndarray, Tuple[int]],
                transformer_depth: int,
                transformer_heads: int,
                cell_type: str,
                cell_size: int,
                cell_stack_size: int,
                filters: List[int],
                kernel_size: List[int],
                pool_size: List[int],
                conv_padding: str,
                pool_padding: str,
                dense_size: List[int],
                loaded_model: Optional[str] = None,
                l2_penalty: Optional[float] = None,
                embedding_dropout: float = 0.6,
                transformer_dropout: float = 0.1,
                conv_dropout: float = 0.1,
                dense_dropout: Union[float, List[float]] = 0.3,
                classifier_dropout: float = 0.1,
                no_cnn: bool = False,
                train_lm: bool = True) -> Model:

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

    if not no_cnn and len(filters) != len(kernel_size) or len(filters) != len(pool_size) or len(kernel_size) != len(pool_size):
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

    # regularizer for embedding layer
    l2_regularizer = l2(l2_penalty) if l2_penalty else None

    # input encoded as integers
    raw_input = Input(shape=(max_length,), name="input")

    # embedding layer, initialised with embedding matrix weights for now
    embedding_layer = ReusableEmbedding(
        input_dim=(embedding_matrix[0] if type(embedding_matrix) == tuple else embedding_matrix.shape[0]),
        output_dim=(embedding_matrix[1] if type(embedding_matrix) == tuple else embedding_matrix.shape[1]),
        input_length=max_length,
        name="word_embedding",
        weights=(None if type(embedding_matrix) == tuple else [embedding_matrix]),
        embeddings_regularizer=l2_regularizer)

    # "transpose" of embedding matrix to map back to vocabulary
    lm_output = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name="word_prediction_logits")

    # transformer as taken from here: https://github.com/kpot/keras-transformer/blob/master/example/models.py
    position_embedding = TransformerCoordinateEmbedding(
        max_transformer_depth=1,
        name="position_embedding")

    transformer_blocks = []
    for i in range(transformer_depth):
        block_name = "transformer" + str(i)

        # define transformer block
        transformer_block = TransformerBlock(
            name=block_name,
            num_heads=transformer_heads,
            residual_dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
            use_masking=True,
            vanilla_wiring=True)

        transformer_blocks.append(transformer_block)

    # nothing special to load for softmax
    lm_prediction = Softmax(name="word_predictions")

    # RNN
    rnn_cells = []
    for i in range(cell_stack_size - 1):
        cell_name = "{}_{}".format(cell_type_name, i)
        cell = Bidirectional(cell_type(cell_size, return_sequences=True, name=cell_name),
                             name="bidirectional_{}".format(cell_name))
        rnn_cells.append(cell)

    cell_name = "{}_{}".format(cell_type_name, cell_stack_size - 1)
    cell = Bidirectional(cell_type(cell_size, return_sequences=(not no_cnn), name=cell_name),
                         name="bidirectional_{}".format(cell_name))
    rnn_cells.append(cell)

    # flattening for RNN/transformer outputs and CNN outputs
    flatten = Flatten(name="flatten")

    # convolution layer(s)
    conv_dropout = Dropout(conv_dropout, name="conv_dropout")
    conv_layers = []
    for i in range(len(filters)):
        # construct and possibly load convolutional layer
        conv_layer_name = "conv_{}".format(i)
        convolution = Conv1D(filters[i], kernel_size[i], padding=conv_padding,
                             activation="relu", name=conv_layer_name)

        # construct max pooling, no weights to load
        pooling = MaxPooling1D(pool_size[i], padding=pool_padding, name="max_pool_{}".format(i))

        conv_layers.append((convolution, pooling))

    # dense layer(s)
    dense_layers = []
    for i in range(len(dense_size)):
        # nothing to load for dropout
        dropout = Dropout(rate=dense_dropout[i], name="dense_dropout_{}".format(i))

        # construct and possibly load dense layer
        dense_layer_name = "dense_{}".format(i)
        dense = Dense(dense_size[i], name=dense_layer_name)

        # get output
        dense_layers.append((dropout, dense))

    # classification layer
    classifier_dropout = Dropout(classifier_dropout, name="classifier_dropout")
    classifier = Dense(1, name="classifier")
    classifier_prediction = Activation("sigmoid", name="classifier_prediction")

    # embedding used for both parts of the model
    embedded, embedding_matrix = embedding_layer(raw_input)

    # BUILD THE ACTUAL MODEL
    # transformer
    transformer_output = position_embedding(embedded, step=0)
    for tb in transformer_blocks:
        transformer_output = tb(transformer_output)

    # language model loss if it is used
    lm_output = lm_output([transformer_output, embedding_matrix])
    lm_output = lm_prediction(lm_output)

    # RNN
    rnn_output = embedded
    for rc in rnn_cells:
        rnn_output = rc(rnn_output)

    # handle concatenation of both outputs
    if no_cnn:
        # flatten both and concatenate them
        concat = Concatenate(name="concat_flattened")
        classifier_output = concat([flatten(transformer_output), flatten(rnn_output)])
    else:
        # concatenate the extracted features
        concat = Concatenate(name="concat_features")
        classifier_output = concat([transformer_output, rnn_output])

        # pass through CNN
        for cl in conv_layers:
            classifier_output = cl[1](cl[0](conv_dropout(classifier_output)))

    # pass through dense layer(s)
    classifier_output = flatten(classifier_output)
    for dl in dense_layers:
        classifier_output = dl[1](dl[0](classifier_output))

    # do classification
    classifier_output = classifier_prediction(classifier(classifier_dropout(classifier_output)))

    if train_lm:
        m = Model(inputs=raw_input, outputs=[lm_output, classifier_output])
    else:
        m = Model(inputs=raw_input, outputs=classifier_output)
    return m


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

    train_lm = args.lm_loss_weight != 0.0

    # create the model
    optimizer = None
    if args.optimizer.lower() == "adam":
        # optimizer = Adam(lr=args.learning_rate, clipvalue=5.0)
        optimizer = Adam(lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        # same as below
        optimizer = SGD(lr=args.learning_rate, momentum=args.momentum, nesterov=True, clipvalue=5.0)
    elif args.optimizer.lower() == "adadelta":
        # pretty steady loss decline but training and eval losses behave very differently
        optimizer = Adadelta(lr=args.learning_rate, clipvalue=5.0)
    elif args.optimizer.lower() == "nadam":
        optimizer = Nadam(lr=args.learning_rate, schedule_decay=0.0005, clipvalue=5.0)

    if args.model_load_file:
        model = load_model(args.model_load_file,
                           custom_objects={"perplexity": perplexity,
                                           "lm_accuracy": lm_accuracy})
    else:
        model = build_model(max_length=args.max_length,
                            embedding_matrix=embedding,
                            transformer_depth=args.transformer_depth,
                            transformer_heads=args.transformer_heads,
                            cell_type=args.cell_type,
                            cell_size=args.cell_size,
                            cell_stack_size=args.cell_stack_size,
                            filters=args.filters,
                            kernel_size=args.kernel_size,
                            pool_size=args.pool_size,
                            conv_padding=args.conv_padding,
                            pool_padding=args.pool_padding,
                            dense_size=args.dense_size,
                            l2_penalty=args.l2_penalty,
                            embedding_dropout=args.embedding_dropout,
                            transformer_dropout=args.transformer_dropout,
                            conv_dropout=args.conv_dropout,
                            dense_dropout=args.dense_dropout,
                            classifier_dropout=args.classifier_dropout,
                            no_cnn=args.no_cnn,
                            train_lm=train_lm)

    if args.gpus > 1:
        logger.warning("Using {} GPUs. This means that batches are divided up between the GPUs!".format(args.gpus))
        model = multi_gpu_model(model, gpus=args.gpus)

    if train_lm:
        # need one classification loss and one language model loss
        model.compile(optimizer=optimizer,
                      loss={"classifier_prediction": binary_crossentropy,
                            "word_predictions": sparse_categorical_crossentropy},
                      loss_weights={"classifier_prediction": args.classifier_loss_weight,
                                    "word_predictions": args.lm_loss_weight},
                      metrics={"classifier_prediction": ["accuracy"],
                               "word_predictions": [perplexity]})
    else:
        # only need classification loss
        model.compile(optimizer=optimizer,
                      loss={"classifier_prediction": binary_crossentropy},
                      metrics={"classifier_prediction": ["accuracy"]})

    if 2 != args.verbosity > 0:
        model.summary()

    # define all the callbacks during training, starting with the LR scheduler
    cb = []
    lr_scheduler = LearningRateScheduler(CosineLRSchedule(lr_high=args.learning_rate,
                                                          lr_low=args.learning_rate / 32,
                                                          initial_period=args.epochs), verbose=1)
    cb.append(lr_scheduler)

    # stuff for saving models and displaying progress
    directory = os.path.dirname(os.path.abspath(checkpoint_template))
    if not os.path.exists(directory):
        os.makedirs(directory)
    period = args.save_every if args.save_every else max(int(args.epochs / 10), 1)
    checkpoint = ModelCheckpoint(checkpoint_template, monitor="loss", verbose=1,
                                 save_best_only=True, mode="min", period=period)

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


def train(args):
    if args.model_save_dir:
        directory = os.path.abspath(args.model_save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        json_file = os.path.join(args.model_save_dir, args.model_name + ".json")
        args_dict = vars(args)
        with open(json_file, "w") as f:
            json.dump(args_dict, f)

    if not args.vocabulary_source_file:
        args.vocabulary_source_file = args.data_path

    # name stuff
    checkpoint_template = os.path.join(args.checkpoint_dir, args.model_name, "model-{epoch:03d}.hdf5")
    log_dir = os.path.join(args.log_dir, args.model_name)
    log_file = os.path.join(log_dir, args.model_name + ".log")

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

    # load data
    data, labels = load_data(args.data_path, split=(not args.use_bpe))

    # potentially sample from the data set to reduce its size
    if args.train_on_subset < 1.0:
        data, _, labels, _ = split_data(data, labels, train_val_split=args.train_on_subset,
                                        random_seed=args.random_seed)

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

    # encode data (and get LM targets)
    data, targets = encode_data(data, vocabulary, labels=labels, max_length=args.max_length,
                                use_bpe=args.use_bpe, for_classification=True)
    if not args.max_length:
        args.max_length = data.shape[1]
        logger.debug("Maximum length of the encoded data is {}.".format(args.max_length))
    if args.lm_loss_weight == 0.0:
        targets = [targets[1]]

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

        if args.lm_loss_weight == 0.0:
            targets_val = [targets_val[1]]

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
        for i, (train_ind, val_ind) in enumerate(split.split(data, targets[1])):
            # get the data
            data_train = data[train_ind]
            targets_train = [t[train_ind] for t in targets]
            data_val = data[val_ind]
            targets_val = [t[val_ind] for t in targets]

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


def predict_after_training(model, data, file_name, splits=10):
    split_length = int(len(data[0]) / splits)
    split_indices = [0] + [i * split_length for i in range(1, splits)] + [len(data[0])]

    if ".csv" not in file_name:
        file_name += ".csv"

    directory = os.path.dirname(os.path.abspath(file_name))
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.info("Predicting in splits...")
    with open(file_name, "a") as f:
        sum = 0
        for i in range(1, len(split_indices)):
            current_data = [d[split_indices[i - 1]:split_indices[i]] for d in data]
            predictions = model.predict(current_data, batch_size=1, verbose=0)
            predictions = np.argmax(predictions[0], axis=-1)
            sum += np.sum(predictions)
            for label in predictions:
                f.write("{}\n".format(label + 1))


def predict(args):
    # load data
    data_test = load_data(args.data_path, predict=False, split=(not args.use_bpe))

    # load vocabulary
    vocabulary = construct_vocabulary(args.vocabulary_source_file or data_test, vocabulary_size=args.vocabulary_size,
                                      use_bpe=args.use_bpe, bpe_percentage=args.bpe_percentage,
                                      vocabulary_save_file=args.vocabulary_save_file)

    # encode the data, IMPORTANT: need max_length to be specified
    data_test = encode_data(data_test, vocabulary, max_length=args.max_length,
                            use_bpe=args.use_bpe, for_classification=args.fine_tune_model)

    # load the model
    model = load_model(args.model_load_file, custom_objects={"perplexity": perplexity,
                                                             "lm_accuracy": lm_accuracy})

    # do the predictions:
    predictions = model.predict(data_test, batch_size=1, verbose=0)
    predictions = np.argmax(predictions[0], axis=-1)

    # save predictions to file
    file_name = os.path.abspath(args.data_path)
    file_name = re.sub(r"\.csv", "", file_name)
    file_name = re.sub(r"\.tsv", "", file_name)
    file_name += "-predictions.csv"
    with open(file_name, "w") as f:
        f.write("Id,Prediction\n")
        for p_idx, p in enumerate(predictions):
            f.write(str(p_idx))
            f.write(",")
            f.write(str(-1 if p == 0 else 1))
            f.write("\n")
    logger.info("Saved predictions to \"{}\".".format(file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("--validation_data", "-vd", type=str, default=None,
                        help="Data to use for validation (instead of splitting the training data).")
    parser.add_argument("--embedding_file", "-ef", type=str, default=None,
                        help="File path of the word embedding vectors to be used (e.g. from GLOVE).")
    parser.add_argument("--embedding_size", "-es", type=int, default=256,
                        help="Size of the BPE embedding matrix. Only used when --use_bpe is specified.")
    parser.add_argument("--model_name", "-mn", type=str, default="transformer",
                        help="Name for the model to be trained, which is used to name log files and checkpoints.")
    parser.add_argument("--model_load_file", "-mlf", type=str, default=None,
                        help="File to load a model from for further training.")
    parser.add_argument("--model_save_dir", "-msd", type=str, default=None,
                        help="Directory to which to save the final model if not doing CV.")
    parser.add_argument("--predict", "-p", action="store_true", default=False,
                        help="If specified, do predictions on the provided test data (--prediction_data).")
    parser.add_argument("--use_bpe", "-bpe", action="store_true", default=False,
                        help="If specified, use byte pair encoding instead of pre-trained word embeddings.")
    parser.add_argument("--cross_validation", "-cv", action="store_true", default=False,
                        help="If specified, do cross validation instead of a train/validation split.")
    parser.add_argument("--no_cnn", "-nc", action="store_true", default=False,
                        help="If specified, do not construct a CNN on top of the Transformer+RNN output.")
    parser.add_argument("--cross_validation_folds", "-cvf", type=int, default=5,
                        help="Number of folds to use in cross validation.")
    parser.add_argument("--bpe_percentage", "-pct", type=float, default=0.2,
                        help="Percentage of vocabulary that is BPE. Only used when --use_bpe is specified.")
    parser.add_argument("--vocabulary_size", "-vs", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--vocabulary_source_file", "-vso", type=str, default=None,
                        help="Vocabulary source file from which to construct or load the vocabulary.")
    parser.add_argument("--vocabulary_save_file", "-vsv", type=str, default=None,
                        help="Vocabulary save file to which to save a constructed vocabulary.")
    parser.add_argument("--max_length", "-ml", type=int, default=None,
                        help="Maximum length of sequences to be fed to the model. NOTE: depends on embedding type!")
    parser.add_argument("--train_val_split", "-tvs", type=float, default=0.8,
                        help="Fraction of data to be used for training if not doing CV (the rest used for validation).")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="The number of GPUs to train on.")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="The batch size.")
    parser.add_argument("--epochs", "-ep", type=int, default=10, help="The number of epochs to train the model.")
    parser.add_argument("--save_every", "-se", type=int, default=None,
                        help="Frequency at which to save models to checkpoints (default is 10 total checkpoints).")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0005, help="The learning rate.")
    parser.add_argument("--optimizer", "-opt", type=str, default="adam", help="The optimizer to use.")
    parser.add_argument("--momentum", "-mom", type=float, default=0.9, help="The momentum value to use for SGD.")
    parser.add_argument("--transformer_depth", "-tfd", type=int, default=8,
                        help="Number of transformer blocks in the model.")
    parser.add_argument("--transformer_heads", "-tfh", type=int, default=8,
                        help="Number of heads of each transformer block. "
                             "The embeddings size needs to be divisible by this.")
    parser.add_argument("--cell_type", "-ct", type=str, default="LSTM", help="The type of RNN cell to use.")
    parser.add_argument("--cell_size", "-cs", type=int, default=128, help="The size of the used RNN cells.")
    parser.add_argument("--cell_stack_size", "-css", type=int, default=2, help="The number of RNN cells to stack.")
    parser.add_argument("--filters", "-f", type=int, nargs="+", default=[256],
                        help="The number of filters for each convolutional layer.")
    parser.add_argument("--kernel_size", "-ks", type=int, nargs="+", default=[5],
                        help="The kernel size for each convolutional layer.")
    parser.add_argument("--pool_size", "-ps", type=int, nargs="+", default=[2],
                        help="The pool size for each max pooling layer (following a convolutional layer).")
    parser.add_argument("--conv_padding", "-cvp", type=str, default="valid", choices=["valid", "same", "causal"],
                        help="The padding to use for convolutional layers.")
    parser.add_argument("--pool_padding", "-pp", type=str, default="valid", choices=["valid", "same", "causal"],
                        help="The padding to use for max pooling layers.")
    parser.add_argument("--dense_size", "-ds", type=int, nargs="+", default=[],
                        help="The sizes of the dense layers before the classifier.")
    parser.add_argument("--classifier_loss_weight", "-clw", type=float, default=1.0,
                        help="Weight for the classification loss when finetuning the model.")
    parser.add_argument("--lm_loss_weight", "-llw", type=float, default=0.25,
                        help="Weight for the language model (word prediction) loss when finetuning the model.")
    parser.add_argument("--l2_penalty", "-l2", type=float, default=0.0000001,
                        help="L2 penalty on weights in the transformer blocks (?).")
    parser.add_argument("--embedding_dropout", "-edo", type=float, default=0.6,
                        help="Dropout rate for the embedding layer.")
    parser.add_argument("--transformer_dropout", "-tdo", type=float, default=0.1,
                        help="Dropout rate for the weights in the transformer blocks.")
    parser.add_argument("--conv_dropout", "-cvdo", type=float, default=0.1,
                        help="Dropout rate between the convolutional layers.")
    parser.add_argument("--dense_dropout", "-ddo", type=float, nargs="+", default=[0.3],
                        help="Dropout rates for the dense layers.")
    parser.add_argument("--classifier_dropout", "-cdo", type=float, default=0.1,
                        help="Dropout rate for the classifier.")
    parser.add_argument("--train_on_subset", "-tos", type=float, default=1.0,
                        help="Fraction of data to be used. Mostly relevant when using a large data set")
    parser.add_argument("--random_seed", "-rs", type=int, default=42, help="Random seed to use for splitting data.")
    parser.add_argument("--checkpoint_dir", "-cpd", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../../checkpoints/"),
                        help="Directory to which checkpoints should be saved during training.")
    parser.add_argument("--log_dir", "-ld", type=str, default=os.path.join(os.path.dirname(__file__), "../../logs"),
                        help="Directory to which both TensorBoard and custom logs should be written.")
    parser.add_argument("--verbosity", "-v", type=int, default=1,
                        help="Verbosity of the output of model training and evaluation (same as Keras verbosity).")

    arguments = parser.parse_args()
    if arguments.predict:
        predict(arguments)
    else:
        train(arguments)

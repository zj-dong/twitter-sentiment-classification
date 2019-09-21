import sys
import os
import argparse
import logging
import json
import re

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Dropout, Softmax, Flatten, Lambda
from keras.models import Model, save_model, load_model
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model
from keras_transformer.transformer import TransformerBlock
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention

from data import *
from keras_components import *

logger = logging.getLogger()


def build_model(max_length,
                loaded_model=None,
                fine_tune_model=False,
                embedding_matrix=None,
                transformer_depth=8,
                transformer_heads=8,
                l2_penalty=None,
                embedding_dropout=0.6,
                transformer_dropout=0.1,
                classifier_dropout=0.1,
                transformer_output_handling="flatten",
                print_info=False,
                train_lm=True):

    original_model = None
    if loaded_model:
        # load the specified model
        original_model = load_model(loaded_model,
                                    custom_objects={"perplexity": perplexity,
                                                    "lm_accuracy": lm_accuracy,
                                                    "SeqSelfAttention": SeqSelfAttention,
                                                    "ScaledDotProductAttention": ScaledDotProductAttention})

    # regularizer for embedding layer
    l2_regularizer = l2(l2_penalty) if l2_penalty else None

    # input encoded as integers
    raw_input = Input(shape=(max_length,), name="input")

    # embedding layer, initialised with embedding matrix weights for now
    embedding_weights = [original_model.get_layer(name="word_embedding").get_weights()[0]
                         if loaded_model else embedding_matrix]
    embedding_layer = ReusableEmbedding(
        input_dim=(embedding_matrix[0] if type(embedding_matrix) == tuple else embedding_matrix.shape[0]),
        output_dim=(embedding_matrix[1] if type(embedding_matrix) == tuple else embedding_matrix.shape[1]),
        input_length=max_length,
        name="word_embedding",
        weights=(None if type(embedding_matrix) == tuple and not loaded_model else embedding_weights),
        embeddings_regularizer=l2_regularizer)

    # "transpose" of embedding matrix to map back to vocabulary
    if loaded_model:
        output_weights = original_model.get_layer(name="word_prediction_logits").get_weights()
        output_layer = TiedOutputEmbedding(
            projection_regularizer=l2_regularizer,
            projection_dropout=embedding_dropout,
            name="word_prediction_logits",
            weights=output_weights)
    else:
        output_layer = TiedOutputEmbedding(
            projection_regularizer=l2_regularizer,
            projection_dropout=embedding_dropout,
            name="word_prediction_logits")

    # transformer as taken from here: https://github.com/kpot/keras-transformer/blob/master/example/models.py
    if loaded_model:
        position_weights = original_model.get_layer(name="position_embedding").get_weights()
        position_embedding = TransformerCoordinateEmbedding(
            max_transformer_depth=1,
            name="position_embedding",
            weights=position_weights)
    else:
        position_embedding = TransformerCoordinateEmbedding(
            max_transformer_depth=1,
            name="position_embedding")

    transformer_input, embedding_matrix = embedding_layer(raw_input)
    transformer_output = position_embedding(transformer_input, step=0)
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

        # build the layers in the block because apparently you have to do that
        if loaded_model:
            if i == 0:
                transformer_block.attention_layer.build(original_model.get_layer("position_embedding").output_shape)
            else:
                transformer_block.attention_layer.build(
                    original_model.get_layer("transformer{}_normalization2".format(i - 1)).output_shape)
            transformer_block.norm1_layer.build(original_model.get_layer(block_name + "_self_attention").output_shape)
            transformer_block.norm2_layer.build(original_model.get_layer(block_name + "_normalization1").output_shape)
            transformer_block.transition_layer.build(
                original_model.get_layer(block_name + "_normalization1").output_shape)

            # set weights for all the contained layers manually
            transformer_block.attention_layer.set_weights(
                original_model.get_layer(name=(block_name + "_self_attention")).get_weights())
            transformer_block.norm1_layer.set_weights(
                original_model.get_layer(name=(block_name + "_normalization1")).get_weights())
            transformer_block.norm2_layer.set_weights(
                original_model.get_layer(name=(block_name + "_normalization2")).get_weights())
            transformer_block.transition_layer.set_weights(
                original_model.get_layer(name=(block_name + "_transition")).get_weights())

        # pass output of last layer through transformer
        transformer_output = transformer_block(transformer_output)

    if print_info:
        logger.debug("transformer_output shape: {}".format(
            K.int_shape(transformer_output[0] if fine_tune_model else transformer_output)))

    # nothing special to load for softmax
    softmax_layer = Softmax(name="word_predictions")
    lm_output_logits = output_layer([transformer_output, embedding_matrix])
    lm_output = softmax_layer(lm_output_logits)
    if print_info:
        logger.debug("lm_output_logits shape: {}".format(K.int_shape(lm_output_logits)))
        logger.debug("output shape: {}".format(K.int_shape(lm_output)))

    if not fine_tune_model:
        m = Model(inputs=raw_input, outputs=lm_output)
        return m

    loaded_layer_names = []
    if loaded_model:
        loaded_layer_names = [layer.name for layer in original_model.layers]

    # for concatenation transformer outputs early
    flatten = Flatten(name="flatten_transformer_output")
    max_pooling = Lambda(lambda x: K.max(x, axis=1), name="max_pooling")
    mean_pooling = Lambda(lambda x: K.mean(x, axis=1), name="mean_pooling")
    self_attention = SeqSelfAttention(name="self_attention")
    scaled_dot_attention = ScaledDotProductAttention(name="scaled_dot_attention")
    dropout = Dropout(rate=classifier_dropout, name="classifier_dropout")
    options = {
        "flatten": flatten,
        "max_pooling": max_pooling,
        "mean_pooling": mean_pooling,
        "self_attention": self_attention,
        "scaled_dot_attention": scaled_dot_attention
    }

    dense = Dense(2, activation=None, name="dense")
    if loaded_model and "dense" in loaded_layer_names:
        layer = original_model.get_layer(name="dense")
        dense.build(layer.input_shape)
        dense.set_weights(layer.get_weights())

    pooling_layer = options[transformer_output_handling]
    if loaded_model and transformer_output_handling in loaded_layer_names:
        layer = original_model.get_layer(name=transformer_output_handling)
        pooling_layer.build(layer.input_shape)
        pooling_layer.set_weights(layer.get_weights())

    if "attention" in transformer_output_handling:
        handled_output = flatten(pooling_layer(transformer_output))
    else:
        handled_output = pooling_layer(transformer_output)

    classifier_logits = dense(dropout(handled_output))
    classifier_output = Softmax(name="classifier_prediction")(classifier_logits)

    if train_lm:
        m = Model(inputs=raw_input, outputs=[lm_output, classifier_output])
    else:
        m = Model(inputs=raw_input, outputs=classifier_output)
    # m = Model(inputs=raw_input, outputs=lm_output)
    return m


def build_and_train(args,
                    checkpoint_template,
                    log_dir,
                    embedding,
                    data_train,
                    targets_train,
                    data_val=None,
                    targets_val=None,
                    cv_fold=None):

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

    if args.model_load_file and (not args.fine_tune_model or args.load_extended):
        model = load_model(args.model_load_file,
                           custom_objects={"perplexity": perplexity,
                                           "lm_accuracy": lm_accuracy,
                                           "SeqSelfAttention": SeqSelfAttention,
                                           "ScaledDotProductAttention": ScaledDotProductAttention})
    else:
        model = build_model(max_length=args.max_length,
                            loaded_model=args.model_load_file,
                            fine_tune_model=args.fine_tune_model,
                            embedding_matrix=embedding,
                            transformer_depth=args.transformer_depth,
                            transformer_heads=args.transformer_heads,
                            l2_penalty=args.l2_penalty,
                            embedding_dropout=args.embedding_dropout,
                            transformer_dropout=args.transformer_dropout,
                            transformer_output_handling=args.transformer_output_handling,
                            print_info=True, train_lm=train_lm)

    if args.gpus > 1:
        logger.warning("Using {} GPUs. This means that batches are divided up between the GPUs!".format(args.gpus))
        model = multi_gpu_model(model, gpus=args.gpus)

    # losses depend on what model is loaded
    if not args.fine_tune_model:
        # only need one loss for the language model
        model.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy, metrics=[perplexity, lm_accuracy])
    else:
        # need one classification loss and two language model losses
        if train_lm:
            model.compile(optimizer=optimizer,
                          loss={"classifier_prediction": sparse_categorical_crossentropy,
                                "word_predictions": sparse_categorical_crossentropy},
                          loss_weights={"classifier_prediction": args.classifier_loss_weight,
                                        "word_predictions": args.lm_loss_weight},
                          metrics={"classifier_prediction": ["accuracy"],
                                   "word_predictions": [perplexity]})
        else:
            model.compile(optimizer=optimizer,
                          loss={"classifier_prediction": sparse_categorical_crossentropy},
                          metrics={"classifier_prediction": ["accuracy"]})
        """
        model.compile(optimizer=optimizer,
                      loss={"word_predictions": sparse_categorical_crossentropy},
                      metrics={"word_predictions": [perplexity, lm_accuracy]})
        """
    if 2 != args.verbosity > 0:
        model.summary()

    # use different learning rate schedulers for the different models
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

    if args.fine_tune_model and "finetune" not in args.model_name:
        args.model_name += "-finetuned"

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
                                use_bpe=args.use_bpe, for_classification=args.fine_tune_model)
    if not args.max_length:
        args.max_length = data.shape[1]
        logger.debug("Maximum length of the encoded data is {}.".format(args.max_length))
    if not args.fine_tune_model:
        targets = [targets[0]]
    elif args.lm_loss_weight == 0.0:
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

        if not args.fine_tune_model:
            targets_val = [targets_val[0]]
        elif args.lm_loss_weight == 0.0:
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
        log_string = "Saving model to \"{}\"{}".format(
            model_save_file, "after finetuning." if args.fine_tune_model else ".")
        if args.model_load_file:
            log_string += " Model was previously loaded from \"{}\".".format(args.model_load_file)
        logger.info(log_string)
        save_model(model, model_save_file)


def predict_in_splits(args, model, data_test):
    splits = args.prediction_splits
    split_length = int(len(data_test) / splits)
    split_indices = [0] + [i * split_length for i in range(1, splits)] + [len(data_test)]

    logger.info("Predicting in splits...")
    all_prediction_probabilities = []
    all_predictions = []
    for i in tqdm(range(1, len(split_indices))):
        current_data = data_test[split_indices[i - 1]:split_indices[i]]
        predictions = model.predict(current_data, batch_size=args.batch_size, verbose=1)
        predictions = predictions[1]
        prediction_probabilities = predictions[:, 1]
        predictions = np.round(prediction_probabilities).astype(int)
        all_prediction_probabilities.append(prediction_probabilities)
        all_predictions.append(predictions)

    return np.concatenate(all_predictions), np.concatenate(all_prediction_probabilities)


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
    model = load_model(args.model_load_file, custom_objects={"perplexity": perplexity, "lm_accuracy": lm_accuracy})

    # do the predictions:
    if args.predict_in_splits:
        predictions, prediction_probabilities = predict_in_splits(args, model, data_test)
    else:
        predictions = model.predict(data_test, batch_size=args.batch_size, verbose=1)
        predictions = predictions[1][:, 1]
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
    parser.add_argument("--model_name", "-mn", type=str, default="transformer",
                        help="Name for the model to be trained, which is used to name log files and checkpoints.")
    parser.add_argument("--model_load_file", "-mlf", type=str, default=None,
                        help="File to load a model from for further training.")
    parser.add_argument("--model_save_dir", "-msd", type=str, default=None,
                        help="Directory to which to save the final model if not doing CV.")

    # arguments related to predictions
    parser.add_argument("--predict", "-p", action="store_true", default=False,
                        help="If specified, do predictions on the input data.")
    parser.add_argument("--prediction_save_file", "-psf", type=str, default=None,
                        help="File path at which to save predictions.")
    parser.add_argument("--predict_probabilities", "-ppr", action="store_true", default=False,
                        help="If specified, don't predict a label but instead save the probability of class 1.")
    parser.add_argument("--predict_in_splits", "-pis", action="store_true", default=False,
                        help="If specified, split data into chunks for prediction (for memory issues).")
    parser.add_argument("--prediction_splits", "-psp", type=int, default=50,
                        help="Number of chunks to split the data when predicting.")

    # arguments related to cross validation
    parser.add_argument("--cross_validation", "-cv", action="store_true", default=False,
                        help="If specified, do cross validation instead of a train/validation split.")
    parser.add_argument("--cross_validation_folds", "-cvf", type=int, default=5,
                        help="Number of folds to use in cross validation.")

    # arguments related to training
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="The batch size.")
    parser.add_argument("--epochs", "-ep", type=int, default=10, help="The number of epochs to train the model.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0005, help="The learning rate.")
    parser.add_argument("--optimizer", "-opt", type=str, default="adam", help="The optimizer to use.")
    parser.add_argument("--momentum", "-mom", type=float, default=0.9, help="The momentum value to use for SGD.")
    parser.add_argument("--classifier_loss_weight", "-clw", type=float, default=1.0,
                        help="Weight for the classification loss when finetuning the model.")
    parser.add_argument("--lm_loss_weight", "-llw", type=float, default=0.25,
                        help="Weight for the language model (word prediction) loss when finetuning the model.")

    # arguments related to the architecture
    parser.add_argument("--transformer_depth", "-tfd", type=int, default=8,
                        help="Number of transformer blocks in the model.")
    parser.add_argument("--transformer_heads", "-tfh", type=int, default=8,
                        help="Number of heads of each transformer block. "
                             "The embeddings size needs to be divisible by this.")
    parser.add_argument("--transformer_output_handling", "-toh", type=str, default="flatten",
                        choices=["flatten", "max_pooling", "mean_pooling", "self_attention", "scaled_dot_attention"],
                        help="How to deal with the transformer output when training a classifier.")
    parser.add_argument("--transformer_dropout", "-tdo", type=float, default=0.1,
                        help="Dropout rate for the weights in the transformer blocks.")
    parser.add_argument("--l2_penalty", "-l2", type=float, default=0.0000001,
                        help="L2 penalty on weights in the transformer blocks (?).")
    # the argument below is essentially deprecated for our use without generative pre-training, fine_tune_model should
    # always be true and load_extended is redundant since all models are already "extended" with a classifier
    parser.add_argument("--fine_tune_model", "-ftm", action="store_true", default=False,
                        help="If specified, train a classifier on top of the language model architecture.")
    parser.add_argument("--load_extended", "-lom", action="store_true", default=False,
                        help="Signals that instead of loading an LM architecture, a model with a classifier is loaded.")

    # argument related to the classifier
    parser.add_argument("--classifier_dropout", "-cdo", type=float, default=0.1,
                        help="Dropout rate for the classifier.")

    # other meta-level arguments
    parser.add_argument("--train_val_split", "-tvs", type=float, default=0.8,
                        help="Fraction of data to be used for training if not doing CV (the rest used for validation).")
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
    if arguments.predict:
        predict(arguments)
    else:
        train(arguments)

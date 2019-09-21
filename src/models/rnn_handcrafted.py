import sys
import os
import argparse
import logging
import json

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, GRU, Concatenate, Bidirectional, Activation
from keras.models import Model, save_model, load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.utils import multi_gpu_model

from data import *
from keras_components import *

from typing import *

logger = logging.getLogger()


def build_model(max_length: int,
                handcrafted_size: int,
                cell_type: str,
                cell_size: int,
                cell_stack_size: int,
                dense_size: List[int],
                embedding_matrix: Union[np.ndarray, Tuple[int]] = None,
                embedding_dropout: float = 0.6,
                dense_dropout: Union[float, List[float]] = 0.3,
                classifier_dropout: float = 0.1) -> Model:

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

    cell_type_name = cell_type.lower()
    if cell_type_name == "lstm":
        cell_type = LSTM
    elif cell_type_name == "gru":
        cell_type = GRU

    # input 1: word indices, input 2: handcrafted_features
    word_input = Input(shape=(max_length,), name="word_input")
    handcrafted_input = Input(shape=(handcrafted_size,), name="handcrafted_input")

    # embedding layer
    embedding_layer = Embedding(
        input_dim=(embedding_matrix[0] if type(embedding_matrix) == tuple else embedding_matrix.shape[0]),
        output_dim=(embedding_matrix[1] if type(embedding_matrix) == tuple else embedding_matrix.shape[1]),
        input_length=max_length,
        name="word_embedding",
        weights=(None if type(embedding_matrix) == tuple else [embedding_matrix]))
    embedding_dropout = Dropout(embedding_dropout, name="embedding_dropout")

    # bidirectional RNN
    rnn_cells = []
    for i in range(cell_stack_size - 1):
        rnn_cells.append(Bidirectional(cell_type(cell_size, return_sequences=True, name="{}_{}".format(cell_type_name, i))))
    rnn_cells.append(Bidirectional(cell_type(cell_size, name="{}_{}".format(cell_type_name, cell_stack_size - 1))))

    # concatenating RNN output and handcrafted features
    concat = Concatenate(name="rnn_handcrafted_concat")

    # dense layer(s)
    dense_layers = []
    for i in range(len(dense_size)):
        dense_layers.append((Dropout(rate=dense_dropout[i], name="dense_dropout_{}".format(i)),
                             Dense(dense_size[i], name="dense_{}".format(i))))

    # just the classifier layer for now, might add more dense layers
    classifier_dropout = Dropout(classifier_dropout, name="classifier_dropout")
    classifier = Dense(1, name="classifier")
    classifier_prediction = Activation("sigmoid", name="classifier_prediction")

    # build the actual model
    output = embedding_dropout(embedding_layer(word_input))
    for c in rnn_cells:
        output = c(output)
    output = concat([output, handcrafted_input])
    for l in dense_layers:
        output = l[1](l[0](output))
    output = classifier_prediction(classifier(classifier_dropout(output)))
    model = Model(inputs=[word_input, handcrafted_input], outputs=output)

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
                            handcrafted_size=data_train[1].shape[-1],
                            cell_type=args.cell_type,
                            cell_size=args.cell_size,
                            cell_stack_size=args.cell_stack_size,
                            dense_size=args.dense_size,
                            embedding_matrix=embedding,
                            embedding_dropout=args.embedding_dropout,
                            dense_dropout=args.dense_dropout,
                            classifier_dropout=args.classifier_dropout)

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
    data, targets = load_data(args.data_path, split=(not args.use_bpe))

    # load handcrafted features
    handcrafted_features = load_handcrafted_features(args.handcrafted_path)

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
    data = [data, handcrafted_features]

    # split data
    data_train = data_val = targets_train = targets_val = split = None
    split = split_data(data, targets,
                       cross_validation=args.cross_validation,
                       train_val_split=args.train_val_split,
                       cv_folds=args.cross_validation_folds,
                       random_seed=args.random_seed)
    if not args.cross_validation:
        data_train, data_val, targets_train, targets_val = split

    if not args.max_length:
        args.max_length = (data[0] if args.fine_tune_model else data).shape[1]

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
            if type(data) == list:
                data_train = [d[train_ind] for d in data]
                data_val = [d[val_ind] for d in data]
            else:
                data_train = data[train_ind]
                data_val = data[val_ind]

            if type(targets) == list:
                targets_train = [t[train_ind] for t in targets]
                targets_val = [t[val_ind] for t in targets]
            else:
                targets_train = targets[train_ind]
                targets_val = targets[val_ind]

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


def predict(args):
    """
    # NOTE: as described in the README using this function to do predictions gives different results from
    # doing predictions immediately after training; therefore it cannot be used to recreate our results
    # load vocabulary
    if not args.use_bpe:
        vocabulary = construct_corpus_vocabulary(args.vocabulary_source_file,
                                                 data_set_type=args.vocabulary_type,
                                                 vocabulary_size=args.vocabulary_size,
                                                 vocabulary_save_file=args.vocabulary_save_file)
    else:
        vocabulary = construct_corpus_bpe_vocabulary(args.vocabulary_source_file,
                                                     vocabulary_size=args.vocabulary_size,
                                                     vocabulary_save_file=args.vocabulary_save_file)

    model = load_model(args.model_load_file, custom_objects={"perplexity": perplexity,
                                                             "lm_accuracy": lm_accuracy,
                                                             "SeqSelfAttention": SeqSelfAttention,
                                                             "ScaledDotProductAttention": ScaledDotProductAttention})

    if args.data_path:
        data_test, _ = load_corpus_data(args.data_path, vocabulary, max_length=args.max_length,
                                        use_bpe=args.use_bpe, for_classification=True, test_data=True)
        file_name = os.path.abspath(args.data_path)
        file_name = re.sub(r"\.csv", "", file_name)
        file_name += "-predictions.csv"
        predict_after_training(model, data_test, file_name)
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("handcrafted_path", type=str, help="Path to the handcrafted features.")
    parser.add_argument("--validation_data", "-vd", type=str, default=None,
                        help="Data to use for validation (instead of splitting the training data).")
    parser.add_argument("--prediction_data", "-pd", type=str, default=None,
                        help="Data to use for prediction once training is finished.")
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
    parser.add_argument("--cell_type", "-ct", type=str, default="LSTM", help="The type of RNN cell to use.")
    parser.add_argument("--cell_size", "-cs", type=int, default=128, help="The size of the used RNN cells.")
    parser.add_argument("--cell_stack_size", "-css", type=int, default=2, help="The number of RNN cells to stack.")
    parser.add_argument("--dense_size", "-ds", type=int, nargs="+", default=[],
                        help="The sizes of the dense layers before the classifier.")
    parser.add_argument("--embedding_dropout", "-edo", type=float, default=0.6,
                        help="Dropout rate for the embedding layer.")
    parser.add_argument("--dense_dropout", "-ddo", type=float, nargs="+", default=[0.3],
                        help="Dropout rates for the dense layers.")
    parser.add_argument("--classifier_dropout", "-cdo", type=float, default=0.1,
                        help="Dropout rate for the classifier.")
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

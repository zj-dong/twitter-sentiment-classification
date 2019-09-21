import os
import numpy as np
import argparse

from data.load_data import prepare_data
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Embedding, Bidirectional
from keras.layers import LSTM, GRU
from keras.models import Sequential

np.random.seed(50)

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("freq_dist_path", type=str)
parser.add_argument("embedding_path", type=str)
parser.add_argument("--vocabulary_size", "-vs", type=int, default=20000)
parser.add_argument("--embedding_size", "-es", type=int, default=200)
parser.add_argument("--embedding_type", "-et", type=str, default="GLOVE")
parser.add_argument("--train_val_split", "-tvs", type=float, default=0.9)
parser.add_argument("--batch_size", "-bs", type=int, default=1024)
parser.add_argument("--epochs", "-ep", type=int, default=10)
parser.add_argument("--cell_type", "-ct", type=str, default="LSTM")
parser.add_argument("--num_cells", "-nc", type=int, default=3)
parser.add_argument("--checkpoint_path", "-cpp", type=str, default="../checkpoints/bi_lstm-{epoch:02d}-{loss:0.3f}-"
                                                                   "{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5")
parser.add_argument("--log_dir", "-ld", type=str, default="../../logs")
parser.add_argument("--model_path", type=str, default='../checkpoints/bi_lstm-07-0.275-0.882-0.308-0.869.hdf5')


def build_model(vocab_size, embedding_size, embedding_matrix, max_length, cell_type, cell_stack_size):
    assert cell_type.lower() in ["lstm", "gru"]
    assert cell_stack_size >= 1

    if cell_type.lower() == "lstm":
        cell_type = LSTM
    elif cell_type.lower() == "gru":
        cell_type = GRU

    model = Sequential()
    model.add(Embedding(vocab_size + 1, embedding_size, weights=[embedding_matrix], input_length=max_length))
    for i in range(cell_stack_size - 1):
        model.add(Bidirectional(cell_type(256, return_sequences=True, recurrent_initializer='orthogonal', dropout=0.5)))
    model.add(Bidirectional(cell_type(256, recurrent_initializer='orthogonal', dropout=0.5)))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    args = parser.parse_args()

    # load data and convert it
    vocabulary, data, labels, embedding_matrix = prepare_data(args.data_path, args.freq_dist_path,
                                                              args.embedding_path, args.vocabulary_size,
                                                              args.embedding_size, args.embedding_type)

    # split into training and validation set
    indices = np.array(range(len(data)))
    np.random.shuffle(indices)
    split_index = int(len(data) * args.train_val_split)
    train_idx = list(np.load("training_data_full_indices.npy"))
    val_idx = list(set([i for i in range(0, 2500000)]) - set(train_idx))
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    validation_data = data[val_idx]
    validation_labels = labels[val_idx]

    # create the model
    model = build_model(args.vocabulary_size, args.embedding_size, embedding_matrix,
                        len(data[0]), args.cell_type, args.num_cells)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.000001)
    print(model.summary())

    # stuff for saving models and displaying progress
    directory = os.path.dirname(os.path.abspath(args.checkpoint_path))
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint = ModelCheckpoint(args.checkpoint_path, monitor="loss", verbose=1, save_best_only=True, mode="min")

    directory = os.path.dirname(os.path.abspath(args.log_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    tensorboard = TensorBoard(args.log_dir, write_graph=False, update_freq=10000)

    # train model
    model.fit(train_data, train_labels, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(validation_data, validation_labels), shuffle=True,
              verbose=1, callbacks=[checkpoint, tensorboard, reduce_lr])


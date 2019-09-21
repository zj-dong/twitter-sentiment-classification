import os
import numpy as np
import argparse

from data.load_data import prepare_data
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Embedding, Layer, Lambda, TimeDistributed, Multiply, Bidirectional, Input
from keras.layers import LSTM, GRU
from keras.models import Sequential, Model
from keras.backend import int_shape
from keras_self_attention import SeqSelfAttention

np.random.seed(50)
GLOVE_FILE = "/media/simon/Storage-Disk/Simon/Linux/CIL-Project/glove.twitter.27B.200d.txt"

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("freq_dist_path", type=str)
parser.add_argument("embedding_path", type=str)
parser.add_argument("--use_own_implementation", "-uo", dest="use_own_implementation", action="store_true", default=False)
parser.add_argument("--vocabulary_size", "-vs", type=int, default=20000)
parser.add_argument("--embedding_size", "-es", type=int, default=200)
parser.add_argument("--embedding_type", "-et", type=str, default="GLOVE")
parser.add_argument("--train_val_split", "-tvs", type=float, default=0.9)
parser.add_argument("--batch_size", "-bs", type=int, default=64)
parser.add_argument("--epochs", "-ep", type=int, default=10)
parser.add_argument("--cell_type", "-ct", type=str, default="LSTM")
parser.add_argument("--cell_num", "-cn", type=int, default=1)
parser.add_argument("--cell_size", "-cs", type=int, default=128)
parser.add_argument("--dense_size", "-ds", type=int, default=64)
parser.add_argument("--attention_size", "-as", type=int, default=64)
parser.add_argument("--checkpoint_path", "-cpp", type=str, default="../../checkpoints/attention_lstm-{epoch:02d}-"
                                                                   "{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-"
                                                                   "{val_acc:0.3f}.hdf5")
parser.add_argument("--log_dir", "-ld", type=str, default="../../logs")


class Attention(Layer):

    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        print("input_shape (build): {}".format(input_shape))
        # matrix W multiplied by hidden state vector
        # self.W = TimeDistributed(Dense(self.units), input_shape=(input_shape[1], input_shape[2]))
        self.W = Dense(self.units, name="W", activation="tanh", input_shape=(input_shape[2],), trainable=True)
        self.W.build((input_shape[0], input_shape[2]))
        self.W_td = TimeDistributed(self.W, name="W_td", input_shape=(input_shape[1], input_shape[2]))
        print("'W' trainable weights (build): {}".format(self.W.trainable_weights))
        print("'W_td output shape: {}".format(self.W_td.compute_output_shape(input_shape)))
        self._trainable_weights.extend(self.W.trainable_weights)

        # vector V multiplied by tanh of the output of the above
        self.V = Dense(1, use_bias=False, activation="softmax", name="V", trainable=True)
        self.V.build((input_shape[0], input_shape[1], self.units))
        self.V_td = TimeDistributed(self.V, name="V_td", input_shape=(input_shape[1], self.units))
        print("'V' trainable weights (build): {}".format(self.V.trainable_weights))
        self._trainable_weights.extend(self.V.trainable_weights)
        print("trainable weights (build): {}".format(self._trainable_weights))

        super(Attention, self).build(input_shape)

    def call(self, hidden, **kwargs):
        batch_size, time_steps, hidden_size = int_shape(hidden)
        # hidden shape: (batch_size, time_steps, hidden_size)
        print("'hidden' shape: {}".format(int_shape(hidden)))
        print("'W' trainable weights: {}".format(self.W.trainable_weights))
        print("'W' matrix shape: {}".format(int_shape(self.W.trainable_weights[0])))

        # TODO: this is pretty much useless and not really self-attention => should be corrected
        scores = self.W_td(hidden)
        # score shape: (batch_size, time_steps, self.units)
        print("'scores' shape: {}".format(int_shape(scores)))

        attention_weights = self.V_td(scores)
        # attention_weights shape: (batch_size, time_steps, 1) => attention distribution over time
        # TODO: should potentially mask this (different length sequences)
        print("'attention_weights' shape: {}".format(int_shape(attention_weights)))

        attention_weights = K.repeat_elements(attention_weights, hidden_size, 2)
        # attention_weights shape: (batch_size, time_steps, hidden_size) => repeat so that Multiply() can be used
        print("'attention_weights' shape: {}".format(int_shape(attention_weights)))

        context_vector = Multiply(name="context_vector_mul")([attention_weights, hidden])
        # context_vector shape: (batch_size, time_steps, hidden_size) => weighted hidden states
        print("'context_vector' shape: {}".format(int_shape(context_vector)))

        context_vector = Lambda(lambda x: K.sum(x, axis=1), name="context_vector_max")(context_vector)
        # context_vector shape: (batch_size, hidden_size) => context vector which can be used for w/e
        # => basically a feature vector which can be passed through a dense classification layer
        print("'context_vector' shape: {}".format(int_shape(context_vector)))

        if kwargs.get("return_attention", None):
            return context_vector, attention_weights
        return context_vector

    def compute_output_shape(self, input_shape):
        # should be (batch_size, hidden_size)
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"units": self.units})
        return config


def build_model(vocab_size, embedding_size, embedding_matrix, max_length, cell_type,
                cell_stack_size, cell_size, dense_size, attention_size, use_own_implementation):
    assert cell_type.lower() in ["lstm", "gru"]
    assert cell_stack_size >= 1

    if cell_type.lower() == "lstm":
        cell_type = LSTM
    elif cell_type.lower() == "gru":
        cell_type = GRU

    # TODO: should probably use a Model class instead (at some point at least)
    model = Sequential()
    model.add(Embedding(inputs_dim=vocab_size + 1, output_dim=embedding_size, weights=[embedding_matrix], input_length=max_length))
    model.add(Dropout(rate=0.4))
    for i in range(cell_stack_size):
        model.add(Bidirectional(layer=cell_type(units=cell_size, return_sequences=True)))
        # model.add(Dropout(rate=0.5))
    if use_own_implementation:
        model.add(Attention(units=attention_size))
    else:
        model.add(SeqSelfAttention(units=attention_size, attention_activation="sigmoid", return_attention=True))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=dense_size, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_model_test(vocab_size, embedding_size, embedding_matrix, max_length, cell_type,
                     cell_stack_size, cell_size, dense_size, attention_size, use_own_implementation):
    assert cell_type.lower() in ["lstm", "gru"]
    assert cell_stack_size >= 1

    if cell_type.lower() == "lstm":
        cell_type = LSTM
    elif cell_type.lower() == "gru":
        cell_type = GRU

    # TODO: should probably use a Model class instead (at some point at least)
    inputs = Input(shape=(None,))
    embedded = Dropout(rate=0.4)(Embedding(input_dim=vocab_size + 1, output_dim=embedding_size,
                                           weights=[embedding_matrix], input_length=max_length)(inputs))
    lstm = embedded
    for i in range(cell_stack_size):
        lstm = Bidirectional(layer=cell_type(units=cell_size, return_sequences=True))(lstm)
    if use_own_implementation:
        context_vector = Attention(units=attention_size)(lstm)
    else:
        vectors_and_attention = SeqSelfAttention(units=attention_size, attention_activation="sigmoid",
                                                 return_attention=True)(lstm)
        context_vector = vectors_and_attention[0]
        # since the we are using self-attention, there are max_length context vectors (one for each word
        # in the input); therefore this matrix has to be reduced to a fixed-size vector representation
        # => max (https://arxiv.org/pdf/1703.03130.pdf section 4.1)
        # => average vector
        # => summation
        # context_vector = Lambda(function=lambda x: K.max(x, axis=1))(context_vector)  # try out max for now
        context_vector = Lambda(function=lambda x: K.mean(x, axis=1))(context_vector)
    dense = Dense(units=dense_size, activation="relu")(Dropout(rate=0.5)(context_vector))
    dense = Dense(units=1, activation="sigmoid")(Dropout(rate=0.5)(dense))
    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    args = parser.parse_args()

    # load data and convert it
    vocabulary, data, labels, embedding_matrix = prepare_data(args.data_path, args.freq_dist_path,
                                                              args.embedding_path, args.vocabulary_size,
                                                              args.embedding_size, args.embedding_type)

    exit()

    # split into training and validation set
    indices = np.array(range(len(data)))
    np.random.shuffle(indices)
    split_index = int(len(data) * args.train_val_split)
    train_data = data[indices[:split_index]]
    train_labels = labels[indices[:split_index]]
    validation_data = data[indices[split_index:]]
    validation_labels = labels[indices[split_index:]]
    print("Lengths:", len(train_data), len(train_labels), len(validation_data), len(validation_labels))

    # create the model
    model = build_model_test(args.vocabulary_size, args.embedding_size, embedding_matrix, len(data[0]), args.cell_type,
                        args.cell_num, args.cell_size, args.dense_size, args.attention_size, args.use_own_implementation)
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


import os
import numpy as np
import argparse
import keras
from data.load_data import prepare_data
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Embedding, Bidirectional
from keras.layers import LSTM, GRU
from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.models import *
np.random.seed(50)
GLOVE_FILE = "/media/simon/Storage-Disk/Simon/Linux/CIL-Project/glove.twitter.27B.200d.txt"

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("freq_dist_path", type=str)
parser.add_argument("embedding_path", type=str)
parser.add_argument("--vocabulary_size", "-vs", type=int, default=20000)
parser.add_argument("--embedding_size", "-es", type=int, default=200)
parser.add_argument("--embedding_type", "-et", type=str, default="GLOVE")
parser.add_argument("--train_val_split", "-tvs", type=float, default=0.9)
parser.add_argument("--batch_size", "-bs", type=int, default=2048)
parser.add_argument("--epochs", "-ep", type=int, default=10)
parser.add_argument("--cell_type", "-ct", type=str, default="LSTM")
parser.add_argument("--num_cells", "-nc", type=int, default=2)
parser.add_argument("--checkpoint_path", "-cpp", type=str, default="../checkpoints/bi_lstm-{epoch:02d}-{loss:0.3f}-"
                                                                   "{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5")
parser.add_argument("--log_dir", "-ld", type=str, default="../../logs")
parser.add_argument("--model_path", type=str,default='../checkpoints/bi_lstm-01-0.260-0.889-0.305-0.872.hdf5')





# for tensorflow 1.12.0
from tensorflow.python.ops import array_ops
from tensorflow.python.training import adam
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer

class MaskedAdamOptimizer(adam.AdamOptimizer):
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        gather_m_t = array_ops.gather(m_t, indices)
        gather_v_t = array_ops.gather(v_t, indices)
        gather_v_sqrt = math_ops.sqrt(gather_v_t)
        var_update = scatter_add(var, indices, -lr * gather_m_t / (gather_v_sqrt + epsilon_t))
        return control_flow_ops.group(*[var_update, m_t, v_t])


def build_model(vocab_size, embedding_size, embedding_matrix, max_length, cell_type, cell_stack_size):
    assert cell_type.lower() in ["lstm", "gru"]
    assert cell_stack_size >= 1

    if cell_type.lower() == "lstm":
        cell_type = LSTM
    elif cell_type.lower() == "gru":
        cell_type = GRU

    # TODO: should probably use a Model class instead (at some point at least)
    model = Sequential()
    model.add(Embedding(vocab_size + 1, embedding_size, weights=[embedding_matrix], input_length=max_length))
    model.add(SpatialDropout1D(0.25))
    for i in range(cell_stack_size - 1):
        model.add(Bidirectional(cell_type(256, return_sequences=True,recurrent_initializer='orthogonal',dropout=0.3)))
        model.add(Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform'))
    model.add(Bidirectional(cell_type(256,return_sequences=True,recurrent_initializer='orthogonal',dropout=0.3)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
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
    
    train_data = data[indices[:split_index]]
    train_labels = labels[indices[:split_index]]
    validation_data = data[indices[split_index:]]
    validation_labels = labels[indices[split_index:]]
    #print("***************",len(train_data))
    #print("***************",len(validation_data))

    # create the model
    model = build_model(args.vocabulary_size, args.embedding_size, embedding_matrix,
                        len(data[0]), args.cell_type, args.num_cells)
    #model = load_model(args.model_path)
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


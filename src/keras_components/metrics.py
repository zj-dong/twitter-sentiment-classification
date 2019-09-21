from keras import backend as K


def perplexity(y_true, y_pred):
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


def lm_accuracy(y_true, y_pred):
    shape = K.shape(y_pred)
    new_shape = [shape[0], shape[1], 1]
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.reshape(y_pred, shape=new_shape)
    y_pred = K.cast(y_pred, dtype="float32")
    correct = K.flatten(K.equal(y_true, y_pred))
    correct = K.cast(correct, dtype="float32")
    num_elements = K.cast(K.prod(shape[:-1], axis=None), dtype="float32")
    return K.sum(correct, axis=None) / num_elements


def lm_accuracy_one_liner(y_true, y_pred):
    shape = K.shape(y_pred)
    new_shape = [shape[0], shape[1], 1]
    num_elements = K.cast(K.prod(shape[:-1], axis=None), dtype="float32")
    return K.sum(K.cast(K.flatten(K.equal(y_true, K.cast(K.reshape(K.argmax(y_pred, axis=-1), shape=new_shape),
                                                         dtype="float32"))), dtype="float32"), axis=None) / num_elements


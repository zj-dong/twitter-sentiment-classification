import numpy as np
import pickle

from keras.preprocessing.sequence import pad_sequences
from bpe import Encoder


def prepare_data(data_path, freq_dist_path, embedding_path, vocabulary_size=10000,
                 embedding_size=200, predict=False, max_length=None, use_bpe=False):
    max_length_provided = max_length is not None

    separator = ","
    if data_path.endswith("tsv"):
        separator = "\t"

    # construct vocabulary
    vocabulary = None
    if not use_bpe:
        with open(freq_dist_path, "rb") as freq_dist_file:
            freq_dist = pickle.load(freq_dist_file)
        vocabulary = {"<pad>": 0, "<unk>": 1, "<user>": 2, "<url>": 3}
        most_common = freq_dist.most_common(vocabulary_size - len(vocabulary))
        vocabulary.update({w[0]: i + 2 for i, w in enumerate(most_common)})
        print("Constructed vocabulary of size {}.".format(vocabulary_size))

    # load data and convert it to indices
    data = []
    labels = []
    if not max_length_provided:
        max_length = 0
    with open(data_path, "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(lines):
            if not predict:
                tweet_id, sentiment, tweet = line.split(separator)
            else:
                tweet_id, tweet = line.split(separator)
            data.append(tweet.strip())

            if not predict:
                labels.append(int(sentiment))
    print("Loaded data ({} tweets).".format(len(data)))

    if not use_bpe:
        new_data = []
        for tweet in data:
            words = tweet.split()
            indices = []
            for w_idx, w in enumerate(words):
                if max_length_provided and w_idx == max_length:
                    break

                index = vocabulary.get(w)
                if index is not None:
                    indices.append(index)
                else:
                    indices.append(vocabulary.get("<unk>"))

            if not max_length_provided and len(indices) > max_length:
                max_length = len(indices)

            new_data.append(indices)
        data = new_data

        pad_value = vocabulary.get("<pad>")
    else:
        print("Training BPE encoder...")
        encoder = Encoder(vocab_size=vocabulary_size,
                          required_tokens=["<user>", "<url>"],
                          UNK="<unk>", PAD="<pad>")
        encoder.fit(data)
        vocabulary = encoder.vocabs_to_dict()
        print("Constructed BPE vocabulary of size {}.".format(vocabulary_size))

        new_data = []
        for tweet in data:
            indices = list(next(encoder.transform([tweet])))
            if not max_length_provided and len(indices) > max_length:
                max_length = len(indices)
            new_data.append(indices)
        data = new_data

        pad_value = encoder.word_vocab[encoder.PAD]

    # load embedding vectors
    embedding_vectors = {}
    if not use_bpe:
        with open(embedding_path, "r") as glove_file:
            for i, line in enumerate(glove_file):
                tokens = line.split()
                word = tokens[0]
                if vocabulary.get(word):
                    vector = [float(e) for e in tokens[1:]]
                    embedding_vectors[word] = np.array(vector)
        print("Found {} GLOVE vectors for vocabulary of size {}.".format(len(embedding_vectors), len(vocabulary)))
        print("Loaded embedding vectors ({} dimensions).".format(embedding_size))

    # construct embedding matrix
    embedding_matrix = np.random.randn(vocabulary_size, embedding_size) * 0.01
    if not use_bpe:
        for word, i in list(vocabulary.items()):
            embedding_vector = embedding_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print("Constructed embedding matrix.")

    # pad data (might want to change max_length to be CLI argument)
    data = pad_sequences(data, maxlen=max_length, padding="post", value=pad_value)
    if not predict:
        labels = np.array(labels)
    print("Padded sequences to length {}.".format(max_length))

    if not predict:
        return vocabulary, data, labels, embedding_matrix
    return vocabulary, data, embedding_matrix

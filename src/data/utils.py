import numpy as np
import nltk
import pickle
import logging

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from bpe import Encoder

from typing import *

logger = logging.getLogger(__name__)

TOKENIZER = nltk.tokenize.casual.TweetTokenizer()


def load_data(data_path: str,
              predict: bool = False,
              split: bool = True) -> Union[List[Union[List[str], str]], Tuple[List[Union[List[str], str]], np.ndarray]]:

    separator = ","
    if data_path.endswith("tsv"):
        separator = "\t"

    data = []
    labels = []
    logger.info("Loading data...")
    with open(data_path, "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(tqdm(lines)):
            try:
                if not predict:
                    tweet_id, sentiment, tweet = line.split(separator)
                else:
                    tweet_id, tweet = line.split(separator)
            except ValueError:
                print(line)
                raise

            if split:
                data.append(TOKENIZER.tokenize(tweet))
            else:
                data.append(tweet.strip())

            if not predict:
                labels.append(int(sentiment))
    logger.info("Loaded {} tweets.".format(len(data)))

    if not predict:
        return data, np.array(labels)

    return data


def load_handcrafted_features(handcrafted_path: str) -> np.ndarray:
    logger.info("Loading handcrafted features.")
    return np.load(handcrafted_path)


def construct_vocabulary(data: Union[str, List[Union[List[str], str]]],
                         vocabulary_size: int = 10000,
                         use_bpe: bool = False,
                         bpe_percentage: float = 0.2,
                         vocabulary_save_file: str = None) -> dict:
    counts = None
    if type(data) == str and ".pkl" in data:
        with open(data, "rb") as f:
            counts = pickle.load(f)
        if type(counts) != nltk.FreqDist:
            logger.info("Loaded vocabulary from file.")
            return counts
        elif use_bpe:
            logger.error("Cannot construct BPE vocabulary from frequency distribution file.")
            raise ValueError("Cannot construct BPE vocabulary from frequency distribution file.")
        else:
            logger.info("Constructing vocabulary from frequency distribution file.")
    elif not use_bpe:
        logger.info("Constructing vocabulary from data.")

        if type(data) == str:
            separator = ","
            if data.endswith("tsv"):
                separator = "\t"

            # load data from file
            new_data = []
            with open(data, "r") as data_file:
                lines = data_file.readlines()
                for i, line in enumerate(lines):
                    _, _, tweet = line.split(separator)
                    new_data.append(TOKENIZER.tokenize(tweet))
            data = new_data
        elif type(data[0]) != list:
            data = [TOKENIZER.tokenize(t) for t in data]

        all_words = []
        for tweet in data:
            all_words.extend(tweet)

        counts = nltk.FreqDist(all_words)

    if use_bpe:
        logger.info("Training BPE encoder...")
        encoder = Encoder(vocab_size=vocabulary_size,
                          pct_bpe=bpe_percentage,
                          word_tokenizer=lambda x: TOKENIZER.tokenize(x),
                          required_tokens=["<start>", "<extract>", "<user>", "<url>"],
                          UNK="<unk>", PAD="<pad>")
        encoder.fit(data)
        vocabulary = encoder.vocabs_to_dict()
        logger.info("Constructed BPE vocabulary of size {}.".format(vocabulary_size))
    else:
        vocabulary = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<extract>": 3}
        initial_vocab_length = len(vocabulary)
        most_common = counts.most_common(vocabulary_size - initial_vocab_length)
        vocabulary.update({w[0]: i + initial_vocab_length for i, w in enumerate(most_common)})
        logger.info("Constructed embedding vocabulary of size {}.".format(len(vocabulary)))

    if vocabulary_save_file:
        if not vocabulary_save_file.endswith(".pkl"):
            vocabulary_save_file += ".pkl"
        with open(vocabulary_save_file, "wb") as f:
            pickle.dump(vocabulary, f)
        logger.info("Saved vocabulary to \"{}\".".format(vocabulary_save_file))

    return vocabulary


def get_index_split(data: Union[np.ndarray, List[np.ndarray]],
                    split: float = 0.8,
                    random_seed: int = 42):
    n = len(data[0]) if type(data) == list and type(data[0]) == np.ndarray else len(data)
    indices_train, indices_val = train_test_split(
        list(range(n)), train_size=split, random_state=random_seed, shuffle=True)
    return np.array(indices_train), np.array(indices_val)


def split_by_index(indices_1: np.ndarray, indices_2: np.ndarray, *data) -> Tuple[Union[list, Any], ...]:
    splits = []
    for dt in data:
        if type(dt) == list:
            if type(dt[0]) == np.ndarray:
                split_1 = [d[indices_1] for d in dt]
                split_2 = [d[indices_2] for d in dt]
            else:
                split_1 = []
                for i in indices_1.tolist():
                    split_1.append(dt[i])
                split_2 = []
                for i in indices_2.tolist():
                    split_2.append(dt[i])
        else:
            split_1 = dt[indices_1]
            split_2 = dt[indices_2]
        splits.append(split_1)
        splits.append(split_2)
    return tuple(splits)


def split_data(data: Union[np.ndarray, List[Any]],
               targets: Union[np.ndarray, List[Any]],
               cross_validation: bool = False,
               train_val_split: float = 0.8,
               cv_folds: int = 5,
               random_seed: int = 42) -> Union[Tuple[Union[list, np.ndarray], Union[list, Any],
                                                     Union[list, np.ndarray], Union[list, Any]], StratifiedKFold]:

    if not cross_validation:
        if train_val_split == 1.0:
            return data, None, targets, None

        if len(data) == len(targets) and not (type(data) == list or type(targets) == list):
            data_train, data_val, targets_train, targets_val = train_test_split(
                data, targets, train_size=train_val_split, random_state=random_seed, shuffle=True)
        else:
            indices_train, indices_val = get_index_split(data, train_val_split, random_seed)
            data_train, data_val, targets_train, targets_val = split_by_index(indices_train, indices_val, data, targets)

        return data_train, data_val, targets_train, targets_val
    else:
        cv_folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        return cv_folds


def load_embedding_vectors(vocabulary: dict, embedding_file: str):
    logger.info("Loading word embedding vectors...")
    embedding_vectors = {}
    with open(embedding_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            tokens = line.split()
            if vocabulary.get(tokens[0]) is not None:
                embedding_vectors[tokens[0]] = np.array([float(e) for e in tokens[1:]])
    logger.info("Loaded {} word embedding vectors.".format(len(embedding_vectors)))
    return embedding_vectors


def create_embedding_matrix(vocabulary: dict, embedding_vectors: dict):
    embedding_size = len(embedding_vectors[next(iter(embedding_vectors))])
    embedding_matrix = np.random.randn(len(vocabulary), embedding_size) * 0.02
    for word, index in list(vocabulary.items()):
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    logger.info("Created embedding matrix.")
    return embedding_matrix


def encode_data(data: List[Union[List[str], str]],
                vocabulary: dict,
                labels: Union[np.ndarray, List[int]] = None,
                max_length: int = None,
                use_bpe: bool = False,
                for_classification: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:

    if not use_bpe and type(data[0]) != list:
        logger.error("Tweets need to be tokenized for encoding.")
        raise ValueError("Tweets need to be tokenized for encoding.")

    max_length_given = max_length is not None
    if not max_length_given:
        max_length = 0

    encoder = None
    if use_bpe:
        encoder = Encoder.from_dict(vocabulary)
        encoder.word_tokenizer = lambda x: TOKENIZER.tokenize(x)
        encoder.custom_tokenizer = True

    encoded_data = []
    logger.info("Encoding data...")
    for tweet in tqdm(data):
        current_tweet = []
        if use_bpe:
            current_tweet.extend(list(next(encoder.transform([tweet]))))
        else:
            for token_idx, token in enumerate(tweet):
                if max_length_given and token_idx == max_length:
                    break
                if vocabulary.get(token):
                    current_tweet.append(vocabulary.get(token))
                else:
                    current_tweet.append(vocabulary.get("<unk>"))
        if for_classification:
            start = (encoder.word_vocab if use_bpe else vocabulary).get("<start>")
            extract = (encoder.word_vocab if use_bpe else vocabulary).get("<extract>")
            current_tweet.insert(0, start)
            current_tweet.append(extract)
        encoded_data.append(current_tweet)

        if not max_length_given:
            max_length = max(max_length, len(current_tweet))
    logger.info("Encoded data.")

    if not max_length_given and not for_classification:
        # add these two to account for <start> and <extract> tokens later
        max_length += 2

    pad_value = encoder.word_vocab[encoder.PAD] if use_bpe else 0
    encoded_data = pad_sequences(encoded_data, maxlen=max_length, padding="post", value=pad_value)
    if labels is not None:
        encoded_targets = np.concatenate([encoded_data[:, 1:], np.full((len(encoded_data), 1), pad_value)], axis=1)
        encoded_targets = np.reshape(encoded_targets, encoded_targets.shape + (1,))
        return encoded_data, [encoded_targets, np.array(labels)]

    return encoded_data

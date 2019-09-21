import argparse
import pickle
import os
import numpy as np
import re

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from tqdm import tqdm

CLASSIFIERS = {
    "svm": SVC,
    "lr": LogisticRegression,
    "p": Perceptron,
    "nb": GaussianNB
}


def encode_data(data, vocabulary, embedding_map):
    print("Encoding data...")
    feature_vectors = []
    for tweet in tqdm(data):
        current_vector = np.zeros_like(next(iter(embedding_map.values())))
        for token in tweet:
            if (token in vocabulary) and (token in embedding_map):
                current_vector += embedding_map[token]
        feature_vectors.append(current_vector)
    feature_vectors = np.array(feature_vectors)
    print("Encoding data into feature vectors of size {}.".format(feature_vectors.shape[1]))
    return feature_vectors


def load_embedding(args, vocabulary):
    if args.pre_trained:
        print("Using pre-trained embeddings from:", args.embedding_path)
        embedding_map = {}
        with open(args.embedding_path,'r') as f:
            for line in f:
                key, val = line.strip().split(" ", 1)
                embedding_map[key] = np.fromstring(val, dtype=float, sep=' ')
    else:
        # load the embedding matrix
        embedding_vectors = np.load(args.embedding_path)

        embedding_vectors = [embedding_vectors[k] for k in embedding_vectors]
        # use word embedding, discard the context embedding
        embedding_vectors = embedding_vectors[0]

        # get the vocab to index map
        with open(args.index_map_path, "rb") as f:
            index_map = pickle.load(f)

        # build a dictionary mapping from words in the vocab to embedding vectors
        embedding_map = {k: embedding_vectors[index_map[k]] for k in vocabulary}
    return embedding_map


def load_data(args):
    # load data and labels and split into tokens
    separator = ","
    if args.data_path.endswith("tsv"):
        separator = "\t"

    data = []
    labels = []
    print("Loading data...")
    with open(args.data_path, "r") as data_file:
        lines = data_file.readlines()
        for i, line in enumerate(tqdm(lines)):
            try:
                if not args.predict:
                    tweet_id, sentiment, tweet = line.split(separator)
                else:
                    tweet_id, tweet = line.split(separator)
            except ValueError:
                print(line)
                raise

            data.append([t.strip() for t in tweet.split()])
            if not args.predict:
                labels.append(int(sentiment))
    print("Loaded {} tweets.".format(len(data)))

    # load the vocabulary from a txt file with counts
    with open(args.vocabulary_file, "r") as f:
        vocab_counts = ((s[1], int(s[0])) for s in [x.strip().split() for x in f.readlines()])
    vocab_counts = sorted(vocab_counts, key=lambda x: -x[1])
    vocabulary = [t[0] for t in list(filter(lambda x: x[1] >= 5, vocab_counts))]

    # load the embedding
    embedding_map = load_embedding(args, vocabulary)

    # encode the data using the embeddings
    data = encode_data(data, vocabulary, embedding_map)

    if not args.predict:
        return data, np.array(labels)

    return data


def parse_params(args):
    params_dict = {}
    for param in args.params:
        name, value = param.split(":")
        values = value.split(",")
        new_values = []
        for v in values:
            try:
                new_values.append(int(v))
            except ValueError:
                try:
                    new_values.append(float(v))
                except ValueError:
                    new_values.append(v)
        values = new_values
        if len(values) == 1 and not args.grid_search:
            values = values[0]
        params_dict[name] = values
    return params_dict


def grid_search(args, classifier, params, x, y):
    classifier = classifier()
    gs_classifier = GridSearchCV(classifier, params, scoring="accuracy", n_jobs=args.n_jobs,
                                 cv=args.cv_splits, verbose=args.verbosity)
    gs_classifier.fit(x, y)

    print("Best parameters found for classifier \"{}\":".format(args.classifier))
    pprint(gs_classifier.best_params_)
    print("\nGrid scores:")
    means = gs_classifier.cv_results_["mean_test_score"]
    stds = gs_classifier.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, gs_classifier.cv_results_["params"]):
        print("{:0.3f} (+/-{:0.03f}) for {}".format(mean, std * 2, params))

    return gs_classifier


def train(args):
    # params and classifier
    params = parse_params(args)
    classifier = CLASSIFIERS[args.classifier]

    # load data
    x, y = load_data(args)

    # do grid search or train
    if args.grid_search:
        classifier = grid_search(args, classifier, params, x, y)
        classifier = classifier.best_estimator_
    else:
        if (args.classifier == "lr") or (args.classifier == "p"):
            classifier = classifier(**params, n_jobs=args.n_jobs, verbose=args.verbosity)
        elif args.classifier == "svm":
            classifier = classifier(**params, verbose=args.verbosity)
        elif args.classifier == "nb":
            classifier = classifier(**params)

        classifier.fit(x, y)

    if args.save_file:
        if os.path.isdir(args.save_file):
            save_file = os.path.join(args.save_file, "{}-classifier{}.csv".format(
                args.classifier, "-gs" if args.grid_search else ""))
        else:
            save_file = args.save_file + ("" if ".pkl" in args.save_file else ".pkl")
        with open(save_file, "wb") as f:
            pickle.dump(classifier, f)
        print("Saved classifier to \"{}\".".format(save_file))


def predict(args):
    # load unlabeled data
    x = load_data(args)

    # load classifier
    with open(args.load_file, "rb") as f:
        classifier = pickle.load(f)

    # do the predictions
    predictions = classifier.predict(x)

    # save the predictions
    if args.prediction_save_file:
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
            f.write(str(0 if p == 0 else 1))
            f.write("\n")
    print("Saved predictions to \"{}\".".format(file_name))


def main(args):
    if args.predict and not args.load_file:
        raise ValueError("Can only do predictions when a model to load is specified.")

    if not args.grid_search and not args.save_file:
        print("WARNING: no classifier save file specified even though "
              "the classifier will only be trained and not tested!")

    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("data_path", type=str)
    parser.add_argument("embedding_path", type=str)
    parser.add_argument("vocabulary_file", type=str)
    parser.add_argument("index_map_path", type=str)

    # classifier selection and parameters
    parser.add_argument("--classifier", "-c", type=str, default="svm", choices=["svm", "lr", "p", "nb"])
    parser.add_argument("--params", "-pr", type=str, default="", nargs="+",
                        help="Should be specified as \"name:value\". Multiple parameters can be specified "
                             "separated by space and multiple values can be separated by commas.")

    parser.add_argument("--grid_search", "-gs", action="store_true")
    parser.add_argument("--cv_splits", "-cv", type=int, default=5)

    parser.add_argument("--n_jobs", "-j", type=int, default=1)
    parser.add_argument("--verbosity", "-v", type=int, default=1)

    parser.add_argument("--save_file", "-sf", type=str, default=None)
    parser.add_argument("--load_file", "-lf", type=str, default=None)

    parser.add_argument("--predict", "-p", action="store_true")
    parser.add_argument("--pre_trained", "-ptr", action="store_true")

    parser.add_argument("--prediction_save_file", "-psf", type=str, default=None)

    arguments = parser.parse_args()
    main(arguments)

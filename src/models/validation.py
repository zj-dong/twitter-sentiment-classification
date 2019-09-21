import argparse
import os
import numpy as np

from data import *


def generate_and_save_validation_data(args):
    # load all of the data
    data, targets = load_data(args.data_path, split=False)

    # split data
    indices_train = indices_val = None
    if args.save_indices:
        indices_train, indices_val = get_index_split(data, 0.8, random_seed=42)
        data_train, data_val, targets_train, targets_val = split_by_index(indices_train, indices_val, data, targets)

        # save indices for full training data
        save_path_indices = os.path.join(args.save_path, "training_data_full_indices.npy")
        np.save(save_path_indices, indices_train)
        print("Saved full training data indices of size {}.".format(len(indices_train)))

        # save indices for full validation set
        save_path_indices = os.path.join(args.save_path, "validation_data_full_indices.npy")
        np.save(save_path_indices, indices_val)
        print("Saved full validation data indices of size {}.".format(len(indices_val)))
    else:
        data_train, data_val, targets_train, targets_val = split_data(data, targets, train_val_split=0.8, random_seed=42)

    if args.down_sample_val < 1.0:
        # save the validation data with labels
        save_path_labeled = os.path.join(args.save_path, "validation_data_labeled_full.tsv")
        with open(save_path_labeled, "w") as f:
            for i in range(len(targets_val)):
                f.write("{}\t{}\t{}\n".format(i, targets_val[i], data_val[i]))

        # save the validation data in test data format
        save_path_unlabeled = os.path.join(args.save_path, "validation_data_unlabeled_full.tsv")
        with open(save_path_unlabeled, "w") as f:
            for i in range(len(targets_val)):
                f.write("{}\t{}\n".format(i + 1, data_val[i]))
        print("Saved full validation data of size {}.".format(len(targets_val)))

        indices_in, indices_ex = get_index_split(data_val, args.down_sample_val, random_seed=42)
        data_val, _, targets_val, _ = split_by_index(indices_in, indices_ex, data_val, targets_val)

        if args.save_indices:
            indices_val_small = indices_val[indices_in]
            save_path_indices = os.path.join(args.save_path, "validation_data_small_indices{}.npy".format(
                "" if args.down_sample_val == 1.0 else "_ds0" + str(args.down_sample_val)[2:]))
            np.save(save_path_indices, indices_val_small)
            print("Saved small validation data indices of size {}.".format(len(indices_val_small)))

    # save the training data
    save_path_labeled = os.path.join(args.save_path, "training_data_full.tsv")
    with open(save_path_labeled, "w") as f:
        for i in range(len(targets_train)):
            f.write("{}\t{}\t{}\n".format(i, targets_train[i], data_train[i]))
    print("Saved full training data of size {}.".format(len(targets_train)))

    if args.down_sample_train < 1.0:
        indices_in, indices_ex = get_index_split(data_train, args.down_sample_train, random_seed=42)
        data_train, _, targets_train, _ = split_by_index(indices_in, indices_ex, data_train, targets_train)

        # save the small training data set
        save_path_labeled = os.path.join(args.save_path, "training_data_small{}.tsv".format(
            "" if args.down_sample_train == 1.0 else "_ds0" + str(args.down_sample_train)[2:]))
        with open(save_path_labeled, "w") as f:
            for i in range(len(targets_train)):
                f.write("{}\t{}\t{}\n".format(i, targets_train[i], data_train[i]))

        if args.save_indices:
            indices_train_small = indices_train[indices_in]
            save_path_indices = os.path.join(args.save_path, "training_data_small_indices{}.npy".format(
                "" if args.down_sample_train == 1.0 else "_ds0" + str(args.down_sample_train)[2:]))
            np.save(save_path_indices, indices_train_small)
            print("Saved small training data indices of size {}.".format(len(indices_train_small)))

    # save the validation data with labels
    save_path_labeled = os.path.join(args.save_path, "validation_data_labeled{}.tsv".format(
        "_full" if args.down_sample_val == 1.0 else "_ds0" + str(args.down_sample_val)[2:]))
    with open(save_path_labeled, "w") as f:
        for i in range(len(targets_val)):
            f.write("{}\t{}\t{}\n".format(i, targets_val[i], data_val[i]))

    # save the validation data in test data format
    save_path_unlabeled = os.path.join(args.save_path, "validation_data_unlabeled{}.tsv".format(
        "_full" if args.down_sample_val == 1.0 else "_ds0" + str(args.down_sample_val)[2:]))
    with open(save_path_unlabeled, "w") as f:
        for i in range(len(targets_val)):
            f.write("{}\t{}\n".format(i + 1, data_val[i]))
    print("Saved validation data of size {}.".format(len(targets_val)))

    print("Saved data to \"{}\".".format(args.save_path))


def evaluate(args):
    # load data with true labels
    _, targets = load_data(args.data_path, split=True)

    # load predictions
    with open(args.prediction_path, "r") as f:
        lines = [int(l.split(",")[1]) for l in f.readlines()[1:]]
        lines = [0 if l == -1 else l for l in lines]
        predictions = np.array(lines)

    # evaluate
    accuracy = (predictions == targets).mean() * 100
    print("Achieved accuracy of {:2f}% on validation set of size {}".format(accuracy, len(targets)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, default=None)

    # for generating the validation data files
    parser.add_argument("--down_sample_val", "-dsv", type=float, default=1.0)
    parser.add_argument("--down_sample_train", "-dst", type=float, default=1.0)
    parser.add_argument("--save_path", "-s", type=str, default=None)
    parser.add_argument("--save_indices", "-si", action="store_true")

    # for evaluation the predictions
    parser.add_argument("--evaluate", "-e", action="store_true")
    parser.add_argument("--prediction_path", "-p", type=str, default=None)

    arguments = parser.parse_args()

    if arguments.evaluate:
        evaluate(arguments)
    else:
        generate_and_save_validation_data(arguments)

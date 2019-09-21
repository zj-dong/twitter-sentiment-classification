#!/usr/bin/env python3
import numpy as np
import pickle
import argparse
import os

from tqdm import tqdm

#########################################################################################################
# Adapted from: https://github.com/dalab/lecture_cil_public/blob/master/exercises/ex6/glove_solution.py #
#########################################################################################################


def main(args):
    print("Loading co-occurrence matrix...")
    co_occurrence_file = args.cooc_file or "cooc.pkl"
    if args.data_dir and not args.cooc_file:
        co_occurrence_file = os.path.join(args.data_dir, co_occurrence_file)
    with open(co_occurrence_file, "rb") as f:
        co_occurrence_matrix = pickle.load(f)
    print("{} nonzero entries".format(co_occurrence_matrix.nnz))

    n_max = 100
    print("Using n_max =", n_max, "and co_occurrence_matrix.max() =", co_occurrence_matrix.max())

    print("Initializing embeddings...")
    print("co_occurrence_matrix shape:", co_occurrence_matrix.shape)
    embedding_dim = args.embedding_dim
    xs = np.random.normal(size=(co_occurrence_matrix.shape[0], embedding_dim))
    ys = np.random.normal(size=(co_occurrence_matrix.shape[1], embedding_dim))

    eta = args.eta
    alpha = args.alpha

    epochs = args.epochs

    for epoch in tqdm(range(epochs)):
        print("epoch {}".format(epoch))
        for ix, jy, n in tqdm(zip(co_occurrence_matrix.row, co_occurrence_matrix.col, co_occurrence_matrix.data)):
            log_n = np.log(n)
            fn = min(1.0, (n / n_max) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (log_n - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x

    save_file_name = args.out_file or "custom_glove_{}_dim".format(embedding_dim)
    if args.data_dir:
        save_file_name = os.path.join(args.data_dir, save_file_name)
    np.savez(save_file_name, xs, ys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_max", "-n", type=int, default=100)
    parser.add_argument("--embedding_dim", "-ed", type=int, default=20)
    parser.add_argument("--eta", "-e", type=float, default=0.001)
    parser.add_argument("--alpha", "-a", type=float, default=(3 / 4))
    parser.add_argument("--epochs", "-ep", type=int, default=20)
    parser.add_argument("--cooc_file", "-c", type=str, default=None)
    parser.add_argument("--out_file", "-o", type=str, default=None)
    parser.add_argument("--data_dir", "-d", type=str, default=None)
    arguments = parser.parse_args()

    main(arguments)

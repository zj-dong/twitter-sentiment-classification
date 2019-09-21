import argparse
import os
import numpy as np


def main(args):
    # load all predictions
    if not os.path.isdir(args.prediction_dir):
        raise ValueError("Need to specify a directory with prediction files.")

    predictions = []
    for fn in os.listdir(args.prediction_dir):
        if ".csv" in fn:
            with open(os.path.join(args.prediction_dir, fn), "r") as f:
                if args.soft_voting:
                    lines = [float(l.split(",")[1]) for l in f.readlines()[1:]]
                else:
                    lines = [int(l.split(",")[1]) for l in f.readlines()[1:]]
                    lines = [0 if l == -1 else l for l in lines]
                predictions.append(lines)

    weights = args.weights
    if not weights or len(weights) != len(predictions):
        weights = np.array([1.0 / len(predictions)] * len(predictions))
    else:
        weights = np.array(weights)
        weights /= weights.sum()

    predictions = np.array(predictions)
    predictions = predictions * weights[:, None]
    # predictions = np.round(np.mean(predictions, axis=0))
    predictions = np.round(np.sum(predictions, axis=0))
    predictions = list(predictions)

    if args.output_file:
        if os.path.isdir(args.output_file):
            file_name = os.path.join(args.output_file, "{}_voting_predictions_{}_clf.csv".format(
                "soft" if args.soft_voting else "hard", len(weights)))
        else:
            file_name = args.output_file + ("" if ".csv" in args.output_file else ".csv")
    else:
        file_name = os.path.join(args.prediction_dir, "{}_voting_predictions_{}_clf.csv".format(
            "soft" if args.soft_voting else "hard", len(weights)))
    with open(file_name, "w") as f:
        f.write("Id,Prediction\n")
        for p_idx, p in enumerate(predictions):
            f.write(str(p_idx + 1))
            f.write(",")
            f.write(str(-1 if p == 0 else 1))
            f.write("\n")
    print("Saved predictions to \"{}\".".format(file_name))


if __name__ == "__main__":
    # vso, vs, bpe, pct, ml, weights
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=str)
    parser.add_argument("--weights", "-w", type=float, nargs="+", default=None)
    parser.add_argument("--soft_voting", "-sv", action="store_true")
    parser.add_argument("--output_file", "-o", type=str, default=None)

    arguments = parser.parse_args()
    main(arguments)

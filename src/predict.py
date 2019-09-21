import os
import numpy as np
import argparse

from data.load_data import prepare_data
from keras.models import load_model
from keras_self_attention import SeqSelfAttention

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("data_path", type=str)
parser.add_argument("freq_dist_path", type=str)
parser.add_argument("embedding_path", type=str)
parser.add_argument("--use_own_implementation", "-uo", dest="use_own_implementation", action="store_true", default=False)
parser.add_argument("--max_length", "-ml", type=int, default=46)
parser.add_argument("--vocabulary_size", "-vs", type=int, default=10000)
parser.add_argument("--embedding_size", "-es", type=int, default=200)
parser.add_argument("--embedding_type", "-et", type=str, default="GLOVE")
parser.add_argument("--batch_size", "-bs", type=int, default=256)

if __name__ == "__main__":
    args = parser.parse_args()

    # load data and convert it
    vocabulary, data, embedding_matrix = prepare_data(args.data_path, args.freq_dist_path, args.embedding_path,
                                                      args.vocabulary_size, args.embedding_size,
                                                      predict=True, max_length=args.max_length)

    # load model and predict
    if args.use_own_implementation:
        model = load_model(args.model_path)
    else:
        model = load_model(args.model_path, custom_objects=SeqSelfAttention.get_custom_objects())
    print(model.summary())

    # predict and save
    prediction_path = os.path.splitext(args.data_path)[0] + "_prediction.csv"
    predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
    predictions = list(zip(list(range(1, len(data) + 1)), np.round(predictions[:, 0]).astype(int)))
    with open(prediction_path, "w") as f:
        f.write("Id,Prediction\n")
        for tweet_id, prediction in predictions:
            f.write(str(tweet_id))
            f.write(",")
            f.write(str(-1 if prediction == 0 else 1))
            f.write("\n")
    print("Saved predictions to '{}'.".format(prediction_path))

import argparse
import os
import re
import nltk
import normalise

from tqdm import tqdm

TRAIN_POS_SMALL = "train_pos.txt"
TRAIN_NEG_SMALL = "train_neg.txt"
TRAIN_POS_FULL = "train_pos_full.txt"
TRAIN_NEG_FULL = "train_neg_full.txt"
TEST = "test_data.txt"
TRAIN_DATA_SMALL = "train_data_small.tsv"
TRAIN_DATA_FULL = "train_data_full.tsv"
TEST_DATA = "test_data.tsv"

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--train_pos_small", type=str, default=TRAIN_POS_SMALL)
parser.add_argument("--train_neg_small", type=str, default=TRAIN_NEG_SMALL)
parser.add_argument("--train_pos_full", type=str, default=TRAIN_POS_FULL)
parser.add_argument("--train_neg_full", type=str, default=TRAIN_NEG_FULL)
parser.add_argument("--test", type=str, default=TEST)
parser.add_argument("--train_data_small", type=str, default=TRAIN_DATA_SMALL)
parser.add_argument("--train_data_full", type=str, default=TRAIN_DATA_FULL)
parser.add_argument("--test_data", type=str, default=TEST_DATA)
parser.add_argument("--emojis", action="store_true")
parser.add_argument("--hashtags", action="store_true")
parser.add_argument("--some_punctuation", action="store_true")
parser.add_argument("--stopwords", action="store_true")
parser.add_argument("--spelling", action="store_true")
parser.add_argument("--normalise", action="store_true")


STOPWORDS = set(nltk.corpus.stopwords.words("english"))
TOKENIZER = nltk.tokenize.casual.TweetTokenizer()


def save_set(file_path, tweets, labels=None, ids=None):
    assert labels is None or len(tweets) == len(labels)

    with open(file_path, "w") as f:
        for i in range(len(tweets)):
            if labels:
                f.write("{}\t{}\t{}\n".format(i if not ids else ids[i], labels[i], tweets[i]))
            else:
                f.write("{}\t{}\n".format(i if not ids else ids[i], tweets[i]))


def process_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))", " EMO_POS ", tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r"(:\s?D|:-D|x-?D|X-?D)", " EMO_POS ", tweet)
    # Love -- <3, :*
    tweet = re.sub(r"(<3|:\*)", " EMO_POS ", tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r"(;-?\)|;-?D|\(-?;)", " EMO_POS ", tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r"(:\s?\(|:-\(|\)\s?:|\)-:)", " EMO_NEG ", tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', " EMO_NEG ", tweet)
    return tweet


def process_hashtags(tweet):
    return re.sub(r"#(\S+)", r" \1 ", tweet)


def process_some_punctuation(tweet):
    return re.sub(r"\s[\"',-]\s", "", tweet)


def process_stopwords(tweet):
    words = tweet.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


def process_spelling(tweet):
    words = tweet.split()
    new_words = []
    for w in words:
        word = re.sub(r"(.)\1+", r"\1\1", w)
        word = re.sub(r"(\S)\1(-|')(\S)\2", r"\1\2", word)
        new_words.append(word)
    return " ".join(new_words)


def process_normalise(tweet):
    normalised_words = normalise.normalise(tweet, lambda x: TOKENIZER.tokenize(x), verbose=False, variety="AmE")
    return " ".join(normalised_words)


def process_tweet(tweet, args):
    new_tweet = re.sub(r"\s+", " ", tweet)
    if args.emojis:
        new_tweet = process_emojis(new_tweet)
    if args.hashtags:
        new_tweet = process_hashtags(new_tweet)
    if args.some_punctuation:
        new_tweet = process_some_punctuation(new_tweet)
    if args.stopwords:
        new_tweet = process_stopwords(new_tweet)
    if args.spelling:
        new_tweet = process_spelling(new_tweet)
    if args.normalise:
        new_tweet = process_normalise(new_tweet)
    new_tweet = re.sub(r"\s+", " ", new_tweet)
    return new_tweet.strip()


def process_set(tweets, args):
    processed_tweets = []
    for t in tqdm(tweets):
        processed_tweets.append(process_tweet(t, args))
    return processed_tweets


def parse_test(tweets):
    ids = []
    new_tweets = []
    for t in tweets:
        index = t.find(",")
        ids.append(int(t[:index]))
        new_tweets.append(t[(index + 1):])
    return ids, new_tweets


def main(args):
    # read the data
    with open(os.path.join(args.data_path, args.train_pos_small), "r") as f:
        train_pos_small = f.readlines()
    with open(os.path.join(args.data_path, args.train_neg_small), "r") as f:
        train_neg_small = f.readlines()
    with open(os.path.join(args.data_path, args.train_pos_full), "r") as f:
        train_pos_full = f.readlines()
    with open(os.path.join(args.data_path, args.train_neg_full), "r") as f:
        train_neg_full = f.readlines()
    with open(os.path.join(args.data_path, args.test), "r") as f:
        test = f.readlines()
    test_ids, test = parse_test(test)

    # process the sets
    train_pos_small = process_set(train_pos_small, arguments)
    train_neg_small = process_set(train_neg_small, arguments)
    train_pos_full = process_set(train_pos_full, arguments)
    train_neg_full = process_set(train_neg_full, arguments)
    test = process_set(test, arguments)

    # combine positive and negative sets
    train_small = train_pos_small + train_neg_small
    train_small_labels = [1 for _ in train_pos_small] + [0 for _ in train_neg_small]

    train_full = train_pos_full + train_neg_full
    train_full_labels = [1 for _ in train_pos_full] + [0 for _ in train_neg_full]

    # save sets
    output_path = args.output_path or args.data_path
    save_set(os.path.join(output_path, args.train_data_small), train_small, train_small_labels)
    save_set(os.path.join(output_path, args.train_data_full), train_full, train_full_labels)
    save_set(os.path.join(output_path, args.test_data), test, ids=test_ids)


if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)

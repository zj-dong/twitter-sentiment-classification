import re
import argparse

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("--use_stemmer", "-us", dest="use_stemmer", action="store_true", default=False)


def is_valid_word(word):
    return re.search(r"^[a-zA-Z][a-z0-9A-Z\._]*$", word) is not None


def preprocess_sentence(sentence):
    # convert to lower case
    sentence = sentence.lower()

    # stem words
    words = sentence.split()
    processed_sentence = []
    for word in words:
        # remove punctuation (TODO: probably want to keep this for RNN)
        word = word.strip("'\"?!,.():;")
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_sentence.append(word)

    return " ".join(processed_sentence)


def preprocess_csv(csv_file_name, processed_file_name):
    save_to_file = open(processed_file_name, "w")
    with open(csv_file_name, "r") as csv:
        lines = csv.readlines()
        for i, line in enumerate(tqdm(lines[1:])):
            data = re.sub(r"(,)(?=(?:[^\"]|\"[^\"]*\")*$)", "<sep>", line)
            data = data.split("<sep>")
            story_id = data[0]
            sentences = data[2:]
            save_to_file.write(story_id + ",")

            for s_idx, sentence in enumerate(sentences):
                processed_sentence = preprocess_sentence(sentence)
                save_to_file.write(processed_sentence + ("," if s_idx < len(sentences) - 1 else "\n"))
            # TODO: might want to make this possible for the validation file too
    save_to_file.close()
    print('\nSaved processed stories to: {}'.format(processed_file_name))
    return processed_file_name


if __name__ == '__main__':
    args = parser.parse_args()
    input_file = args.input_file
    use_stemmer = args.use_stemmer
    processed_file_name = input_file[:-4] + "-processed.csv"
    if use_stemmer:
        porter_stemmer = EnglishStemmer()
        processed_file_name = input_file[:-4] + "-processed-stemmed.csv"
    preprocess_csv(input_file, processed_file_name)

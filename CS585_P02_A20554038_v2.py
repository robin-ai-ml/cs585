import sys
import os
import pandas as pd
import time
import re
from string import punctuation
from CS585_P02_A20499169_bayes_classifier import train, test
from CS585_P02_A20499169_docset import Docset
# from CS585_P02_A20499169_stemmer import PorterStemmer

from CS585_P02_A20554038_contractions import expand_contractions

# stemer = PorterStemmer()


def lowercase_text(text):
    return text.lower()


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def stem(word):
    # return stemer.stem(word)
    # Implement the Porter Stemmer algorithm here
    # This is a simplified example
    if word.endswith("ing"):
        return word[:-3]
    elif word.endswith("ed"):
        return word[:-2]
    else:
        return word


LEMMA_DICT = {
    "am": "be",
    "are": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "having": "have",
    "has": "have",
    "had": "have",
}


STOPWORDS = [
    "a",
    "an",
    "the",
    "is",
    "are",
    "and",
    "or",
    "but",
    "for",
    "in",
    "on",
    "at",
    "with",
    "to",
    "from",
    "of",
    "by",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "we",
    "you",
    "me",
    "him",
    "her",
    "us",
    "them",
    "i",
    "my",
    "mine",
    "your",
    "yours",
    "his",
    "hers",
    "their",
    "theirs",
    "our",
    "ours",
    "not",
    "no",
    "nor",
    "so",
    "too",
    "very",
    "just",
    "can",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
]

STOPWORD_DICT = {w: 1 for w in STOPWORDS}


def lemmatize(word):
    # Check if the word is in the lemma dictionary
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]
    else:
        return word


def split_by_punc(text):
    r = re.compile(r"[{}]+".format(re.escape(punctuation)))
    return r.split(text)


def train_test_split(
    df: pd.DataFrame, train_size=0.8, test_size=0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = int(len(df) * train_size)
    test_rows = int(len(df) * (1 - test_size))
    return df.iloc[:train_rows], df.iloc[test_rows:]


def process_bar(i, total):
    progress = int((i / total) * 100)
    sys.stdout.write("\r")
    sys.stdout.write(
        "[{:<50}] {:.2f} {}/{}%".format("=" * int(progress // 2), progress, i, total)
    )
    sys.stdout.flush()


def pre_process(text: str):
    text = text.lower()
    text = remove_html_tags(text)
    text = expand_contractions(text)
    bag = {}
    for phrase in split_by_punc(text):
        for w in pre_process_phrase(phrase):
            if w not in bag:
                bag[w] = 0
            bag[w] += 1
    return bag
    # return phrases


def pre_process_phrase(phrase: str):
    negations = False
    words = []

    for w in phrase.split():
        w = stem(w)
        w = lemmatize(w)

        if w.isnumeric():
            continue
        if len(w) <= 1:
            continue
        if w in STOPWORD_DICT:
            continue
        if negations:
            w = "not_" + w
        if w == "not":
            negations = not negations

        words.append(w)

    return words


def main():
    train_size = 0.8
    dataset_file = "Reviews.csv"

    args = sys.argv[1:]
    if len(args) == 1:
        try:
            train_size = float(args[0]) / 100
        except Exception:
            train_size = 0.8

    train_size = train_size if train_size >= 0.2 and train_size <= 0.8 else 0.8
    print(train_size)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = cur_dir + "/" + dataset_file
    df = pd.read_csv(csv_file_path, header=0)
    print(df.head())

    print("remove duplicates in train dataset")
    df.drop_duplicates(subset=["Text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_df, test_df = train_test_split(df)
    print(len(df), len(train_df), len(test_df))
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    # ------------------------clean up the dataset --------------------------------------
    # remove the duplicate rows(texts) in dataset

    start_time = time.time()
    # df["Text"] = df["Text"].apply(lambda x: lowercase_text(x))
    print(train_df.head())
    # for i, row in train_df.iloc[:100].iterrows():
    docs = []
    # train_df_len = len(train_df)
    row_len, _ = train_df.shape
    train_set = Docset(docs)
    for i, row in train_df[["Text", "Score"]].iterrows():
        bag = pre_process(str(row["Text"]))
        score = int(row["Score"])
        label = "good" if score > 3 else "bad"
        train_set.add(bag, label)
        # docs.append((bag, label))
        process_bar(i, row_len - 1)

    # print("------------------", len(docs))

    # train_set = Docset(train_df)

    end_time = time.time()
    print("---  clean up data in seconds --- %s" % (end_time - start_time))
    # print(df.head())
    # ---------------------------Bayes  Classifier network learning ---------------------------------
    C = ["good", "bad"]
    print("\nTraining classifier... ")
    logprior, loglikelihood, V = train(train_set, C, process_bar)

    i = 0
    total_reviews = len(test_df)
    print("\nTesting classifier... ")
    # Initialize lists to store predictions and actual labels
    predictions = []
    actual_labels = []

    for index, row in test_df[["Text", "Score"]].iterrows():
        text = str(row["Text"])
        bag = pre_process(text)
        score = int(row["Score"])
        actual_label = "good" if score > 3 else "bad"
        predicted_label, _ = test(bag, logprior, loglikelihood, C, V)

        predictions.append(predicted_label)
        actual_labels.append(actual_label)

        # Update and display the progress bar
        # i += 1
        process_bar(index, total_reviews - 1)
        # progress = (i / total_reviews) * 100
        # sys.stdout.write("\r")
        # sys.stdout.write("[{:<50}] {:.2f}%".format("=" * int(progress // 2), progress))
        # sys.stdout.flush()

    print("\nTest results / metrics:\n")
    # Initialize counters for each metric
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Count occurrences
    for predicted, actual in zip(predictions, actual_labels):
        if predicted == "good" and actual == "good":
            true_positives += 1
        elif predicted == "bad" and actual == "bad":
            true_negatives += 1
        elif predicted == "good" and actual == "bad":
            false_positives += 1
        elif predicted == "bad" and actual == "good":
            false_negatives += 1

    # Calculate metrics
    sensitivity = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    negative_predictive_value = (
        true_negatives / (true_negatives + false_negatives)
        if (true_negatives + false_negatives) > 0
        else 0
    )
    accuracy = (
        (true_positives + true_negatives) / len(predictions)
        if len(predictions) > 0
        else 0
    )
    f_score = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )

    # Display metrics
    print(f"Number of true positives: {true_positives}")
    print(f"Number of true negatives: {true_negatives}")
    print(f"Number of false positives: {false_positives}")
    print(f"Number of false negatives: {false_negatives}")
    print(f"Sensitivity (recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Negative predictive value: {negative_predictive_value:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F-score: {f_score:.2f}")


if __name__ == "__main__":
    main()

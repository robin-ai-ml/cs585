import sys
import os
from numpy.lib import math
import pandas as pd
import time

from CS585_P02_A20499169_bayes_classifier import train, predict
from CS585_P02_A20499169_docset import Docset
from CS585_P02_A20499169_pre_process import pre_process
from CS585_P02_A20554038_util import print_test_metrics, process_bar


def train_test_split(
    df: pd.DataFrame, train_size, test_size
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = int(len(df) * train_size)
    test_rows = int(len(df) * (1 - test_size))
    train, test = df.iloc[:train_rows], df.iloc[test_rows:]

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test


def extract_train_size(args: list[str]) -> float:
    train_size = 0.8
    if len(args) == 1:
        try:
            train_size = float(args[0]) / 100
        except Exception:
            train_size = 0.8

    train_size = train_size if train_size >= 0.2 and train_size <= 0.8 else 0.8
    return train_size


def load_csv(file: str) -> pd.DataFrame:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = cur_dir + "/" + file
    df = pd.read_csv(csv_file_path, header=0)
    return df


def parse_row(row) -> tuple[dict[str, int], str]:
    bag = pre_process(str(row["Text"]))
    score = int(row["Score"])
    label = "good" if score > 3 else "bad"
    return bag, label


def classify_sentence(logprior, loglikelihood, C, V):
    while True:
        sentence = input("Enter your sentence:\n\nSentence S:\n")

        if len(sentence) < 2:
            print("invalid sentence, the length of sentence must be greater than 2")
            continue
        bag = pre_process(sentence)
        # print(bag)
        predicted, log_probs = predict(bag, logprior, loglikelihood, C, V)

        # debug Naive Bayes Classifier
        # for c in logprior:
        #     print(c, math.exp(logprior[c]))
        # for w in bag:
        #     if w in V:
        #         print(
        #             w,
        #             "good",
        #             math.exp(loglikelihood[(w, "good")]),
        #             "bad",
        #             math.exp(loglikelihood[(w, "bad")]),
        #             bag[w],
        #         )
        #     else:
        #         print(w, "-", bag[w])

        print("\n")
        print(f"was classified as {predicted}")

        probs = {label: math.exp(log_prob) for label, log_prob in log_probs.items()}
        total = sum(probs.values())
        for label, prob in probs.items():
            print(f"P({label}|S)={(prob/total):.4f}")
        print("\n")
        another = input("Do you want to enter another sentence [Y/N]? ")
        if another.lower() != "y":
            break


def bal(df: pd.DataFrame, num: int) -> pd.DataFrame:
    goods = 0
    bads = 0
    rows = []
    for i, row in df.iterrows():
        score = int(row["Score"])
        # label = "good" if score > 3 else "bad"
        if score > 3:
            if goods < num:
                rows.append(row)
                goods += 1
        else:
            if bads < num:
                rows.append(row)
                bads += 1

    return pd.DataFrame(rows)


def main():
    start_time = time.time()

    train_size = extract_train_size(sys.argv[1:])

    print("Peng, Xiaonan, A20554038")
    print("Zhan, Junhui, A20499169")
    print("solution:")
    print("Training set size: ", train_size * 100, "%")
    print()

    print("Load and process input data set ...")
    df = load_csv("Reviews.csv")
    # df = df.iloc[:30000]
    # print(df.head())

    df.drop_duplicates(subset=["Text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_df, test_df = train_test_split(df, train_size=train_size, test_size=0.2)
    # train_df = train_df.iloc[:10000]
    # train_df = bal(train_df, 10000)
    # train_df.reset_index(drop=True, inplace=True)

    train_df_size = len(train_df)
    train_set = Docset()
    for i, row in train_df.iterrows():
        bag, label = parse_row(row)
        train_set.add(bag, label)
        process_bar(i, train_df_size - 1)

    print("\n")
    print("Training classifier ...")

    C = ["good", "bad"]
    logprior, loglikelihood, V = train(train_set, C)

    print("\n")
    print("Testing classifier ...")
    test_size = len(test_df)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # err_items = []
    for i, row in test_df.iterrows():
        bag, actual = parse_row(row)
        predicted, _ = predict(bag, logprior, loglikelihood, C, V)
        process_bar(i, test_size - 1)

        if predicted == C[0]:
            if predicted == actual:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if predicted == actual:
                true_negatives += 1
            else:
                false_negatives += 1

        # if predicted != actual:
        #     err_items.append((row["Text"], row["Score"], bag, predicted))

    # for _text, _score, _bag, _predicted in err_items[:10]:
    #     print(_text)
    #     print(_score)
    #     print(_bag)
    #     print(_predicted)
    #     print("-------")

    print("\n")
    print("Test results / metrics:")
    print_test_metrics(true_positives, false_positives, true_negatives, false_negatives)

    end_time = time.time()
    print("\n")
    print(f"Time consumed in seconds: {(end_time - start_time):.4f}s")
    print("\n")

    classify_sentence(logprior, loglikelihood, C, V)


if __name__ == "__main__":
    main()

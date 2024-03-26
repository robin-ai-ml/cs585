import sys
import re

# from CS585_P02_A20499169_stemmer import PorterStemmer

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


def process_bar(i, total):
    progress = int((i / total) * 100)
    sys.stdout.write("\r")
    sys.stdout.write(
        "[{:<50}] {:.2f} {}/{} ".format("=" * int(progress // 2), progress, i, total)
    )
    sys.stdout.flush()


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def stem(word):
    # Porter Stemmer
    # stemmer = PorterStemmer()
    # return stemmer.stem(word)

    # simplified stemmer
    if word.endswith("ing"):
        return word[:-3]
    elif word.endswith("ed"):
        return word[:-2]
    else:
        return word


def lemmatize(word):
    # Check if the word is in the lemma dictionary
    if word in LEMMA_DICT:
        return LEMMA_DICT[word]
    else:
        return word


def print_test_metrics(
    true_positives: int, false_positives: int, true_negatives: int, false_negatives: int
):
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
    accuracy = (true_positives + true_negatives) / (
        true_positives + false_positives + true_negatives + false_negatives
    )

    f_score = (
        2 * (precision * sensitivity) / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0
    )
    print(f"Number of true positives: {true_positives}")
    print(f"Number of true negatives: {true_negatives}")
    print(f"Number of false positives: {false_positives}")
    print(f"Number of false negatives: {false_negatives}")
    print(f"Sensitivity (recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Negative predictive value: {negative_predictive_value:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-score: {f_score:.4f}")

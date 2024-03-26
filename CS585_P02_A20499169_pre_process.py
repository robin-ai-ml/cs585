import re
from string import punctuation
from CS585_P02_A20554038_contractions import expand_contractions

from CS585_P02_A20554038_util import STOPWORD_DICT, lemmatize, remove_html_tags, stem


# Text pre-processing
# Convert text to bag
def pre_process(text: str) -> dict[str, int]:
    text = text.lower()
    text = remove_html_tags(text)
    text = expand_contractions(text)
    bag = {}
    for line in text.splitlines():
        for phrase in split_by_punc(line):
            for w in parse_phrase(phrase):
                if w not in bag:
                    bag[w] = 0
                bag[w] += 1
    return bag


def split_by_punc(text):
    r = re.compile(r"[{}]+".format(re.escape(punctuation)))
    return r.split(text)


def parse_phrase(phrase: str):
    negations = False
    words = []

    for w in phrase.split():
        w = stem(w)
        w = lemmatize(w)

        # skip numbers
        if w.isnumeric():
            continue

        # skip empty and 1 char words
        if len(w) <= 1:
            continue

        # Add not_ prefix to every word between
        # negation and following punctuation
        if negations:
            w = "not_" + w
        if w == "not":
            negations = not negations

        # skip stop words
        if w in STOPWORD_DICT:
            continue

        words.append(w)

    return words

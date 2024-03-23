import re

# Contractions mapping
CONTRACTIONS_MAP = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "i've": "i have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(text):
    mapping = CONTRACTIONS_MAP
    rep = dict((re.escape(k).lower(), v) for k, v in mapping.items())
    pattern = re.compile("|".join(rep.keys()), flags=re.IGNORECASE | re.DOTALL)
    text = pattern.sub(lambda m: rep[re.escape(m.group(0)).lower()], text)
    return text
    # contractions_pattern = re.compile(
    #     "({})".format("|".join(contraction_mapping.keys())),
    #     flags=re.IGNORECASE | re.DOTALL,
    # )

    # def expand_match(contraction):
    #     match = contraction.group(0)
    #     first_char = match[0]
    #     expanded_contraction = (
    #         contraction_mapping.get(match)
    #         if contraction_mapping.get(match)
    #         else contraction_mapping.get(match.lower())
    #     )
    #     expanded_contraction = first_char + expanded_contraction[1:]
    #     return expanded_contraction

    # expanded_text = contractions_pattern.sub(expand_match, text)
    # return expanded_text


if __name__ == "__main__":
    print(expand_contractions("What'll you'D woulDn't"))

import math

from CS585_P02_A20499169_docset import Docset
from CS585_P02_A20554038_util import process_bar


# Naive Bayes: Training/Testing
# lecture 10, CS585_Lecture_February12th.pdf


def train(D: Docset, C: list[str]):
    logprior = {}
    loglikelihood = {}
    V = D.vocabulary()
    V_len = len(V)
    total_words = V_len * len(C)
    i = 0
    for c in C:
        N_doc = D.total_doc_count()
        N_c = D.class_doc_count(c)
        logprior[c] = math.log(N_c / N_doc)
        class_count = D.class_token_count(c)
        # class_count = D.class_doc_count(c)

        for w in V:
            word_class_count = D.word_class_token_count(w, c)
            # word_class_count = D.word_class_doc_count(w, c)
            loglikelihood[(w, c)] = math.log(
                (word_class_count + 1) / (class_count + V_len)
            )
            i += 1
            process_bar(i, total_words)

    return logprior, loglikelihood, V


def predict(testdoc: dict[str, int], logprior, loglikelihood, C, V):
    sum = {}
    max_sum = -9999
    max_c = None
    for c in C:
        sum[c] = logprior[c]
        for word in testdoc:
            if word in V:
                sum[c] += loglikelihood[(word, c)]
        if sum[c] > max_sum:
            max_sum = sum[c]
            max_c = c

    return max_c, sum

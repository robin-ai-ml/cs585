import math
import sys

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
        c_count = D.class_token_count(c)

        for w in V:
            loglikelihood[(w, c)] = math.log(
                (D.word_class_token_count(w, c) + 1) / (c_count + V_len)
            )
            i += 1
            process_bar(i, total_words)
            # progress = (i / total_words) * 100
            # sys.stdout.write("\r")
            # sys.stdout.write(
            #     "[{:<50}] {:.2f}%".format("=" * int(progress // 2), progress)
            # )
            # sys.stdout.flush()

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

import math
import sys


# Naive Bayes: Training/Testing
# lecture 10, CS585_Lecture_February12th.pdf


def train(D, C):
    logprior = {}
    loglikelihood = {}
    V = D.vocabulary()
    V_len = len(V)
    total_words = V_len * len(C)
    i = 0
    for c in C:
        N_doc = D.total_doc_num()
        N_c = D.c_doc_num(c)
        logprior[c] = math.log(N_c / N_doc)
        c_count = D.c_count(c)

        for w in V:
            loglikelihood[(w, c)] = math.log((D.wc_count(w, c) + 1) / (c_count + V_len))
            i += 1
            progress = (i / total_words) * 100
            sys.stdout.write("\r")
            sys.stdout.write(
                "[{:<50}] {:.2f}%".format("=" * int(progress // 2), progress)
            )
            sys.stdout.flush()

    return logprior, loglikelihood, V


def test(testdoc, logprior, loglikelihood, C, V):
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

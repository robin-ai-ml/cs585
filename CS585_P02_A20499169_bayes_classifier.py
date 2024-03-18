import math


# Naive Bayes: Training/Testing
# lecture 10, CS585_Lecture_February12th.pdf


def train(D, C):
    logprior = {}
    loglikelihood = {}
    V = D.vocabulary()
    for c in C:
        N_doc = D.total_doc_num()
        N_c = D.doc_num_of_class(c)
        logprior[c] = math.log(N_c / N_doc)

        for w in V:
            loglikelihood[(w, c)] = math.log(
                (D.word_count(w, c) + 1) / (D.class_count(c) + V.size())
            )
    return logprior, loglikelihood, V


def test(testdoc, logprior, loglikelihood, C, V):
    sum = {}
    max_sum = -9999
    max_c = None
    for c in C:
        sum[c] = logprior[c]
        for word in testdoc:
            if V.contains(word):
                sum[c] += loglikelihood[(word, c)]
        if sum[c] > max_sum:
            max_sum = sum[c]
            max_c = c

    return max_c

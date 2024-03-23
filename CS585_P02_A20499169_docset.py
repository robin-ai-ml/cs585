import pandas as pd
import os


# class Doc:
#     def __init__(self, row):
#         self.label = row["Label"]
#         self.bag = {}
#         for w in row["Text"].split():
#             if w not in self.bag:
#                 self.bag[w] = 0
#             self.bag[w] += 1
#
#     def words(self):
#         return self.bag.keys()


# def inc[T](d: dict[T, int], key: T, num: int):
def inc(d, key, num: int):
    if key not in d:
        d[key] = 0
    d[key] += num


class Docset:
    # docs: list[tuple[dict[str, int], str]]
    # C: dict[str, int] = {}
    # WC: dict[tuple[str, str], int] = {}

    V: dict[str, int] = {}
    _class_token_count: dict[str, int] = {}
    _class_doc_count: dict[str, int] = {}

    _word_class_token_count: dict[tuple[str, str], int] = {}
    _word_class_doc_count: dict[tuple[str, str], int] = {}

    _total_doc_count: int = 0

    def __init__(self, docs: list[tuple[dict[str, int], str]]) -> None:
        self.docs = docs
        # for _, row in df[["Text", "Label"]].iterrows():
        #     self.docs.append(Doc(row))

    def add(self, bag: dict[str, int], label: str):
        self._total_doc_count += 1
        # bag of binary
        # if label not in self.class_doc_count:
        #     self.class_doc_count[label] = 0
        # self.class_doc_count[label] += 1
        inc(self._class_doc_count, label, 1)

        for w in bag:
            # if w not in self.V:
            #     self.V[w] = 0
            # self.V[w] += 1

            # if (w, label) not in self.word_class_doc_count:
            #     self.word_class_doc_count[(w, label)] = 0
            # self.word_class_doc_count[(w, label)] += 1
            inc(self.V, w, 1)
            inc(self._word_class_doc_count, (w, label), 1)
            inc(self._word_class_token_count, (w, label), bag[w])

            inc(self._class_token_count, label, bag[w])

    def vocabulary(self):
        return self.V.keys()

    def total_doc_count(self) -> int:
        return self._total_doc_count

    def class_doc_count(self, c: str) -> int:
        return self._class_doc_count[c]
        # return self.C[c]
        # return sum(map(lambda x: x[1] == c, self.docs))
        # return len(self.df[self.df["Label"] == c])

    def class_token_count(self, c: str):
        return self._class_token_count[c]

    def word_class_doc_count(self, w: str, c: str):
        if (w, c) in self._word_class_doc_count:
            return self._word_class_doc_count[(w, c)]
        return 0
        # return sum(map(lambda x: x[1] == c and w in x[0], self.docs))
        # return len(self.df[(self.df["Label"] == c) & (self.df["Text"].str.contains(w))])
        # s = 0
        # for _, row in self.df[["Text", "Label"]].iterrows():
        #     if row["Label"] == c:
        #         if w in row["Text"]:
        #             s += 1
        # s += row["Text"].split().count(w)
        # return s

    def word_class_token_count(self, w: str, c: str):
        if (w, c) in self._word_class_token_count:
            return self._word_class_token_count[(w, c)]
        return 0

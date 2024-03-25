def inc(d, key, num: int):
    if key not in d:
        d[key] = 0
    d[key] += num


class Docset:
    V: dict[str, int] = {}
    _class_token_count: dict[str, int] = {}
    _class_doc_count: dict[str, int] = {}

    _word_class_token_count: dict[tuple[str, str], int] = {}
    _word_class_doc_count: dict[tuple[str, str], int] = {}

    _total_doc_count: int = 0

    def add(self, bag: dict[str, int], label: str):
        self._total_doc_count += 1
        inc(self._class_doc_count, label, 1)

        for w in bag:
            inc(self.V, w, 1)
            inc(self._word_class_doc_count, (w, label), 1)
            inc(self._word_class_token_count, (w, label), bag[w])

            inc(self._class_token_count, label, bag[w])

    def vocabulary(self):
        return self.V.keys()

    def total_doc_count(self) -> int:
        return self._total_doc_count

    # count of documents with class `c`
    def class_doc_count(self, c: str) -> int:
        return self._class_doc_count[c]

    # count of tokens in documents with class `c`
    def class_token_count(self, c: str):
        return self._class_token_count[c]

    # count of documents with both word `w` and class `c`
    def word_class_doc_count(self, w: str, c: str):
        if (w, c) in self._word_class_doc_count:
            return self._word_class_doc_count[(w, c)]
        return 0

    # count of word `w` in documents with class `c`
    def word_class_token_count(self, w: str, c: str):
        if (w, c) in self._word_class_token_count:
            return self._word_class_token_count[(w, c)]
        return 0

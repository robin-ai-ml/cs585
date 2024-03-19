import pandas as pd
import os


class Doc:
    def __init__(self, row):
        self.label = row["Label"]
        self.bag = {}
        for w in row["Text"].split():
            if w not in self.bag:
                self.bag[w] = 0
            self.bag[w] += 1

    def words(self):
        return self.bag.keys()


class Docset:
    def __init__(self, df) -> None:
        self.docs = []
        for _, row in df[["Text", "Label"]].iterrows():
            self.docs.append(Doc(row))

    def vocabulary(self):
        V = set()
        for doc in self.docs:
            V.update(doc.words())
        return V

    def total_doc_num(self) -> int:
        return len(self.docs)

    def c_doc_num(self, c) -> int:
        return sum(map(lambda x: x.label == c, self.docs))
        # return len(self.df[self.df["Label"] == c])

    def wc_count(self, w, c):
        return sum(map(lambda x: x.label == c and w in x.bag, self.docs))
        # return len(self.df[(self.df["Label"] == c) & (self.df["Text"].str.contains(w))])
        # s = 0
        # for _, row in self.df[["Text", "Label"]].iterrows():
        #     if row["Label"] == c:
        #         if w in row["Text"]:
        #             s += 1
        # s += row["Text"].split().count(w)
        # return s

    def c_count(self, c):
        # return len(self.df[self.df["Label"] == c])
        return sum(map(lambda x: x.label == c, self.docs))
        # s = 0
        # for _, row in self.df[["Text", "Label"]].iterrows():
        #     if row["Label"] == c:
        #         s += 1
        #         # s += len(row["Text"].split())
        # return s


if __name__ == "__main__":
    dataset_file = "Reviews2.csv"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = cur_dir + "/" + dataset_file
    df = pd.read_csv(csv_file_path, header=0)
    print(df.head())

    D = Docset(df)
    print("total", D.total_doc_num())
    print("good", D.c_doc_num("good"))
    print("bad", D.c_doc_num("bad"))
    print("fresh, good", D.wc_count("fresh", "good"))
    print("good", D.c_count("good"))

    V = D.vocabulary()
    print("V len", len(V))

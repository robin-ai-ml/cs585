import pandas as pd
import os


class Dataset:
    def __init__(self, df) -> None:
        self.df = df

    def vocabulary(self):
        V = set()
        for text in self.df["Text"]:
            V.update(text.split())
        return V

    def total_doc_num(self) -> int:
        return len(self.df)

    def c_doc_num(self, c) -> int:
        return len(self.df[self.df["Label"] == c])

    def wc_count(self, w, c):
        s = 0
        for _, row in self.df[["Text", "Label"]].iterrows():
            if row["Label"] == c:
                s += row["Text"].split().count(w)
        return s

    def c_count(self, c):
        s = 0
        for _, row in self.df[["Text", "Label"]].iterrows():
            if row["Label"] == c:
                s += len(row["Text"].split())
        return s


if __name__ == "__main__":
    dataset_file = "Reviews2.csv"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = cur_dir + "/" + dataset_file
    df = pd.read_csv(csv_file_path, header=0)
    print(df.head())

    D = Dataset(df)
    print("total", D.total_doc_num())
    print("good", D.c_doc_num("good"))
    print("bad", D.c_doc_num("bad"))
    print("fresh, good", D.wc_count("fresh", "good"))
    print("good", D.c_count("good"))

    V = D.vocabulary()
    print("V len", len(V))

    for w in V:
        print(w)

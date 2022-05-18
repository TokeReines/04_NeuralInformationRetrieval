import pyterrier as pt
import pandas as pd

if __name__ == '__main__':
    pt.init()
    docs_df = pd.read_csv('data/lab_docs.csv', dtype=str)
    print(docs_df.shape)
    print(docs_df.head())

    indexer = pt.DFIndexer("./indexes/default", overwrite=True)
    index_ref = indexer.index(docs_df["text"], docs_df["docno"])
    index = pt.IndexFactory.of(index_ref)
    pt.BatchRetrieve(index, num_results=10, wmodel=)
    print(index_ref.toString())
    index = pt.IndexFactory.of(index_ref)
    a = 2

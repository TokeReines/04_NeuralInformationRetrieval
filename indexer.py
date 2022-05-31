from collections import defaultdict

import numpy as np
import pyterrier as pt
import pandas as pd

import json
from time import time

if __name__ == '__main__':
    INDEX_BASE_PATH = "./indexes"
    INDEX_PATH_NO_PREPROCESSING = f"{INDEX_BASE_PATH}/no_preprocessing"
    INDEX_PATH_DEFAULT = f"{INDEX_BASE_PATH}/default"
    INDEX_PATH_DEFAULT_POSITIONS = f"{INDEX_BASE_PATH}/default_positions"
    INDEX_PATH_STOPWORDS = f"{INDEX_BASE_PATH}/stopwords"
    INDEX_PATH_STEMMING = f"{INDEX_BASE_PATH}/stemming"

    CORPUS_PATH = "./data/corpus.jsonl"
    THREADS = 6

    if not pt.started():
        pt.init()


    def get_corpus(path: str):
        """
        Loads the corpus from the given path.
        :param path: Path to the corpus.
        :return: Generator of documents (lines in the document/path).
        """
        with open(path, "r") as f:
            for line in f:
                line_dict = json.loads(line)
                line_dict['docno'] = line_dict.pop('_id')
                yield line_dict

    # %%


    def generate_index(index_path, termpipelines=None, overwrite=True, blocks=False):
        iter_indexer = pt.IterDictIndexer(
            index_path,
            overwrite=overwrite,
            meta=["docno", "title", "text"],
            meta_lengths=[20, 256, 4096],
            threads=THREADS,
            blocks=blocks
        )
        if termpipelines is not None:
            iter_indexer.setProperty("termpipelines", termpipelines)

        iter_indexer.index(get_corpus(CORPUS_PATH), fields=["title", "text"])
        index = pt.IndexFactory.of(index_path)
        return index

    # t = time()
    # index = generate_index(INDEX_PATH_DEFAULT_POSITIONS, termpipelines="Stopwords,PorterStemmer", blocks=True)
    # m, s = divmod(time() - t, 60)
    # print(f"Indexing with blocks/positions and both stemming and stopword removal took {m:.02f} minutes and {s:.02f} seconds")
    #
    # t = time()
    # index = generate_index(INDEX_PATH_DEFAULT, termpipelines="Stopwords,PorterStemmer")
    # m, s = divmod(time() - t, 60)
    # print(f"Indexing with both stemming and stopword removal took {m:.02f} minutes and {s:.02f} seconds")

    # t = time()
    # index = generate_index(INDEX_PATH_NO_PREPROCESSING, termpipelines="")
    # m, s = divmod(time() - t, 60)
    # print(f"Indexing with no preprocessing took {m:.02f} minutes and {s:.02f} seconds")
    #
    # t = time()
    # index = generate_index(INDEX_PATH_STOPWORDS, termpipelines="Stopwords")
    # m, s = divmod(time() - t, 60)
    # print(f"Indexing with only stopwords took {m:.02f} minutes and {s:.02f} seconds")
    #
    # t = time()
    # index = generate_index(INDEX_PATH_STEMMING, termpipelines="PorterStemmer")
    # m, s = divmod(time() - t, 60)
    # print(f"Indexing with only PorterStemmer took {m:.02f} minutes and {s:.02f} seconds")

    # Looking into index statistics

    def get_index(index_path: str):
        index = pt.IndexFactory.of(index_path)
        return index

    # index = get_index(INDEX_PATH_DEFAULT_POSITIONS)
    # print("Default + postings", index.getCollectionStatistics().toString())
    #
    # index = get_index(INDEX_PATH_DEFAULT)
    # print("Default", index.getCollectionStatistics().toString())
    #
    # index = get_index(INDEX_PATH_STOPWORDS)
    # print("Stopwords", index.getCollectionStatistics().toString())
    #
    # index = get_index(INDEX_PATH_STEMMING)
    # print("Stemming", index.getCollectionStatistics().toString())
    #
    # index = get_index(INDEX_PATH_NO_PREPROCESSING)
    # print("Nothing", index.getCollectionStatistics().toString())


    def create_folds(num_folds: int, df):
        df_size = len(df)
        fold_size = df_size // num_folds
        for n in range(num_folds):
            if n == num_folds:
                end = df_size
            else:
                end = (n + 1) * fold_size

            start = n * fold_size
            yield df.iloc[start:end]


    query_df = pd.read_csv('data/train_query.csv', dtype=str)
    #query_df = pd.read_csv('data/lab_topics.csv', dtype=str)

    # Load qrels
    qrels_df = pd.read_csv('data/train_qrel.csv', dtype=str)
    #qrels_df = pd.read_csv('data/lab_qrels.csv', dtype=str)
    qrels_df['label'] = qrels_df['label'].astype('int32')
    range(0, len(query_df), len(query_df) // 3)


    num_folds = 3
    query_folds = list(create_folds(num_folds, query_df))
    for index_name in [INDEX_PATH_NO_PREPROCESSING,INDEX_PATH_DEFAULT,INDEX_PATH_DEFAULT_POSITIONS,
                       INDEX_PATH_STOPWORDS,INDEX_PATH_STEMMING]:
        index = get_index(index_name)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.b": 0.7, "bm25.k_1": 0.75, "bm25.k_3": 0.75})
        dir_lm = pt.BatchRetrieve(index, wmodel="DirichletLM", controls={"dirichletlm.mu": 2500})
        print(f"\n\n{index_name}: {index.getCollectionStatistics().toString()}")
        print(pt.Experiment([bm25, dir_lm], query_df, qrels_df, ["map", "ndcg"]))



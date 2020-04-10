# Processor for corpus used in official Seq2Seq Machine Translation Tutorial:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from dataset.core import Dataset, Processor
import pandas as pd
import os


class EngFraDataProcessor(Processor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df_all = None

    def _load_corpus(self):
        df = pd.read_csv(os.path.join(self.data_dir, "eng-fra.txt"), sep="\t", header=None, names=("eng", "fra"))
        self.df_all = df

    def get_train_data(self):
        if not self.df_all:
            self._load_corpus()
        return self.df_all[(self.df_all.index % 10) != 0]

    def get_dev_data(self):
        if not self.df_all:
            self._load_corpus()
        return self.df_all[(self.df_all.index % 10) == 0]

    def get_test_data(self):
        if not self.df_all:
            self._load_corpus()
        return self.df_all[(self.df_all.index % 10) == 0]

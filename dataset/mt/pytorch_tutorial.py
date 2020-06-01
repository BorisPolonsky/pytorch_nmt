# Processor for corpus used in official Seq2Seq Machine Translation Tutorial:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from dataset.core import Dataset, Processor
import pandas as pd
import os
import unicodedata


class EngFraDataProcessor(Processor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df_all = None

    def get_train_data(self):
        if not self.df_all:
            self.build()
        return Dataset(self.df_all[(self.df_all.index % 10) != 0])

    def get_dev_data(self):
        if not self.df_all:
            self.build()
        return Dataset(self.df_all[(self.df_all.index % 10) == 0])

    def get_test_data(self):
        if not self.df_all:
            self.build()
        return Dataset(self.df_all[(self.df_all.index % 10) == 0])

    def build(self):
        def normalize_string(s):
            s = s.rstrip().lower()
            return "".join(filter(lambda ch: unicodedata.category(ch) != 'Mn', unicodedata.normalize("NFD", s)))

        def tokenize(s):
            tokens = []
            # for token in s.split(" "):
            for token in s.split():  # there are some special white spaces in the corpus
                if len(token) > 1 and unicodedata.category(token[-1]) == "Po":
                    tokens.append(token[:-1])
                    tokens.append(token[-1])
                else:
                    tokens.append(token)
            return tokens

        def extend_vocab(vocab, redundancy=10):
            special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
            while len(special_tokens) < redundancy:
                special_tokens.append("[unused{}]".format(len(special_tokens) + 1))
            return special_tokens + vocab

        def convert_tokens2ids(tokens, token2id, oov_id):
            token_ids = [token2id.get(token, oov_id) for token in tokens]
            return token_ids

        print("Building data set.")
        df = pd.read_csv(os.path.join(self.data_dir, "eng-fra.txt"), sep="\t", header=None, names=["eng", "fra"])
        text_english, text_french = df.eng, df.fra

        tokens_english = text_english.apply(normalize_string).apply(tokenize)
        word_freq_english = tokens_english.explode().value_counts()
        vocab_english = word_freq_english[word_freq_english > 2].index.unique().to_list()
        vocab_english = extend_vocab(vocab_english)
        token2id_english = {token: i for i, token in enumerate(vocab_english)}

        tokens_french = text_french.apply(normalize_string).apply(tokenize)
        word_freq_french = tokens_french.explode().value_counts()
        vocab_french = word_freq_french[word_freq_french > 2].index.unique().to_list()
        vocab_french = extend_vocab(vocab_french)
        token2id_french = {token: i for i, token in enumerate(vocab_french)}

        # Determine src and target language
        src_text, src_tokens, src_token2id, src_oov_id = text_english, tokens_english,  token2id_english, token2id_english["[UNK]"]
        target_text, target_tokens, target_token2id, target_oov_id = text_french, tokens_french, token2id_french, token2id_french["[UNK]"]

        src_token_ids = src_tokens.apply(lambda x: convert_tokens2ids(x, src_token2id, src_oov_id))
        target_tokens = target_tokens.apply(lambda x: ["[BOS]"] + x + ["[EOS]"])
        target_token_ids = target_tokens.apply(lambda x: convert_tokens2ids(x, target_token2id, target_oov_id))
        df = pd.DataFrame({"text_src": src_text,
                           "text_target": target_text,
                           "tokens_src": src_tokens,
                           "tokens_target": target_tokens,
                           "token_ids_src": src_token_ids,
                           "token_ids_target": target_token_ids})
        self.df_all = df
        print("Build complete: vocab_size_src: {}, vocab_size_target: {}".format(len(src_token2id), len(target_token2id)))

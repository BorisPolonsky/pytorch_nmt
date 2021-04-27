# Source: https://github.com/brightmart/nlp_chinese_corpus

from dataset.core import Dataset, Processor
import pandas as pd
import os


class Translation2019ZhProcessor(Processor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.TRAINING_SET = "translation2019zh_train.json"
        self.VALIDATION_SET = "translation2019zh_valid.json"

    def get_train_data(self):
        return self._read_from_file(os.path.join(self.data_dir, self.TRAINING_SET))

    def get_dev_data(self):
        return self._read_from_file(os.path.join(self.data_dir, self.VALIDATION_SET))

    def get_test_data(self):
        return self._read_from_file(os.path.join(self.data_dir, self.VALIDATION_SET))

    @classmethod
    def _read_from_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            df = pd.read_json(f, orient="records", lines=True, convert_dates=False)
        df.rename(columns={"english": "text_src", "chinese": "text_target"}, inplace=True)
        return Dataset(df)

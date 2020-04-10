from torch.utils.data import Dataset as PyTorchDataset
import pandas as pd


class Dataset(PyTorchDataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        return {col: self.df.iloc[index, :][col] for col in self.df.columns}

    def __len__(self):
        return len(self.df)

    def transform(self, transform_callable):
        self.df = pd.DataFrame([transform_callable(record.to_dict()) for _, record in self.df.iterrows()],
                               index=self.df.index)
        return self


class Processor:
    def __init__(self, data_dir):
        pass

    def get_train_data(self):
        pass

    def get_dev_data(self):
        pass

    def get_test_data(self):
        pass

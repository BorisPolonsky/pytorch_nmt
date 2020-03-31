from models.seq2seq import Seq2SeqAttn
from tokenization import FullTokenizer
import pandas as pd
import torch
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import DataLoader
import os
import functools
from models.util import sequence_mask
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch, device=None):
    def get_seq_length(token_ids_list):
        seq_length = list(map(lambda x: x.shape[0], token_ids_list))
        seq_length = torch.tensor(seq_length, dtype=torch.int64)
        return seq_length

    token_ids_src, token_ids_target = [], []
    for record in batch:
        token_ids_src.append(torch.tensor(record["token_ids_src"], dtype=torch.int64, device=device))
        token_ids_target.append(torch.tensor(record["token_ids_target"], dtype=torch.int64, device=device))
    seq_length_src = get_seq_length(token_ids_src)
    seq_length_target = get_seq_length(token_ids_src)
    return {"token_ids_src": token_ids_src, "token_ids_target": token_ids_target,
            "seq_length_src": seq_length_src, "seq_length_target": seq_length_target}


class PandasDataset(PyTorchDataset):
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


cache = "./cache.pkl"
tokenizer_src = FullTokenizer("./data/vocab-eng.txt", do_lower_case=True)
tokenizer_target = FullTokenizer("./data/vocab-fra.txt", do_lower_case=True)

if os.path.exists(cache):
    df = pd.read_pickle(cache)
else:
    df = pd.read_csv("./data/eng-fra.txt", sep="\t", header=None)
    df.columns = ("eng", "fra")

    tokens_src = df.eng.apply(tokenizer_src.tokenize)
    token_ids_src = tokens_src.apply(tokenizer_src.convert_tokens_to_ids)

    tokens_target = df.fra.apply(lambda x: ["[BOS]"] + tokenizer_target.tokenize(x) + ["[EOS]"])
    token_ids_target = tokens_target.apply(tokenizer_target.convert_tokens_to_ids)

    df = pd.DataFrame({"tokens_src": tokens_src,
                       "token_ids_src": token_ids_src,
                       "tokens_target": tokens_target,
                       "token_ids_target": token_ids_target})
    df.to_pickle(cache)

training_set = PandasDataset(df)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
n_iter = 0
enc_hidden_dim = 100
dec_hidden_dim = 200
embd_dim_src = len(tokenizer_src.vocab)
embd_dim_target = len(tokenizer_target.vocab)

nn = Seq2SeqAttn(vocab_size_f=embd_dim_src, embedding_dim_f=64, vocab_size_e=embd_dim_target, embedding_dim_e=64,
                 enc_hidden_dim=enc_hidden_dim,
                 dec_hidden_dim=dec_hidden_dim).to(device)

lr = 0.1
optimizer = torch.optim.SGD(nn.parameters(), lr=lr)
print([name for name, param in nn.named_parameters()])
cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")


def loss_fn(inputs, target, sequence_length):
    max_len = inputs.size(1)
    inputs = inputs.transpose(1, 2)  # [batch_size, n_cls, seq_len]
    loss = F.cross_entropy(inputs, target, reduction="none")  # [batch_size, seq_len]

    mask = sequence_mask(sequence_length, maxlen=max_len).to(loss.device)
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    loss = loss.sum() / sequence_length.sum()
    return loss


writer = SummaryWriter()
writer.close()
for batch in DataLoader(training_set,
                        shuffle=True,
                        batch_size=64,
                        collate_fn=functools.partial(collate_fn, device=device)):
    n_iter += 1
    padded_inputs_enc = torch.nn.utils.rnn.pad_sequence(batch["token_ids_src"],
                                                        batch_first=True,
                                                        padding_value=0)
    padded_inputs_dec = torch.nn.utils.rnn.pad_sequence(batch["token_ids_target"],
                                                        batch_first=True,
                                                        padding_value=0)
    # Right shift decoder output by one
    padded_outputs_dec = padded_inputs_dec[:, 1:]
    padded_inputs_dec = padded_inputs_dec[:, :-1]
    seq_length_enc = batch["seq_length_src"]
    seq_length_decoder = batch["seq_length_target"] - 1
    seq_length_decoder = torch.where(seq_length_decoder >= 0, seq_length_decoder, torch.zeros_like(seq_length_decoder))
    optimizer.zero_grad()
    logits = nn(padded_inputs_enc, seq_length_enc, padded_inputs_dec, seq_length_decoder)
    loss = loss_fn(logits, padded_outputs_dec, seq_length_decoder)
    loss.backward()
    optimizer.step()
    model_path = "./model.pkl"
    with open("model_path", "wb") as f:
        torch.save(nn, f)
    nn = torch.load(f)

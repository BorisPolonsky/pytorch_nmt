from models.seq2seq import Seq2SeqAttn
from tokenization import FullTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset.registry import registry
from dataset.core import Dataset
import os
import functools
from models.util import sequence_mask
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


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
    seq_length_target = get_seq_length(token_ids_target)
    return {"token_ids_src": token_ids_src, "token_ids_target": token_ids_target,
            "seq_length_src": seq_length_src, "seq_length_target": seq_length_target}


def loss_fn(inputs, target, sequence_length):
    max_len = inputs.size(1)
    inputs = inputs.transpose(1, 2)  # [batch_size, n_cls, seq_len]
    loss = F.cross_entropy(inputs, target, reduction="none")  # [batch_size, seq_len]

    mask = sequence_mask(sequence_length, maxlen=max_len).to(loss.device)
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    loss = loss.sum() / sequence_length.sum()
    return loss


def inspect_batch(tokenizer_src, tokenizer_target, enc_input, dec_input, dec_output):
    enc_input_tokens = [tokenizer_src.convert_ids_to_tokens(item.tolist()) for item in enc_input]
    dec_input_tokens = [tokenizer_target.convert_ids_to_tokens(item.tolist()) for item in dec_input]
    dec_output_tokens = [tokenizer_target.convert_ids_to_tokens(item.tolist()) for item in dec_output]
    print("enc_input", enc_input_tokens)
    print("dec_input", dec_input_tokens)
    print("dec_output", dec_output_tokens)


def main(args):
    output_dir = args.output_dir
    tokenizer_src = FullTokenizer("./data/eng-fra/vocab-eng.txt", do_lower_case=True)
    tokenizer_target = FullTokenizer("./data/eng-fra/vocab-fra.txt", do_lower_case=True)
    training_set_cache = os.path.join(output_dir, "cache", "train.pkl")
    data_dir = "./data/eng-fra"
    processor = registry.dataset["pytorch-seq2seq-tutorial"](data_dir=data_dir)

    if os.path.exists(training_set_cache):
        training_set = Dataset.load(training_set_cache)
    else:
        training_set = processor.get_train_data()
    print(training_set.df.head())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    exit(0)
    n_iter = 0
    embd_dim_src = 64
    embd_dim_target = 64
    enc_hidden_dim = 100
    dec_hidden_dim = 200
    vocab_size_src = len(tokenizer_src.vocab)
    vocab_size_target = len(tokenizer_target.vocab)

    nn = Seq2SeqAttn(vocab_size_src=vocab_size_src,
                     embedding_dim_src=embd_dim_src,
                     vocab_size_target=vocab_size_target,
                     embedding_dim_target=embd_dim_target,
                     enc_hidden_dim=enc_hidden_dim,
                     dec_hidden_dim=dec_hidden_dim).to(device)

    lr = 0.1
    lr_decay = 0.99
    optimizer = torch.optim.SGD(nn.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_decay)

    print([name for name, param in nn.named_parameters()])

    model_dir = os.path.join(output_dir, "state_dict")

    writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
    num_epoch = 20
    batch_size = 64
    for epoch_i in range(num_epoch):
        print("Epoch {}".format(epoch_i))
        for batch in DataLoader(training_set,
                                shuffle=True,
                                batch_size=batch_size,
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
            # teacher_enforcing
            seq_permute_rate = 0.2
            token_permute_rate = 0.2
            teacher_enforcing_mask = torch.rand(padded_inputs_dec.size(), device=padded_inputs_dec.device) >= token_permute_rate
            teacher_enforcing_mask = (torch.rand([padded_inputs_dec.size(0), 1], device=padded_inputs_dec.device) >= seq_permute_rate) | teacher_enforcing_mask
            permuted_inputs_dec = torch.randint_like(padded_inputs_dec, 0, vocab_size_target)
            padded_inputs_dec = torch.where(teacher_enforcing_mask, padded_inputs_dec, permuted_inputs_dec)
            seq_length_enc = batch["seq_length_src"]
            seq_length_decoder = batch["seq_length_target"] - 1
            seq_length_decoder = torch.where(seq_length_decoder >= 0, seq_length_decoder, torch.zeros_like(seq_length_decoder))

            # inspect_batch(tokenizer_src, tokenizer_target, padded_inputs_enc, padded_inputs_dec, padded_outputs_dec)
            optimizer.zero_grad()
            logits = nn(padded_inputs_enc, seq_length_enc, padded_inputs_dec, seq_length_decoder)
            loss = loss_fn(logits, padded_outputs_dec, seq_length_decoder)
            loss.backward()
            writer.add_scalar("loss", loss, global_step=n_iter)
            writer.add_scalar("lr", lr_scheduler.get_lr()[0], global_step=n_iter)
            optimizer.step()
        lr_scheduler.step()

        with open(os.path.join(model_dir, "model-{}.pkl".format(n_iter)), "wb") as f:
            torch.save(nn.state_dict(), f)

    with open(os.path.join(model_dir, "model-16984.pkl"), "rb") as f:
        nn.load_state_dict(torch.load(f))

    test_set = training_set
    batch_size = 1
    bos_id = tokenizer_target.convert_tokens_to_ids(["[BOS]"])[0]
    eos_id = tokenizer_target.convert_tokens_to_ids(["[EOS]"])[0]
    n_beam = 1
    max_output_length = 20
    print("bos_id: {}, eos_id: {}".format(bos_id, eos_id))

    for batch in DataLoader(test_set,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=functools.partial(collate_fn, device=device)):
        n_iter += 1
        tokens_target = [tokenizer_target.convert_ids_to_tokens(item.cpu().numpy()) for item in batch["token_ids_target"]]

        inputs_enc = torch.nn.utils.rnn.pad_sequence(batch["token_ids_src"],
                                                     batch_first=True,
                                                     padding_value=0)
        seq_length_enc = batch["seq_length_src"]
        dec_init_state = torch.zeros([batch_size, dec_hidden_dim], device=device)
        decoder_init_input = torch.empty([batch_size], dtype=torch.int64, device=device).fill_(bos_id)
        outputs, output_lengths = nn.beam_search(inputs_enc,
                                                 seq_length_enc,
                                                 n_beam=n_beam,
                                                 eos_id=eos_id,
                                                 decoder_init_input=decoder_init_input,
                                                 decoder_init_state=dec_init_state,
                                                 max_length=max_output_length)
        print(outputs, output_lengths)
        for pred, ref in zip(outputs, tokens_target):
            print("Reference:", ref)
            for candidate_i, pred in enumerate(pred):
                pred = tokenizer_target.convert_ids_to_tokens(pred.tolist())
                print("Candidate {}: {}".format(candidate_i, pred))


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=os.path.normpath, help="Directory for storing serilized dataset, model, evaluation results e.t.c..")
    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args)

from models.seq2seq import Seq2SeqAttn
from tokenization import FullTokenizer
import torch
from torch.utils.data import DataLoader
from dataset.registry import registry
from dataset.core import Dataset
import os
import functools
from typing import List
from models.util import sequence_mask
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import json
import itertools
from parallel import DataParallelCriterion
import re
import glob
import sys


class DataParallel(torch.nn.DataParallel):
    """
    Dropped final gathering op, return results as is from its
    corresponding devices
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in itertools.chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        # return self.gather(outputs, self.output_device)
        return outputs


class MaskedCrossEntropyLoss(torch.nn.Module):
    def forward(self, inputs, target, sequence_length):
        max_len = inputs.size(1)
        inputs = inputs.transpose(1, 2)  # [batch_size, n_cls, seq_len]
        loss = F.cross_entropy(inputs, target, reduction="none")  # [batch_size, seq_len]

        mask = sequence_mask(sequence_length, maxlen=max_len).to(loss.device)
        loss = torch.where(mask, loss, torch.zeros_like(loss))
        loss = loss.sum()
        return loss


def collate_fn(batch, device=None):
    def get_seq_length(token_ids_list):
        seq_length = list(map(lambda x: x.shape[0], token_ids_list))
        seq_length = torch.tensor(seq_length, dtype=torch.int64)
        return seq_length
    # Mandatory
    token_ids_src, token_ids_target = [], []
    # Optional
    optional_fields = "text_src", "text_target", "tokens_src", "tokens_target"
    output = {}
    for record in batch:
        token_ids_src.append(torch.tensor(record["token_ids_src"], dtype=torch.int64, device=device))
        token_ids_target.append(torch.tensor(record["token_ids_target"], dtype=torch.int64, device=device))
        for field in optional_fields:
            if field not in record:
                continue
            if field not in output:
                output[field] = []
            output[field].append(record[field])
    seq_length_src = get_seq_length(token_ids_src)
    seq_length_target = get_seq_length(token_ids_target)
    output.update({
        "token_ids_src": token_ids_src,
        "token_ids_target": token_ids_target,
        "seq_length_src": seq_length_src,
        "seq_length_target": seq_length_target
    })
    return output


def inspect_batch(tokenizer_src, tokenizer_target, enc_input, dec_input, dec_output):
    enc_input_tokens = [tokenizer_src.convert_ids_to_tokens(item.tolist()) for item in enc_input]
    dec_input_tokens = [tokenizer_target.convert_ids_to_tokens(item.tolist()) for item in dec_input]
    dec_output_tokens = [tokenizer_target.convert_ids_to_tokens(item.tolist()) for item in dec_output]
    print("enc_input", enc_input_tokens)
    print("dec_input", dec_input_tokens)
    print("dec_output", dec_output_tokens)


def get_tokenizer(config):
    """

    :param config: {"tokenizer": "tokenizer-type", "args": {"key1": "value1"}}
    :return: an object with method `tokenize`, `convert_tokens_to_ids` and `convert_ids_to_tokens`.
    """
    tokenizer_cls_name = config["tokenizer"]
    tokenizer_cls = FullTokenizer  # no support for other tokenizers for now
    kwargs = config.get("args", dict())
    tokenizer = tokenizer_cls(**kwargs)
    return tokenizer


def apply_tokenization(dataset: Dataset, tokenizer_src, tokenizer_target):
    def add_tokens_n_ids(df, tokenizer, group):
        token_column = "tokens_{}".format(group)
        token_id_column = "token_ids_{}".format(group)
        if token_column not in df.columns:
            df[token_column] = df["text_{}".format(group)].apply(tokenizer.tokenize)
        if token_id_column not in df.columns:
            df[token_id_column] = df[token_column].apply(tokenizer.convert_tokens_to_ids)

    add_tokens_n_ids(dataset.df, tokenizer_src, "src")
    add_tokens_n_ids(dataset.df, tokenizer_target, "target")


def get_last_checkpoint(checkpoint_dir: str):
    checkpoints = []
    for path in glob.iglob(os.path.join(checkpoint_dir, "model-*")):
        match = re.search("model-([0-9]*).(pt|pkl)$", path)
        if match:
            n_iter = int(match.group(1))
            checkpoints.append((n_iter, path))
    if not checkpoints:
        raise ValueError("No state_dict found in ".format(checkpoint_dir))
    n_iter, last_checkpoint = max(checkpoints, key=lambda x: x[0])
    return last_checkpoint


def join_sub_tokens(tokens: List[str]) -> List[str]:
    output = []
    for token in tokens:
        if token.startswith("##"):
            sub_token = token[2:]
            if output:
                output[-1] = output[-1] + sub_token
            else:
                output.append(sub_token)
        else:
            output.append(token)
    return output


def main(args):
    output_dir = args.output_dir
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_config = config["model"]
    model_args = model_config.get("args", dict())
    cache_dir = os.path.join(output_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_dir = args.data_dir
    processor = registry.dataset[args.processor](data_dir=data_dir)

    tokenization_config = config["tokenization"]
    tokenizer_src = get_tokenizer(tokenization_config["source"])
    tokenizer_target = get_tokenizer(
        tokenization_config["target"]) if "target" in tokenization_config else tokenizer_src

    vocab_size_src = len(tokenizer_src.vocab)
    vocab_size_target = len(tokenizer_target.vocab)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    n_iter = 0

    use_multi_device = torch.cuda.device_count() > 1
    nn = Seq2SeqAttn(**model_args)
    loss_fn = MaskedCrossEntropyLoss()
    if use_multi_device:
        nn.encoder.flatten_parameters = True
        wrapped_nn = DataParallel(nn).to(device)
        nn = wrapped_nn.module  # Just in case
        loss_fn = DataParallelCriterion(loss_fn)
    else:
        wrapped_nn = nn = nn.to(device)

    print([name for name, param in nn.named_parameters()])

    optimizer_config = config["optimizer"]
    optimizer_cls = torch.optim.SGD  # No support for other optimizers for now
    optimizer_args = optimizer_config.get("args", dict())
    optimizer = optimizer_cls(wrapped_nn.parameters(), **optimizer_args)

    lr_decay = config["lr_schedule"]["decay"]
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_decay)

    model_dir = os.path.join(output_dir, "state_dict")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)
    if args.do_train:
        print("Preparing training data.")
        training_set_cache = os.path.join(cache_dir, "train.pkl")
        if os.path.exists(training_set_cache):
            training_set = Dataset.load(training_set_cache)
        else:
            training_set = processor.get_train_data()
            training_set.save(training_set_cache)
        apply_tokenization(training_set, tokenizer_src, tokenizer_target)
        print(training_set.df.head())

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
                teacher_enforcing_mask = torch.rand(padded_inputs_dec.size(),
                                                    device=padded_inputs_dec.device) >= token_permute_rate
                teacher_enforcing_mask = (torch.rand([padded_inputs_dec.size(0), 1],
                                                     device=padded_inputs_dec.device) >= seq_permute_rate) | teacher_enforcing_mask
                permuted_inputs_dec = torch.randint_like(padded_inputs_dec, 0, vocab_size_target)
                padded_inputs_dec = torch.where(teacher_enforcing_mask, padded_inputs_dec, permuted_inputs_dec)
                seq_length_enc = batch["seq_length_src"]
                seq_length_decoder = batch["seq_length_target"] - 1
                seq_length_decoder = torch.where(seq_length_decoder >= 0, seq_length_decoder,
                                                 torch.zeros_like(seq_length_decoder))

                # inspect_batch(tokenizer_src, tokenizer_target, padded_inputs_enc, padded_inputs_dec, padded_outputs_dec)
                optimizer.zero_grad()
                if not use_multi_device:
                    logits = wrapped_nn(padded_inputs_enc, seq_length_enc, padded_inputs_dec, seq_length_decoder)
                    loss = loss_fn(logits, padded_outputs_dec, seq_length_decoder)
                    loss = loss / seq_length_enc.sum()
                    loss.backward()
                else:
                    logits = wrapped_nn(padded_inputs_enc, seq_length_enc, padded_inputs_dec, seq_length_decoder)
                    loss = loss_fn(logits, padded_outputs_dec, seq_length_decoder)  # [n_devices]
                    loss = loss.sum() / seq_length_enc.sum()
                    loss.backward()
                writer.add_scalar("loss", loss, global_step=n_iter)
                writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step=n_iter)
                optimizer.step()
            lr_scheduler.step()

            with open(os.path.join(model_dir, "model-{}.pt".format(n_iter)), "wb") as f:
                torch.save(nn.state_dict(), f)

    if args.do_predict:
        last_checkpoint = get_last_checkpoint(model_dir)
        print("Loading checkpoint from {}".format(last_checkpoint))
        with open(last_checkpoint, "rb") as f:
            nn.load_state_dict(torch.load(f))
        test_set_cache = os.path.join(cache_dir, "test.pkl")
        if os.path.exists(test_set_cache):
            test_set = Dataset.load(test_set_cache)
        else:
            test_set = processor.get_test_data()
            test_set.save(test_set_cache)

        batch_size = 10
        bos_id = tokenizer_target.convert_tokens_to_ids(["[BOS]"])[0]
        eos_id = tokenizer_target.convert_tokens_to_ids(["[EOS]"])[0]
        n_beam = 10
        dec_hidden_dim = nn.decoder.rnn_cell.weight_hh.size(1)
        max_output_length = 100
        print("bos_id: {}, eos_id: {}".format(bos_id, eos_id))

        for batch in DataLoader(test_set,
                                shuffle=False,
                                batch_size=batch_size,
                                collate_fn=functools.partial(collate_fn, device=device)):
            tokens_src = batch["tokens_src"] if "tokens_src" in batch else [
                tokenizer_src.convert_ids_to_tokens(item.cpu().numpy()) for item in
                batch["token_ids_src"]]
            tokens_target = batch["tokens_target"] if "tokens_target" in batch else [
                tokenizer_target.convert_ids_to_tokens(item.cpu().numpy()) for item in
                batch["tokens_ids_target"]]
            inputs_enc = torch.nn.utils.rnn.pad_sequence(batch["token_ids_src"],
                                                         batch_first=True,
                                                         padding_value=0)
            seq_length_enc = batch["seq_length_src"]
            current_batch_size = seq_length_enc.size(0)
            dec_init_state = torch.zeros([current_batch_size, dec_hidden_dim], device=device)
            decoder_init_input = torch.empty([current_batch_size], dtype=torch.int64, device=device).fill_(bos_id)
            with torch.no_grad():
                outputs, output_lengths = nn.beam_search(inputs_enc,
                                                         seq_length_enc,
                                                         n_beam=n_beam,
                                                         eos_id=eos_id,
                                                         decoder_init_input=decoder_init_input,
                                                         decoder_init_state=dec_init_state,
                                                         max_length=max_output_length)
            outputs = outputs.to("cpu").numpy()
            output_lengths = output_lengths.to("cpu").numpy()
            for src, ref, pred, pred_length in zip(tokens_src, tokens_target, outputs, output_lengths):
                print("Source: {}\nSource (without sub-token):{}".format(src, join_sub_tokens(src)))
                print("Reference:{}\nReference (without sub-token):{}".format(ref, join_sub_tokens(ref)))
                for candidate_i, (pred, pred_length) in enumerate(zip(pred, pred_length)):
                    pred = tokenizer_target.convert_ids_to_tokens(pred.tolist())
                    if pred_length > 0:
                        pred = pred[:pred_length]
                    print("Candidate {}: {}\nCandidate {} (without sub-token): {}".format(candidate_i, pred, candidate_i, join_sub_tokens(pred)))
    if args.do_interactive_predict:
        last_checkpoint = get_last_checkpoint(model_dir)
        print("Loading checkpoint from {}".format(last_checkpoint))
        with open(last_checkpoint, "rb") as f:
            nn.load_state_dict(torch.load(f))
        batch_size = 10
        bos_id = tokenizer_target.convert_tokens_to_ids(["[BOS]"])[0]
        eos_id = tokenizer_target.convert_tokens_to_ids(["[EOS]"])[0]
        n_beam = 10
        dec_hidden_dim = nn.decoder.rnn_cell.weight_hh.size(1)
        max_output_length = 100
        print("bos_id: {}, eos_id: {}".format(bos_id, eos_id))
        while True:
            print("Source sentence:")
            text_src = sys.stdin.readline().rstrip()
            tokens_src = tokenizer_src.tokenize(text_src)
            print("Tokens (source): {}".format(tokens_src))
            inputs_enc = torch.tensor([tokenizer_src.convert_tokens_to_ids(tokens_src)], device=device)
            print(inputs_enc)
            seq_length_enc = torch.tensor([len(tokens_src)])
            current_batch_size = 1
            dec_init_state = torch.zeros([current_batch_size, dec_hidden_dim], device=device)
            decoder_init_input = torch.empty([current_batch_size], dtype=torch.int64, device=device).fill_(bos_id)
            with torch.no_grad():
                outputs, output_lengths = nn.beam_search(inputs_enc,
                                                         seq_length_enc,
                                                         n_beam=n_beam,
                                                         eos_id=eos_id,
                                                         decoder_init_input=decoder_init_input,
                                                         decoder_init_state=dec_init_state,
                                                         max_length=max_output_length)
            outputs = outputs.to("cpu").numpy()
            output_lengths = output_lengths.to("cpu").numpy()
            for src, pred, pred_length in zip(tokens_src, outputs, output_lengths):
                for candidate_i, (pred, pred_length) in enumerate(zip(pred, pred_length)):
                    pred = tokenizer_target.convert_ids_to_tokens(pred.tolist())
                    if pred_length > 0:
                        pred = pred[:pred_length]
                        pred_no_sub_token = join_sub_tokens(pred)
                        print("Candidate {}:\nTokens:{}\nTokens without subword:{}".format(candidate_i, pred, pred_no_sub_token))


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=os.path.normpath,
                        help="Directory for storing serialized dataset, model, evaluation results e.t.c..")
    parser.add_argument("--config", type=os.path.normpath,
                        help="Model specification file in JSON format.")
    parser.add_argument("--data-dir", type=os.path.normpath, help="Directory for dataset.")
    parser.add_argument("--do-train", action="store_true", help="Train the model.")
    parser.add_argument("--do-predict", action="store_true", help="Predict on test set.")
    parser.add_argument("--do-interactive-predict", action="store_true", help="Predict on user input.")
    parser.add_argument("--processor", type=str, help="Processor for dataset.")
    return parser


if __name__ == "__main__":
    args = _get_parser().parse_args()
    main(args)

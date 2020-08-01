import torch
from typing import Tuple
from .attention import ConcatAttention
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, embedding, rnn, flatten_parameters=False):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.flatten_parameters = flatten_parameters

    def forward(self, inputs: torch.Tensor, sequence_length: torch.Tensor):
        out = self.embedding(inputs)
        out = torch.nn.utils.rnn.pack_padded_sequence(out, sequence_length, batch_first=True, enforce_sorted=False)
        if self.flatten_parameters:
            self.rnn.flatten_parameters()
        out, last_state = self.rnn(out)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        return out, last_state


class Decoder(torch.nn.Module):
    def __init__(self, embedding, rnn_cell, context_aggr_type="pre-concat"):
        super().__init__()
        self.embedding = embedding
        self.rnn_cell = rnn_cell
        if context_aggr_type == "pre-concat":
            self.forward_fn = self._pre_concat_forward_fn
        elif context_aggr_type == "post-concat":
            self.forward_fn = self._post_concat_forward_fn
        else:
            raise ValueError("Unrecognized value of parameter ``: {}".format(context_aggr_type))

    def forward(self, inputs: torch.Tensor, state: torch.Tensor, context: torch.Tensor):
        """

        :param inputs: torch.Tensor of shape [batch_size]
        :param state: torch.Tensor of shape [batch_size, state_dim]
        :param context: torch.Tensor of shape [batch_size, context_dim]
        :return:
        """
        return self.forward_fn(inputs, state, context)

    def _pre_concat_forward_fn(self, inputs: torch.Tensor, state: torch.Tensor, context: torch.Tensor):
        """
        Aggregate context before rnn layer as defined in
        Neural Machine Translation by Jointly Learning to Align and Translate
        :param inputs: torch.Tensor of shape [batch_size]
        :param state: torch.Tensor of shape [batch_size, state_dim]
        :param context: torch.Tensor of shape [batch_size, context_dim]
        :return:
        """
        out = self.embedding(inputs)  # [batch_size, embedding_dim]
        out = torch.cat([out, context], dim=1)
        out = self.rnn_cell(out, state)
        return out

    def _post_concat_forward_fn(self, inputs: torch.Tensor, state: torch.Tensor, context: torch.Tensor):
        """
        Aggregate context after applying rnn layer as defined in
        Get To The Point: Summarization with Pointer-Generator Networks
        :param inputs: torch.Tensor of shape [batch_size]
        :param state: torch.Tensor of shape [batch_size, state_dim]
        :param context: torch.Tensor of shape [batch_size, context_dim]
        :return:
        """
        out = self.embedding(inputs)  # [batch_size, embedding_dim]
        out = self.rnn_cell(out, state)
        out = torch.cat([out, context], dim=1)
        return out


class Seq2SeqAttn(torch.nn.Module):
    def __init__(self, vocab_size_src, embedding_dim_src, vocab_size_target, embedding_dim_target,
                 enc_hidden_dim, dec_hidden_dim):
        """
        :param vocab_size_src: Size of vocabulary of original language.
        :param embedding_dim_src:
        :param vocab_size_target: Size of vocabulary of target language.
        :param embedding_dim_target:
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        """
        super().__init__()
        enc_embd = torch.nn.Embedding(vocab_size_src, embedding_dim_src, padding_idx=0)
        enc_rnn = torch.nn.LSTM(input_size=embedding_dim_src, hidden_size=enc_hidden_dim, batch_first=True,
                                bidirectional=True)
        self.encoder = Encoder(enc_embd, enc_rnn)

        dec_embd = torch.nn.Embedding(vocab_size_target, embedding_dim_target, padding_idx=0)
        dec_rnn_cell = torch.nn.GRUCell(input_size=2 * enc_hidden_dim + embedding_dim_target, hidden_size=dec_hidden_dim)
        self.decoder = Decoder(dec_embd, rnn_cell=dec_rnn_cell)

        self.attn = ConcatAttention(2 * enc_hidden_dim + dec_hidden_dim, hidden_dim=8)
        self.clf = torch.nn.Sequential(torch.nn.Linear(dec_hidden_dim, vocab_size_target),
                                       torch.nn.ReLU())
        # TODO: Initialize bias of clf layer to penalize words that are not included in the training set.
        self._dec_hidden_dim = dec_hidden_dim

    def forward(self,
                encoder_inputs: torch.Tensor,
                encoder_seq_length: torch.Tensor,
                decoder_inputs: torch.Tensor,
                decoder_seq_length: torch.Tensor) -> torch.Tensor:
        """
        Calculate output of decoder given input of both encoder & decoder.
        :param encoder_inputs: torch.Tensor of shape [batch_size, max_len_enc]
        :param encoder_seq_length: torch.Tensor of shape [batch_size]
        :param decoder_inputs: torch.Tensor of shape [batch_size, max_len_dec]
        :param decoder_seq_length: torch.Tensor of shape [batch_size]
        :return:
        """
        decoder_inputs = decoder_inputs.transpose(0, 1)  # [max_seq_len_dec, batch_size]
        batch_size = decoder_inputs.size(1)
        decoder_state = torch.zeros([batch_size, self._dec_hidden_dim], device=decoder_inputs.device)
        decoder_outputs = []
        encoder_outputs, last_state = self.encoder(encoder_inputs, encoder_seq_length)

        for decoder_cur_inputs in decoder_inputs:
            dec_out, decoder_state = self.decode_one_step_forward(encoder_outputs, encoder_seq_length,
                                                                  decoder_cur_inputs, decoder_state)
            decoder_outputs.append(dec_out)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return decoder_outputs

    def decode_one_step_forward(self,
                                encoder_outputs: torch.Tensor,
                                encoder_seq_length: torch.Tensor,
                                decoder_cur_inputs: torch.Tensor,
                                decoder_state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Calculate logits for next word in target language & new state vectors of decoder.
        :param encoder_outputs: torch.Tensor of shape [batch_size, max_seq_len, encoder_output_dim]
        :param encoder_seq_length: torch.Tensor of shape [batch_size]
        :param decoder_cur_inputs: torch.Tensor of shape [batch_size]
        :param decoder_state: torch.Tensor
        :return: (logits, state)
                - logits: torch.Tensor of shape [batch_size, vocab_size_e], logits for predicting next word.
                - state: torch.Tensor of shape [batch_size, decoder_hidden_dim], updated state of decoder
        """
        context, _ = self.attn(encoder_outputs=encoder_outputs, sequence_length=encoder_seq_length,
                               decoder_state=decoder_state)
        new_state = self.decoder(decoder_cur_inputs, decoder_state, context)
        out = self.clf(new_state)
        return out, new_state

    def beam_search(self,
                    encoder_inputs: torch.Tensor,
                    sequence_length: torch.Tensor,
                    n_beam: int,
                    eos_id: int,
                    decoder_init_input: torch.Tensor,
                    decoder_init_state: torch.Tensor,
                    max_length=10) -> Tuple[torch.Tensor]:
        """
        :param encoder_inputs: torch.Tensor of shape [batch_size, enc_max_length]
        :param sequence_length: torch.Tensor of shape [batch_size]. Lengths of sequence for encoder.
        :param n_beam: int
        :param eos_id: int
        :param decoder_init_input: torch.Tensor of shape [batch_size]
        :param decoder_init_state: torch.Tensor of shape [batch_size, decoder_state_dim]
        :param max_length: maximum length of sentence (without [BOS] & [EOS]).
        :return: (hypothesis_pool, hypothesis_length)
        hypothesis_pool: torch.Tensor of shape [batch_size, n_beam, max_length]
        hypothesis_length: torch.Tensor of shape [batch_size, n_beam]
        """
        assert max_length > 0

        back_pointers = []
        vocab_ids = []
        batch_size = decoder_init_input.size(0)
        dec_cur_input = decoder_init_input  # [batch_size]
        dec_state = decoder_init_state  # [batch_size, decoder_state_dim]
        enc_outputs, _ = self.encoder(encoder_inputs, sequence_length)  # [batch_size, encoder_max_seq_length, encoder_output_dim]
        # logits: [batch_size, vocab_size], dec_state: [batch_size, decoder_state_dim]
        logits, dec_state = self.decode_one_step_forward(enc_outputs, sequence_length, dec_cur_input, dec_state)
        vocab_size = logits.size(-1)
        assert 0 <= eos_id < vocab_size
        acc_log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
        # acc_log_probs: [batch_size, n_beam], vocab_ids_t: [batch_size, n_beam]
        acc_log_probs, vocab_ids_t = torch.topk(acc_log_probs, k=n_beam, dim=-1)
        if max_length == 1:
            return vocab_ids_t.unsqueeze(-1), torch.where(vocab_ids_t == eos_id, torch.ones_like(vocab_ids_t), torch.zeros_like(vocab_ids_t))
        # log
        vocab_ids_t = vocab_ids_t.flatten()  # [batch_size * n_beam]
        vocab_ids.append(vocab_ids_t)
        # expand batch: batch_size -> batch_size * n_beam
        sequence_length = sequence_length.unsqueeze(1).expand([-1, n_beam]).flatten()  # [batch_size * n_beam)]
        enc_outputs = enc_outputs.unsqueeze(1).expand([-1, n_beam, -1, -1])  # [batch_size, n_beam, encoder_max_seq_length, encoder_output_dim]
        enc_outputs = enc_outputs.flatten(start_dim=0, end_dim=1)  # [batch_size * n_beam, encoder_max_seq_length, encoder_output_dim]
        dec_state = dec_state.unsqueeze(1).expand([-1, n_beam, -1]).flatten(start_dim=0, end_dim=1)  # [batch_size * n_beam, decoder_state_dim]
        dec_cur_input = vocab_ids_t.flatten()  # [batch_size * n_beam]
        is_terminal = (dec_cur_input == eos_id)  # [batch_size * n_beam]
        hypothesis_length = torch.zeros([batch_size * n_beam], dtype=torch.int64, device=is_terminal.device)
        hypothesis_length.masked_fill_(is_terminal, 1)
        for i in torch.arange(1, max_length):
            if torch.all(is_terminal):
                break
            # logits: [batch_size * n_beam, vocab_size], dec_state: [batch_size * n_beam, decoder_dim]
            logits, dec_state = self.decode_one_step_forward(enc_outputs, sequence_length, dec_cur_input, dec_state)

            log_probs_t = F.log_softmax(logits, dim=-1)  # [batch_size * n_beam, vocab_size]
            # [batch_size, k_prev, 1] + [batch_size, n_beam, vocab_size]
            # calculate increment of acc_log_probs
            increment = log_probs_t.reshape([batch_size, -1, vocab_size])  # [batch_size, n_beam, vocab_size]
            increment_mask = is_terminal.reshape([batch_size, n_beam, 1]).expand_as(increment)
            increment = torch.where(increment_mask, torch.zeros_like(increment), increment)
            del increment_mask
            acc_log_probs = acc_log_probs.unsqueeze(-1) + increment  # [batch_size, n_beam, vocab_size]
            # get top n_beam transitions
            low_score = acc_log_probs.min() - 1
            # For hypothesis that has ended, (e.g. [..., "[EOS]", ...]),
            # we penalize all but one branches expanded from it before ranking.
            penalty_mask = is_terminal.reshape([batch_size, n_beam, 1]).repeat([1, 1, vocab_size])  # don't use expand here
            penalty_mask[:, :, 0] = False
            scores = acc_log_probs.masked_fill(penalty_mask, low_score)  # [batch_size, n_beam, vocab_size]
            del penalty_mask, low_score
            scores = scores.view([batch_size, -1])  # [batch_size, n_beam * vocab_size]
            _, flattened_indices = torch.topk(scores, n_beam, dim=-1)  # [batch_size, n_beam], [batch_size, n_beam]
            # Calculate quotient (prev_beam_ind) & remainder (current_vocab_id)
            # For this i_th sample in batch, j_th hypothesis in sample:
            # vocab_ids_t[i, j]: predicted word id
            # branch_ind[i, j]: index of branch this hypothesis is extended from
            branch_ind = torch.floor_divide(flattened_indices, vocab_size)  # [batch_size, n_beam]
            vocab_ids_t = flattened_indices % vocab_size  # [batch_size, n_beam]
            # get acc_log_probs
            acc_log_probs = acc_log_probs.reshape([batch_size, -1])  # [batch_size, n_beam * vocab_size], the very same organization as `scores`
            acc_log_probs = torch.gather(acc_log_probs, dim=-1, index=flattened_indices)

            # log result
            back_pointer = ((torch.arange(batch_size, device=branch_ind.device) * n_beam).unsqueeze(1) + branch_ind).flatten()  # [batch_size * n_beam]
            back_pointers.append(back_pointer)
            vocab_ids_t = vocab_ids_t.flatten()  # [batch_size * n_beam]
            vocab_ids.append(vocab_ids_t)

            # Update `hypothesis_length` and `is_terminal`
            # Index propagation (i.e. entry re-ordering of tensor according to newly generated hypotheses)
            hypothesis_length = hypothesis_length[back_pointer]
            is_terminal = is_terminal.reshape([batch_size, n_beam]).gather(dim=1, index=branch_ind).flatten()
            # Update `hypothesis_length`
            eos_flag = (vocab_ids_t == eos_id)
            length_update_flag = (eos_flag & (~is_terminal))  # Take into account `is_terminal` in case "[EOS]" appears twice (i.e. 0 -> "[EOS]")
            hypothesis_length.masked_fill_(length_update_flag, i + 1)
            # Update `is_terminal`
            is_terminal = is_terminal | eos_flag
            del eos_flag, length_update_flag

            # Select inputs & states to be fed to decoder at the next time stamp
            dec_state = dec_state[back_pointer, :]  # [batch_size * n_beam, decoder_state_dim]
            dec_cur_input = vocab_ids_t  # [batch_size * n_beam]
        vocab_ids = torch.stack(vocab_ids, dim=0)  # [decode_len, batch_size * n_beam]
        back_pointers = torch.stack(back_pointers, dim=0)  # [decode_len - 1, batch_size, n_beam]

        hypothesis_pool = torch.empty_like(vocab_ids)  # [decode_len, batch_size * n_beam]
        hypothesis_pool[-1] = vocab_ids[-1]
        for step in torch.arange(back_pointers.size(0)):
            back_pointer = back_pointers[-1 - step]
            if step > 0:
                back_pointer = back_pointer[prev_back_pointer]
            hypothesis_pool[-2 - step] = vocab_ids[-2 - step][back_pointer]
            prev_back_pointer = back_pointer
        hypothesis_pool.transpose_(0, 1)  # [batch_size * n_beam, decode_len]
        hypothesis_pool = hypothesis_pool.reshape([batch_size, n_beam, -1])
        hypothesis_length = hypothesis_length.reshape([batch_size, n_beam])
        return hypothesis_pool, hypothesis_length

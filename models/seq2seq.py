import torch
from typing import Tuple
from .attention import ConcatAttention
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, embedding, rnn):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn

    def forward(self, inputs: torch.Tensor, sequence_length: torch.Tensor):
        out = self.embedding(inputs)
        out = torch.nn.utils.rnn.pack_padded_sequence(out, sequence_length, batch_first=True, enforce_sorted=False)
        out, last_state = self.rnn(out)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        return out, last_state


class Decoder(torch.nn.Module):
    def __init__(self, embedding, rnn_cell):
        super().__init__()
        self.embedding = embedding
        self.rnn_cell = rnn_cell

    def forward(self, inputs: torch.Tensor, state: torch.Tensor, context: torch.Tensor):
        """

        :param inputs: torch.Tensor of shape [batch_size]
        :param state: torch.Tensor of shape [batch_size, state_dim]
        :param context: torch.Tensor of shape [batch_size, context_dim]
        :return:
        """
        out = self.embedding(inputs)  # [batch_size, embedding_dim]
        out = torch.cat([out, context], dim=1)
        out = self.rnn_cell(out)
        return out


class Seq2SeqAttn(torch.nn.Module):
    def __init__(self, vocab_size_f, embedding_dim_f, vocab_size_e, embedding_dim_e,
                 enc_hidden_dim, dec_hidden_dim):
        """
        Terminology: We assume that the model translates a foreign language (f) to English (e)
        :param vocab_size_f: Size of vocabulary of original language.
        :param embedding_dim_f:
        :param vocab_size_e: Size of vocabulary of target language.
        :param embedding_dim_e:
        :param enc_hidden_dim:
        :param dec_hidden_dim:
        :param attn:
        """
        super().__init__()
        enc_embd = torch.nn.Embedding(vocab_size_f, embedding_dim_f, padding_idx=0)
        enc_rnn = torch.nn.LSTM(input_size=embedding_dim_f, hidden_size=enc_hidden_dim, batch_first=True,
                                bidirectional=True)
        self.encoder = Encoder(enc_embd, enc_rnn)

        dec_embd = torch.nn.Embedding(vocab_size_e, embedding_dim_e, padding_idx=0)
        dec_rnn_cell = torch.nn.GRUCell(input_size=2 * enc_hidden_dim + embedding_dim_e, hidden_size=dec_hidden_dim)
        self.decoder = Decoder(dec_embd, rnn_cell=dec_rnn_cell)

        self.attn = ConcatAttention(2 * enc_hidden_dim + dec_hidden_dim, hidden_dim=8)
        self.clf = torch.nn.Sequential(torch.nn.Linear(dec_hidden_dim, vocab_size_e),
                                       torch.nn.ReLU())
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
            dec_out, decoder_state = self.decode_one_step_forward(encoder_outputs, encoder_seq_length, decoder_cur_inputs, decoder_state)
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
        context, _ = self.attn(encoder_outputs=encoder_outputs, sequence_length=encoder_seq_length, decoder_state=decoder_state)
        new_state = self.decoder(decoder_cur_inputs, decoder_state, context)
        out = self.clf(new_state)
        return out, new_state

    def beam_search(self, encoder_inputs: torch.Tensor, sequence_length, n_beam, decoder_init_input: torch.Tensor, decoder_init_state: torch.Tensor, max_length=10):
        """
        :param encoder_inputs: torch.Tensor of shape [batch_size, enc_max_length]
        :param sequence_length: torch.Tensor of shape [batch_size]. Lengths of sequence for encoder.
        :param n_beam: int
        :param init_input: torch.Tensor
        :param init_state: torch.Tensor
        :param max_length: maximum length of sentence (without [BOS] & [EOS]).
        :return:
        """
        back_pointers = []
        vocab_ids = []
        batch_size = decoder_init_input.size(0)
        k_prev = 1
        # k_prev: num of nodes in previous layer. In case of the first layer that contains
        # [BOS] only, k_prev == 1
        dec_cur_input = decoder_init_input  # [batch_size * k_prev(==1 for now)]
        dec_state = decoder_init_state  # [batch_size * k_prev(==1 for now), decoder_dim]
        enc_outputs, _ = self.encoder(encoder_inputs, sequence_length)  # [batch_size * k_prev(==1) for now,
        acc_log_probs = torch.zeros([batch_size, 1])  # [batch_size, k_prev]
        for i in torch.arange(max_length):
            # logits: [batch_size * prev_k, vocab_size], dec_state: [batch_size * prev_k, decoder_dim]
            logits, dec_state = self.decode_one_step_forward(enc_outputs, sequence_length, dec_cur_input, dec_state)
            vocab_size = logits.size(-1)
            log_probs_t = F.log_softmax(logits, dim=-1)  # [batch_size * k_prev, vocab_size]
            # [batch_size, k_prev, 1] + [batch_size, k_prev, vocab_size]
            acc_log_probs = acc_log_probs.unsqueeze(-1) + log_probs_t.view([batch_size, -1, vocab_size])  # [batch_size, k_prev, vocab_size]
            # get top n_beam transitions
            flattened_log_probs = acc_log_probs.view([batch_size, -1])  # [batch_size, k_prev * vocab_size]
            # TODO: mask invalid probs (e.g. extended from "[..., "[EOS]"])
            acc_log_probs, flattened_indices = torch.topk(flattened_log_probs, n_beam, dim=-1)  # [batch_size, n_beam], [batch_size, n_beam]
            # Calculate quotient (prev_beam_ind) & remainder (current_vocab_id)
            # For this i_th sample in batch, j_th hypothesis in sample:
            # vocab_ids_t[i, j]: predicted word id
            # branch_ind[i, j]: index of branch this hypothesis is extended from
            branch_ind = torch.div(flattened_indices, vocab_size)  # [batch_size, k_cur(==n_beam)]
            vocab_ids_t = flattened_indices - branch_ind * vocab_size  # [batch_size, k_cur(==n_beam)]
            # log result
            back_pointers.append(branch_ind)
            vocab_ids.append(vocab_ids_t)
            # Select inputs & states to be fed to decoder at the next time stamp
            indices = ((torch.arange(batch_size) * k_prev).unsqueeze(1) + branch_ind).flatten()  # [batch_size * k_cur(==n_beam)]
            dec_state = dec_state[indices, :]
            dec_cur_input = vocab_ids_t.flatten()  # [batch_size * k_cur, enc_max_length, ]

            if i == 0:
                # k_prev: 1 -> n_beam
                k_prev = n_beam
                sequence_length = sequence_length.unsqueeze(1).expand([-1, k_prev]).flatten()  # [batch_size * k_prev(==n_beam)]
                enc_outputs = enc_outputs.unsqueeze(1).expand([-1, k_prev, -1, -1]).reshape([-1, enc_outputs.size(-2), enc_outputs.size(-1)])  # [batch_size * k_prev(==n_beam), enc_max_length, enc_output_dim]

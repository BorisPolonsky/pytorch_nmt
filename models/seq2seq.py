import torch
from typing import Tuple
from .attention import ConcatAttention


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

    def forward(self):
        pass

    def decode_one_step_forward(self,
                                encoder_inputs: torch.Tensor,
                                encoder_seq_length: torch.Tensor,
                                decoder_cur_inputs: torch.Tensor,
                                decoder_state: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Calculate logits for next word in target language & new state vectors of decoder.
        :param encoder_inputs: torch.Tensor of shape [batch_size, max_seq_len]
        :param encoder_seq_length: torch.Tensor of shape [batch_size]
        :param decoder_cur_inputs: torch.Tensor of shape [batch_size]
        :param decoder_state: torch.Tensor
        :return: (logits, state)
                - logits: torch.Tensor of shape [batch_size, vocab_size_e], logits for predicting next word.
                - state: torch.Tensor of shape [batch_size, decoder_hidden_dim], updated state of decoder
        """
        out, last_state = self.encoder(encoder_inputs, encoder_seq_length)
        context, _ = self.attn(encoder_outputs=out, sequence_length=encoder_seq_length, decoder_state=decoder_state)
        new_state = self.decoder(decoder_cur_inputs, decoder_state, context)
        out = self.clf(new_state)
        return out, new_state


import torch
from typing import Tuple


def sequence_mask(lengths: torch.Tensor, maxlen) -> torch.Tensor:
    """
    Something like `tensorflow.sequence_mask`.
    :param lengths: `torch.Tensor`
    :param maxlen: `int`
    :return: `torch.Tensor` of dtype `torch.bool`
    """
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = lengths.unsqueeze(dim=-1)
    mask = row_vector < matrix
    return mask


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


class Attention(torch.nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.fc_w = torch.nn.Linear(in_features, hidden_dim, bias=False)
        self.fc_v = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.mask_val = -50000

    def forward(self, encoder_outputs: torch.Tensor, sequence_length: torch.Tensor, decoder_state: torch.Tensor):
        """
        s(x_i, x_j) = v^{T} tanh{W_1x_i+W_2x_j}
        v \in \mathbb{R}^{hidden-dim}
        :param encoder_outputs: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim]
        :param sequence_length: torch.Tensor of shape [batch_size]
        :param decoder_state: torch.Tensor of shape [batch_size, dec_state_dim]
        :return: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim]
        """
        out = torch.cat([encoder_outputs, decoder_state.unsqueeze(1).expand([-1, encoder_outputs.size(1), -1])], dim=-1)
        attn_logits = self.fc_w(out)  # [batch_size, enc_seq_len, hidden_dim]
        attn_logits = attn_logits.tanh()
        attn_logits = self.fc_v(attn_logits).squeeze(2)  # [batch_size, enc_seqlen]
        attn_weight = self.masked_softmax(attn_logits, sequence_length).unsqueeze(1)  # [batch_size, 1, enc_seq_len]
        out = torch.bmm(attn_weight, encoder_outputs).squeeze(1)
        return out

    def masked_softmax(self, logits: torch.Tensor, sequence_length: torch.Tensor) -> torch.Tensor:
        """
        :param logits: torch.Tensor of shape [batch_size, seq_len]
        :param sequence_length: torch.Tensor of shape [batch_size]
        :return: torch.Tensor of shape [batch_size, seq_len]
        """
        mask = sequence_mask(sequence_length, maxlen=logits.size(1)).to(logits.device)
        masked_logits = torch.where(mask, logits, torch.empty_like(logits).fill_(self.mask_val))
        out = torch.softmax(masked_logits, dim=1)
        return out


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

        self.attn = Attention(2 * enc_hidden_dim + dec_hidden_dim, hidden_dim=8)
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
        context = self.attn(encoder_outputs=out, sequence_length=encoder_seq_length, decoder_state=decoder_state)
        new_state = self.decoder(decoder_cur_inputs, decoder_state, context)
        out = self.clf(new_state)
        return out, new_state


if __name__ == "__main__":
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    enc_inputs = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0]], device=device)
    sequence_length = torch.tensor([4, 3])
    enc_hidden_dim = 100
    dec_hidden_dim = 200
    dec_cur_input = torch.tensor([1, 2], device=device)
    nn = Seq2SeqAttn(vocab_size_f=100, embedding_dim_f=64, vocab_size_e=100, embedding_dim_e=64,
                     enc_hidden_dim=enc_hidden_dim,
                     dec_hidden_dim=dec_hidden_dim).to(device)
    dec_state = torch.zeros([dec_cur_input.size(0), dec_hidden_dim], device=dec_cur_input.device)
    for i in range(5):
        out, dec_state = nn.decode_one_step_forward(enc_inputs, sequence_length, dec_cur_input, dec_state)

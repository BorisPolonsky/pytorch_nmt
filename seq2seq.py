import torch

def sequence_mask(lengths: torch.Tensor, maxlen):
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

    def forward(self, encoder_states: torch.Tensor, sequence_length: torch.Tensor, decoder_state: torch.Tensor):
        """
        s(x_i, x_j) = v^{T} tanh{W_1x_i+W_2x_j}
        v \in \mathbb{R}^{hidden-dim}
        :param encoder_states: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim]
        :param sequence_length: torch.Tensor of shape [batch_size]
        :param decoder_state: torch.Tensor of shape [batch_size, dec_state_dim]
        :return: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim]
        """
        out = torch.cat([encoder_states, decoder_state.unsqueeze(1).expand([-1, encoder_states.size(1), -1])], dim=-1)
        attn_logits = self.fc_w(out)  # [batch_size, enc_seq_len, hidden_dim]
        attn_logits = attn_logits.tanh()
        attn_logits = self.fc_v(attn_logits).squeeze(2) # [batch_size, enc_seqlen]
        attn_weight = self.masked_softmax(attn_logits, sequence_length).unsqueeze(1)  # [batch_size, 1, enc_seq_len]
        out = torch.bmm(attn_weight, encoder_states).squeeze(1)
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
    def __init__(self, embedding, rnn, attn_fc_dim):
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn

    def forward(self, enc_inputs, sequence_length, current_input):
        out = self.embedding(inputs)
        out = torch.nn.utils.rnn.pack_padded_sequence(out, sequence_length, batch_first=True, enforce_sorted=False)
        out, last_state = self.rnn(out)


class Seq2SeqAttn(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, encoder_inputs, decoder_inputs):
        pass

embd_dim = 64
enc_hidden_dim = dec_hidden_dim = 50

enc_embd = dec_embd = torch.nn.Embedding(100, embd_dim)
enc_inputs = torch.tensor([[1,2,3,4],[1,2,3,0]])
sequence_length = torch.tensor([4,3])
enc_rnn = torch.nn.LSTM(input_size=embd_dim, hidden_size=enc_hidden_dim, batch_first=True, bidirectional=True)
dec_rnn = torch.nn.LSTM(input_size=enc_hidden_dim, hidden_size=dec_hidden_dim, batch_first=True)

encoder = Encoder(enc_embd, enc_rnn)
out, last_state = encoder(enc_inputs, sequence_length)
attn = Attention(2 * enc_hidden_dim + dec_hidden_dim, hidden_dim=8)
decoder_state = torch.ones([2, dec_hidden_dim])
attn(encoder_states=out, sequence_length=sequence_length, decoder_state=decoder_state)

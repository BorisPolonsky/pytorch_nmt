import torch

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
    def __init__(self, in_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, 1)

    def forward(self, encoder_states: torch.Tensor, sequence_length: torch.Tensor, decoder_state: torch.Tensor):
        print(encoder_states.shape)
        out = torch.cat([encoder_states, decoder_state.unsqueeze(1).expand([-1, encoder_states.size(1), -1])], dim=-1)
        attn_weight = self.fc(out)
        # self.masked_softmax
    def masked_softmax(self, logits, sequence_length):
        pass

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
attn = Attention(2 * enc_hidden_dim + dec_hidden_dim)
decoder_state = torch.ones([2, dec_hidden_dim])
attn(encoder_states=out, sequence_length=sequence_length, decoder_state=decoder_state)

import torch
from models.seq2seq import Seq2SeqAttn

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

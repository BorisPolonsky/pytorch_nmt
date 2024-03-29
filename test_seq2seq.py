import torch
from models.seq2seq import Seq2SeqAttn
from functools import partial

if __name__ == "__main__":
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    enc_inputs = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0]], device=device)
    sequence_length = torch.tensor([4, 3])
    enc_hidden_dim = 100
    dec_hidden_dim = 200
    dec_cur_input = torch.tensor([1, 2], device=device)
    nn = Seq2SeqAttn(vocab_size_src=100, embedding_dim_src=64, vocab_size_target=100, embedding_dim_target=64,
                     enc_hidden_dim=enc_hidden_dim,
                     dec_hidden_dim=dec_hidden_dim).to(device)
    init_dec_state = dec_state = torch.zeros([dec_cur_input.size(0), dec_hidden_dim], device=dec_cur_input.device)
    enc_outputs, _ = nn.encoder(enc_inputs, sequence_length)
    decoder_fn = partial(nn.decode_one_step_forward, enc_outputs, sequence_length)
    for i in range(5):
        out, dec_state = decoder_fn(dec_cur_input, dec_state)
    init_dec_input = torch.tensor([2, 2])
    nn.beam_search(enc_inputs, sequence_length, 10, 2, init_dec_input, init_dec_state, max_length=20)

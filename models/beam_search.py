import torch
import torch.nn.functional as F


def top_k(decoder, k, init_input, init_state, max_depth=10):
    hypotheses = []
    dec_state = init_state
    # not done yet
    # decoder_state [batch_size, k_prev, decoder_dim]
    dec_cur_input = init_input
    batch_size = init_input.size(0)

    for i in range(max_depth):
        logits, dec_state = decoder(dec_cur_input, dec_state)
        vocab_size = logits.size(-1)
        probs = F.softmax(logits, dim=-1)  # [batch_size * k_prev, vocab_size]

        # TODO: probs reshape: [batch_size, k * vocab_size]
        probs = probs.reshape([batch_size, -1])  # [batch_size, k_prev * vocab_size]

        # flatten_indices: [batch_size, k]
        _, flattened_indices = torch.topk(probs, k=k, dim=-1)
        # calculate div & mod
        branch_id = torch.div(flattened_indices, vocab_size)  # [batch_size, k]
        vocab_ind = flattened_indices - (branch_id * vocab_size)  # [batch_size, k]
        # select corresponding decoder state
        print(dec_state.shape)
        dec_state = dec_state[branch_id]  # ???? To be checked
        print(dec_state.shape)
        exit(0)
        # [batch_size, k, decoder_dim]
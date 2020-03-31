import torch
import torch.nn.functional as F


def beam_search(decoder, n_beam, init_input: torch.Tensor, init_state: torch.Tensor, max_length=10):
    """

    :param decoder: callable that accepts (inputs, init_state) and returns (outputs, new_state) tuple.
    :param n_beam: int
    :param init_input: torch.Tensor
    :param init_state: torch.Tensor
    :param max_length: maximum length of sentence (without [BOS] & [EOS]).
    :return:
    """
    back_pointers = []
    vocab_ids = []

    batch_size = init_input.size(0)
    # k_prev: num of spanned nodes in previous layer. In case of the first layer that contains
    # [BOS] only, k_prev == 1
    dec_cur_input = init_input  # [batch_size * k_prev(==1)]
    dec_state = init_state  # [batch_size * k_prev(==1), decoder_dim]
    acc_log_probs = torch.zeros([batch_size, 1])  # [batch_size, k_prev(==1)]
    for i in range(1, max_length + 1):
        logits, dec_state = decoder(dec_cur_input, dec_state)  # logits: [batch_size * prev_k, vocab_size], dec_state: [batch_size * prev_k, decoder_dim]
        vocab_size = logits.size(-1)
        log_probs_t = F.log_softmax(logits, dim=-1)  # [batch_size * k_prev, vocab_size]
        # [batch_size, k_prev, 1] + [batch_size, k_prev, vocab_size]
        acc_log_probs = acc_log_probs.unsqueeze(-1) + log_probs_t.view([batch_size, -1, vocab_size])  # [batch_size, k_prev, vocab_size]
        k_prev = acc_log_probs.size(1)  # num of hypothesis per sample (l.e. num of kept nodes in previous layer of search tree)
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
        print(init_input.size(), init_state.size())
        indices = ((torch.arange(batch_size) * k_prev).unsqueeze(1) + branch_ind).flatten()  # [batch_size * k_cur(==n_beam)]
        dec_state = dec_state[indices, :]
        print(dec_state.size())
        dec_cur_input = vocab_ids_t.flatten()  # [batch_size * k_cur]
        print(dec_cur_input.size())
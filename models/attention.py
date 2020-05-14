import torch
from .util import sequence_mask
from typing import Tuple


def masked_softmax(logits: torch.Tensor, sequence_length: torch.Tensor, mask_val=-50000) -> torch.Tensor:
    """
    :param logits: torch.Tensor of shape [batch_size, seq_len]
    :param sequence_length: torch.Tensor of shape [batch_size]
    :param mask_val: an negative integer for masking logits before applying softmax function.
    :return: torch.Tensor of shape [batch_size, seq_len]
    """
    mask = sequence_mask(sequence_length, maxlen=logits.size(1)).to(logits.device)
    masked_logits = torch.where(mask, logits, torch.empty_like(logits).fill_(mask_val))
    out = torch.softmax(masked_logits, dim=1)
    return out


class ConcatAttention(torch.nn.Module):
    r"""
    Calculate attention score as:
    score(h_i, s_j) = v^{T} tanh(W[h_i;s_j])
    W \in \mathbb{R}^{hidden-dim \times in-features}
    v \in \mathbb{R}^{hidden-dim}
    Or equivalently:
    score(h_i, s_j)  = v^{T} tanh(W_1h_i+W_2s_j)
    """

    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.fc_w = torch.nn.Linear(in_features, hidden_dim, bias=False)
        self.fc_v = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.mask_val = -50000

    def forward(self, encoder_outputs: torch.Tensor, sequence_length: torch.Tensor, decoder_state: torch.Tensor) -> Tuple[torch.Tensor]:
        """

        :param encoder_outputs: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim]
        :param sequence_length: torch.Tensor of shape [batch_size]
        :param decoder_state: torch.Tensor of shape [batch_size, dec_state_dim]
        :return: tuple of (out, attn_weight)
            - out: torch.Tensor of shape [batch_size, enc_seq_len, enc_state_dim], output with attention applied
            - attn_weight: torch.Tensor of shape [batch_size, enc_sec_len], calculated attention weights.
        """
        out = torch.cat([encoder_outputs, decoder_state.unsqueeze(1).expand([-1, encoder_outputs.size(1), -1])], dim=-1)
        attn_logits = self.fc_w(out)  # [batch_size, enc_seq_len, hidden_dim]
        attn_logits = attn_logits.tanh()
        attn_logits = self.fc_v(attn_logits).squeeze(2)  # [batch_size, enc_seqlen]
        attn_weight = masked_softmax(attn_logits, sequence_length, mask_val=self.mask_val)  # [batch_size, enc_seq_len]
        out = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs).squeeze(1)
        return out, attn_weight

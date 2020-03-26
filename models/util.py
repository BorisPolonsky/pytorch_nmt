import torch


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

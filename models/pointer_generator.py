import torch


class PointerGeneratorDistribution(torch.nn.Module):
    def __init__(self, vocab_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not vocab_size > 0:
            raise ValueError('"vocab_size" must be a positive integer, got {}'.format(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, encoder_inputs, attention_weight):
        """
        Return the probability distribution based on Pointer-Generator mechanism.
        :param encoder_inputs: torch.Tensor of size [batch_size, max_sequence_length]
        :param attention_weight: torch.Tensor of size [batch_size, max_sequence_length] s.t. sum of each row equals to 1
        :return: out: torch.Tensor of size [batch_size, vocab_size] probability distribution.
        """
        batch_size = encoder_inputs.size(0)
        out = torch.zeros(batch_size, self.vocab_size, device=attention_weight.device)
        out = torch.scatter_add(out, 1, encoder_inputs, attention_weight)
        return out

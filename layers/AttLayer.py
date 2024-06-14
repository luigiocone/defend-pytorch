import torch
from torch import nn


class AttLayer(nn.Module):
    def __init__(self, input_dim, attention_dim=100):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Parameter(torch.randn(input_dim, attention_dim))
        self.b = nn.Parameter(torch.randn(attention_dim))
        self.u = nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, x: torch.Tensor, mask=None):
        # x.size() == torch.Size([sentence_count, token_count, hidden_dim])
        assert x.ndim == 3

        # uit = tanh(xW+b)
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)

        # ait = uit * u
        ait = torch.matmul(uit, self.u)
        ait = ait.squeeze(-1)
        ait = torch.exp(ait)
        if mask is not None:
            ait = ait * mask

        ait = ait / (torch.sum(ait, dim=1, keepdim=True) + 1e-8)
        ait = ait.unsqueeze(-1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=1)

        # output.size() == torch.Size([sentence_count, hidden_dim])
        return output

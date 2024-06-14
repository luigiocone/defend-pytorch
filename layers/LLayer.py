import torch
from torch import nn, tanh
from torch.nn.functional import softmax


class LLayer(nn.Module):
    def __init__(self, latent_dim, k=80):
        super(LLayer, self).__init__()
        self.Wl = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.Wc = nn.Parameter(torch.randn(k, latent_dim))
        self.Ws = nn.Parameter(torch.randn(k, latent_dim))
        self.whs = nn.Parameter(torch.randn(1, k))
        self.whc = nn.Parameter(torch.randn(1, k))

    def forward(self, content, comments):
        # content, comments == torch.Size([batch_size, sentence_count, latent_dim])

        # Transpose
        content_T = torch.permute(input=content, dims=(0, 2, 1))
        comments_T = torch.permute(input=comments, dims=(0, 2, 1))

        L = tanh(torch.einsum('btd,dD,bDn->btn', comments, self.Wl, content_T))
        L_T = torch.permute(L, dims=(0, 2, 1))

        Hs = tanh(
            torch.einsum('kd,bdn->bkn', self.Ws, content_T)
            + torch.einsum('kd,bdt,btn->bkn', self.Wc, comments_T, L)
        )
        Hc = tanh(
            torch.einsum('kd,bdt->bkt', self.Wc, comments_T)
            + torch.einsum('kd,bdn,bnt->bkt', self.Ws, content_T, L_T)
        )

        As = softmax(torch.einsum('yk,bkn->bn', self.whs, Hs), dim=1)
        Ac = softmax(torch.einsum('yk,bkt->bt', self.whc, Hc), dim=1)
        co_s = torch.einsum('bdn,bn->bd', content_T, As)
        co_c = torch.einsum('bdt,bt->bd', comments_T, Ac)
        co_sc = torch.cat([co_s, co_c], dim=1)

        # co_sc.size() == torch.Size([batch_size, latent_dim * 2])
        return co_sc

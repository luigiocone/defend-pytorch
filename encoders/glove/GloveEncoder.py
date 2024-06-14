import torch
from torch import nn

from encoders.glove.Glove import Glove
from layers.AttLayer import AttLayer


class GloveEncoder(nn.Module):
    def __init__(self, glove: Glove, gru: nn.GRU, attention: AttLayer, trainable=True):
        super(GloveEncoder, self).__init__()
        self.glove = glove
        self.gru = gru
        self.attention = attention

        embedding_matrix = torch.from_numpy(glove.embedding_matrix).float()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, padding_idx=glove.pad_idx)
        self.embedding = self.embedding.requires_grad_(trainable)

    def forward(self, sentences: {}):
        inp = sentences['input_ids']             # inp.size() == torch.Size([token_count, sentence_count])
        inp = inp.permute(1, 0)                  # inp.size() == torch.Size([sentence_count, token_count])
        emb = self.embedding(inp)                # emb.size() = torch.Size([sentence_count, token_count, embed_dim])
        lengths = sentences['sentence_lengths']  # lengths.size() == torch.Size([sentence_count])

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.gru(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(packed, padding_value=self.glove.pad_idx, batch_first=True)
        # padded.size() == torch.Size([sentence_count, token_count, bigru_hidden_dim])

        mask = self.pad_mask(padded)      # mask.size() == torch.Size([sentence_count, token_count, bigru_hidden_dim])
        padded = padded * mask            # padded.size() == torch.Size([sentence_count, token_count, bigru_hidden_dim])
        mask = mask.any(dim=-1).float()   # mask.size() == torch.Size([sentence_count, token_count])

        sentence_representation = self.attention(padded, mask=mask)
        return sentence_representation   # sentence_representation.size() == torch.Size([sentence_count, 200])

    def pad_mask(self, padded: torch.Tensor) -> torch.Tensor:
        # tolerance = 1e-8
        # mask = torch.abs(padded - self.glove.pad_idx) > tolerance
        mask = (padded != self.glove.pad_idx).float()
        return mask
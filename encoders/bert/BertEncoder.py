import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from layers.AttLayer import AttLayer


class BertEncoder(nn.Module):
    def __init__(self, checkpoint: str, gru: nn.GRU, attention: AttLayer, trainable=True):
        super(BertEncoder, self).__init__()
        config = AutoConfig.from_pretrained(checkpoint)
        config.gradient_checkpointing = True  # less memory issues, more computation time
        self.bert = AutoModel.from_pretrained(checkpoint, config=config)
        self.bert = self.bert.requires_grad_(trainable)
        self.gru = gru
        self.attention = attention

    def forward(self, sentences: {}):
        inp = sentences['input_ids']       # inp.size() == torch.Size([sentence_count, token_count])
        att = sentences['attention_mask']  # att.size() == torch.Size([sentence_count, token_count])

        embedding = self.bert(input_ids=inp, attention_mask=att)
        embedding = embedding.last_hidden_state    # embedding.size() == torch.Size([sentence_count, token_count, 768])
        lengths = torch.sum(att, dim=-1)           # lengths.size() == torch.Size([sentence_count])

        # Ignore [PAD] tokens in gru
        packed = nn.utils.rnn.pack_padded_sequence(embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.gru(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        # padded.size() == torch.Size([sentence_count, token_count, 200])

        # Mask [PAD] tokens in AttLayer
        mask = sentences['attention_mask']
        sentence_representation = self.attention(padded, mask=mask)

        # sentence_representation.size() == torch.Size([sentence_count, 200])
        return sentence_representation

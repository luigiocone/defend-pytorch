import torch
from torch import nn
from encoders.bert.BertEncoder import BertEncoder
from encoders.glove.GloveEncoder import GloveEncoder
from layers.AttLayer import AttLayer
from layers.LLayer import LLayer


class Defend(nn.Module):
    def __init__(self, config, gru_hidden_dim=100):
        super(Defend, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim

        self.content_encoder = self.build_encoder(config)
        self.comment_encoder = self.build_encoder(config)

        self.content_sentences_gru = nn.GRU(
            input_size=gru_hidden_dim*2, hidden_size=gru_hidden_dim, bidirectional=True, batch_first=True
        )

        self.llayer = LLayer(latent_dim=gru_hidden_dim*2)
        self.linear = nn.Linear(in_features=gru_hidden_dim*4, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def build_encoder(self, config):
        attention = AttLayer(input_dim=self.gru_hidden_dim*2)
        enc_trainable = config['encoder_trainable']

        if config['encoder'] == 'glove':
            glove = config['glove']
            embed_dim = glove.embed_dim
            words_gru = nn.GRU(input_size=embed_dim, hidden_size=self.gru_hidden_dim, bidirectional=True, batch_first=True)
            encoder = GloveEncoder(glove, words_gru, attention, enc_trainable)
            return encoder
        # else
        bert_preset = config['bert_preset']
        bert_hidden_dim = config['bert_hidden_dim']
        words_gru = nn.GRU(input_size=bert_hidden_dim, hidden_size=self.gru_hidden_dim, bidirectional=True, batch_first=True)
        encoder = BertEncoder(checkpoint=bert_preset, trainable=enc_trainable, gru=words_gru, attention=attention)
        return encoder

    def forward(self, content_batch, comment_batch):
        content_encoded = []
        for content in content_batch:
            enc = self.content_encoder(content)
            enc, _ = self.content_sentences_gru(enc)
            content_encoded.append(enc)                # enc.size() == torch.Size([sentence_count, 200])

        comment_encoded = []
        for comment in comment_batch:
            enc = self.comment_encoder(comment)
            comment_encoded.append(enc)                # enc.size() == torch.Size([sentence_count, 200])

        tensors = []
        for i in range(len(content_encoded)):
            ts = self.llayer(
                content_encoded[i].unsqueeze(0),       # torch.Size([1, sentence_count, 200])
                comment_encoded[i].unsqueeze(0),       # torch.Size([1, sentence_count, 200])
            ).squeeze(0)
            tensors.append(ts)                         # ts.size() == torch.Size([400])
        output = torch.stack(tensors)                  # output.size() == torch.Size([batch_size, 400])

        output = self.linear(output)                   # output.size() == torch.Size([batch_size, 2])
        output = self.softmax(output)
        return output

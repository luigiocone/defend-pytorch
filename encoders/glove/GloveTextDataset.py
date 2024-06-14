import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import encoders.utils as utils


def tokenize(texts, glove):
    ids = []
    for sentence in texts:
        input_ids = [glove.word2idx(tok) for tok in sentence]
        ids.append(input_ids)
    return ids


class GloveTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config):
        self.df = df
        self.glove = config['glove']
        self.max_sentence_count = config['max_content_sentences']
        self.max_comments = config['max_comments']
        self._preprocess()

    def _preprocess(self):
        self.df['content'] = self.df['content'].apply(utils.clean, args=(self.max_sentence_count,))
        self.df['content'] = self.df['content'].apply(utils.word_tokenize)
        self.df['content'] = self.df['content'].apply(tokenize, args=(self.glove,))

        self.df['comment'] = self.df['comment'].apply(utils.clean, args=(self.max_comments,))
        self.df['comment'] = self.df['comment'].apply(utils.word_tokenize)
        self.df['comment'] = self.df['comment'].apply(tokenize, args=(self.glove,))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, sample_id):
        row = self.df.iloc[sample_id]

        tensors = [torch.LongTensor(inp) for inp in row['content']]
        lengths = [t.size(0) for t in tensors]
        content_sample = {
            'input_ids': torch.nn.utils.rnn.pad_sequence(tensors, padding_value=self.glove.pad_idx),
            'sentence_lengths': torch.tensor(lengths),
            'sentence_count': torch.tensor(len(lengths)),
            'labels': torch.tensor(row['labels']),
        }

        tensors = [torch.LongTensor(inp) for inp in row['comment']]
        lengths = [t.size(0) for t in tensors]
        comment_sample = {
            'input_ids': torch.nn.utils.rnn.pad_sequence(tensors, padding_value=self.glove.pad_idx),
            'sentence_lengths': torch.tensor(lengths),
            'sentence_count': torch.tensor(len(lengths)),
            'labels': torch.tensor(row['labels']),
        }

        return {
            'content': content_sample,
            'comment': comment_sample,
        }

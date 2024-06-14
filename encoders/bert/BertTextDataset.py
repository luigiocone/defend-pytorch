import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from torch.utils.data.dataset import Dataset
import encoders.utils as utils


def tokenize(row, tokenizer, column):
    tokenized = tokenizer(row[column], padding=True, truncation=True)
    row[column + '_input_ids'] = tokenized['input_ids']
    row[column + '_attention_mask'] = tokenized['attention_mask']
    return row

class BertTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config, tokenizer):
        self.df = df
        self.pad_id = config['bert_pad_token_id']
        self.max_token_count = config['bert_token_count']
        self.max_sentence_count = config['max_content_sentences']
        self.max_comments = config['max_comments']
        self.tokenizer = tokenizer
        self._preprocess()

    def _preprocess(self):
        self.df['content'] = self.df['content'].apply(utils.clean, args=(self.max_sentence_count,))
        self.df['content'] = self.df['content'].apply(utils.split_long_sequences, args=(self.max_token_count - 64, self.max_sentence_count))
        self.df = self.df.apply(tokenize, axis=1, args=(self.tokenizer, 'content'))

        self.df['comment'] = self.df['comment'].apply(utils.clean, args=(self.max_comments,))
        self.df['comment'] = self.df['comment'].apply(utils.split_long_sequences, args=(self.max_token_count - 64, self.max_comments))
        self.df = self.df.apply(tokenize, axis=1, args=(self.tokenizer, 'comment'))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, sample_id):
        row = self.df.iloc[sample_id]

        ctn_ids = [torch.LongTensor(inp) for inp in row['content_input_ids']]
        ctn_ids = torch.vstack(ctn_ids)
        ctn_mask = [torch.LongTensor(att) for att in row['content_attention_mask']]
        ctn_mask = torch.vstack(ctn_mask)

        content_sample = {
            'input_ids': ctn_ids,
            'attention_mask': ctn_mask,
            'labels': torch.tensor(row['labels'])
        }
        comment_sample = {
            'input_ids': ctn_ids,
            'attention_mask': ctn_mask,
            'labels': torch.tensor(row['labels'])
        }

        return {
            'content': content_sample,
            'comment': comment_sample,
        }

import numpy as np
from tqdm import tqdm


class Glove:
    def __init__(self, path):
        self.path = path
        self.embed_dim = 100
        self._load()

        # Add special token
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.try_add_word(word=self.pad_token, weights=np.zeros(self.embed_dim))
        self.try_add_word(word=self.unk_token, weights=np.mean(self.embeddings, axis=0, keepdims=True))
        self.pad_idx = self.word2idx(self.pad_token)
        #print(self.embedding_matrix.shape)

    def _load(self):
        vocab, embeddings = [], []
        word2idx = {}
        idx = 0

        with open(self.path, 'rt', encoding='utf-8') as glove:
            for line in glove.readlines():
                columns = line.split(' ')

                word = columns[0]
                vocab.append(word)

                word2idx[word] = idx
                idx += 1

                emb = np.array(columns[1:], dtype="float32")
                embeddings.append(emb)

        self.vocab = vocab
        self.embeddings = np.vstack(embeddings)
        self._word2idx = word2idx
        self.idx2word = {v: k for k, v in self._word2idx.items()}
        self.glove = {w: self.embeddings[self._word2idx[w]] for w in self.vocab}

        self.embedding_matrix = np.zeros((len(vocab), self.embed_dim))
        for i, word in enumerate(vocab):
            self.embedding_matrix[i] = self.glove[word]

    def try_add_word(self, word: str, weights: np.array) -> bool:
        if word in self.vocab:
            return False
        idx = len(self.vocab)
        self.vocab.append(word)
        self.embeddings = np.vstack([self.embeddings, weights])
        self._word2idx[word] = idx
        self.idx2word[idx] = word
        self.glove[word] = weights
        self.embedding_matrix = np.vstack([self.embedding_matrix, weights])
        return True

    def fit_on_text(self, words: {str: int}):
        # Remove uncommon words
        min_word_occurrences = 5
        words = {k: v for k, v in words.items() if v >= min_word_occurrences}

        weights = np.random.normal(scale=0.6, size=(self.embed_dim,))
        progress_bar = tqdm(range(len(words)), desc="glove fitting")
        for word in words:
            added = self.try_add_word(word, weights)
            progress_bar.update(n=1)
            if added:
                weights = np.random.normal(scale=0.6, size=(self.embed_dim,))

    def word2idx(self, word):
        if word in self._word2idx.keys():
            return self._word2idx[word]
        return self._word2idx[self.unk_token]

    def word2embeddings(self, word):
        if word in self.glove.keys():
            return self.glove[word]
        return self.glove[self.unk_token]
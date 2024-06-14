import nltk
from bs4 import BeautifulSoup


def word_tokenize(texts):
    txt = []
    for sentence in texts:
        tokens = nltk.tokenize.word_tokenize(sentence)
        if len(tokens) == 0:
            continue
        txt.append(tokens)
    return txt


def clean(texts, max_sentence_count):
    # Cut num of sentences if necessary
    if 0 < max_sentence_count < len(texts):
        # indexes = range(len(texts))
        # indexes = np.random.choice(indexes, size=max_sentence_count, replace=False)
        # texts = [texts[i] for i in indexes]
        start = len(texts) - max_sentence_count
        texts = texts[start:]

    new_texts = []
    for sentence in texts:
        # Some texts are represented in html
        sentence = BeautifulSoup(sentence, features="html5lib").get_text()
        sentence = sentence.encode('ascii', 'ignore')
        sentence = sentence.decode("ascii")
        sentence = sentence.replace("\\", '')
        sentence = sentence.replace("'", '')
        sentence = sentence.replace('"', '')
        sentence = sentence.strip().lower()
        new_texts.append(sentence)
    return new_texts


def split_long_sequences(texts, max_len, max_sentence_count):
    # Split long sentences in a list of sentences
    new_texts = []
    for sentence in texts:
        words = sentence.split()
        short_sentences = []
        while len(words) > max_len:
            s = " ".join(words[:max_len])
            short_sentences.append(s)
            words = words[max_len:]
        s = " ".join(words)
        short_sentences.append(s)

        new_texts = new_texts + short_sentences

    if max_sentence_count <= 0 or len(new_texts) < max_sentence_count:
        return new_texts

    # Select random sentences
    # indexes = range(len(new_texts))
    # new_indexes = np.random.choice(indexes, size=max_sentence_count, replace=False)
    # output = [new_texts[i] for i in new_indexes]
    start = len(new_texts) - max_sentence_count
    output = new_texts[start:]
    return output

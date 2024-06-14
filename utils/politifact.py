import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from params import POLITIFACT_TITLE, POLITIFACT_COMMENT, POLITIFACT_CONTENT


def load() -> pd.DataFrame:
    titles = pd.read_csv(filepath_or_buffer=POLITIFACT_TITLE, sep='\t', names=['id', 'title']).drop_duplicates()
    contents = pd.read_csv(filepath_or_buffer=POLITIFACT_CONTENT, sep='\t', header=0).drop_duplicates()
    comments = pd.read_csv(filepath_or_buffer=POLITIFACT_COMMENT, sep='\t', header=0).drop_duplicates()

    df = pd.merge(titles, contents, on='id', how='inner')
    df = pd.merge(df, comments, on='id', how='inner')
    df = df.dropna()
    df['comment'] = df['comment'].str.split("::")
    df['content'] = df['content'].apply(nltk.tokenize.sent_tokenize)
    # print(df.columns)  # ['id', 'title', 'label', 'content', 'comment']
    return df


def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encoding. Build two columns by transforming true labels as (1, 0) else (0, 1)
    df['fake_news'] = df['label']
    df['true_news'] = 1 - df['label']

    # Convert literal labels to binary classes
    df['label'] = df['fake_news'].map(lambda label: 'fake' if label == 1 else 'true')
    return df


def get_XYL(df: pd.DataFrame):
    # Filter unwanted columns
    X = df[['claim', 'evidence']]
    y = df[['true_news', 'fake_news']]
    l = df['label']
    return X, y, l


def get_splits(df: pd.DataFrame, test_size: float, val_size: float = 0):
    # Group by claim_id. Split dataset without having duplicated claim_id in train and test set
    train, test = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        stratify=df['fake_news'],
        random_state=26
    )

    if val_size <= 0:
        return train, test

    # Validation perc with respect to current train dataset. Example:
    # If wanted val_set is 0.2% the original dataset, then the percentage to use is "0.2/0.8 = 0.25%"
    val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train,
        test_size=val_size,
        shuffle=True,
        stratify=train['label'],
        random_state=26
    )

    return train, test, val



def get():
    df = load()
    df = preprocess_labels(df)
    return df


if __name__ == '__main__':
    df = get()
    print(df.head())
    count = df['content'].apply(len)
    print(df['content'].iloc[0])
    print(count.value_counts())

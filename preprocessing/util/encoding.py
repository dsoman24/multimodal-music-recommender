import numpy as np
from gensim.models import Word2Vec

def encode_multihot(df, col):
    """
    If col is a list of labels, this function will return a new column with a list of 1s and 0s (multi-hot encoding)

    This function will also return the unique labels in the dataset for decoding purposes
    """
    labels = df[[col]]
    unique_labels = tuple(set(label for sublist in labels[col] for label in sublist))
    print(f"Unique labels: {len(unique_labels)}")
    return labels.assign(multi_hot=labels['mbtag'].apply(lambda x: [1 if label in x else 0 for label in unique_labels])), unique_labels

def decode_multihot(df, decoding_labels, encoded_col='multi_hot'):
    """
    Decodes a multi-hot encoded column
    """
    return df.assign(decoded_multihot=df[encoded_col].apply(lambda x: [label for i, label in enumerate(decoding_labels) if x[i] == 1]))

def w2v_embedding(sentences, vector_size, window_size, min_count, workers=4, sg=1):
    """
    Uses Word2Vec to compute embeddings of each label per sentence (list of labels).
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        workers=workers,
        sg=sg  # Skip-gram; use sg=0 for CBOW
    )
    return {genre: model.wv[genre] for genre in model.wv.index_to_key}

def compute_average_embedding(labels, embedding_dict):
    """
    Computes average embedding vector for a list of labels.
    """
    vectors = [embedding_dict[label] for label in labels if label in embedding_dict]
    if not vectors:
        return np.zeros(next(iter(embedding_dict.values())).shape)
    return np.mean(vectors, axis=0)
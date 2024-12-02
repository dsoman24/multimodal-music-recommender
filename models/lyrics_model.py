

import pandas as pd


class LyricsModel:
    """
    Temporary class mimicking the MetadataModel's structure to test the e2e recommender.
    """

    def __init__(self, common_track_ids, embedding_technique='roberta', debug=False):
        self.debug = debug
        self.common_track_ids = common_track_ids
        self.path = f'data/embeddings/lyrics_embeddings/embedding_{embedding_technique}.pkl'

    def get_lyrics_embeddings(self):
        lyrics_embeddings = pd.read_pickle(self.path)
        lyrics_embeddings = lyrics_embeddings[lyrics_embeddings['song_id'].isin(self.common_track_ids)]
        return lyrics_embeddings

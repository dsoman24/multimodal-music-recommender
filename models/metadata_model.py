# All the metadata model does is create an embedding for the metadata for each song in the dataset.
# This embedding is used in the fusion step.

import numpy as np
import pandas as pd
from models.util.encoding import TextEncoder


class MetadataModel:

    def __init__(self, metadata_df, debug=False, configs={}):
        self.metadata_df = metadata_df
        self.debug = debug
        # songs
        self.title_encoder = TextEncoder(
            embedding_technique=configs.get('title_encoder', {}).get('embedding_technique', 'w2v'),
            config=configs.get('title_encoder', {}),
            aggregation_method=configs.get('title_encoder', {}).get('aggregation_method', 'mean'),
            load_from_file=configs.get('title_encoder', {}).get('load_from_file', True),
            save_to_file=configs.get('title_encoder', {}).get('save_to_file', True),
            file_dir_name='title_embeddings',
            debug=self.debug
        )
        # albums
        self.release_encoder = TextEncoder(
            embedding_technique=configs.get('release_encoder', {}).get('embedding_technique', 'w2v'),
            config=configs.get('release_encoder', {}),
            aggregation_method=configs.get('release_encoder', {}).get('aggregation_method', 'mean'),
            load_from_file=configs.get('release_encoder', {}).get('load_from_file', True),
            save_to_file=configs.get('release_encoder', {}).get('save_to_file', True),
            file_dir_name='release_embeddings',
            debug=self.debug
        )
        # artist name
        self.artist_name_encoder = TextEncoder(
            embedding_technique=configs.get('artist_name_encoder', {}).get('embedding_technique', 'w2v'),
            config=configs.get('artist_name_encoder', {}),
            aggregation_method=configs.get('artist_name_encoder', {}).get('aggregation_method', 'mean'),
            load_from_file=configs.get('artist_name_encoder', {}).get('load_from_file', True),
            save_to_file=configs.get('artist_name_encoder', {}).get('save_to_file', True),
            file_dir_name='artist_name_embeddings',
            debug=self.debug
        )
        # other columns to append to the vectorization
        self.columns_to_append = [
            'duration',
            'artist_familiarity',
            'artist_hotttnesss',
            'year'
        ]

    def _generate_metadata_embeddings_by_category(self):
        """
        Generates embeddings for each metadata column.
        """
        title_sentences = [title.split() for title in self.metadata_df['title'].tolist()]
        self.title_encoder.generate_embeddings(title_sentences)
        release_sentences = [release.split() for release in self.metadata_df['release'].tolist()]
        self.release_encoder.generate_embeddings(release_sentences)
        artist_name_sentences = [artist_name.split() for artist_name in self.metadata_df['artist_name'].tolist()]
        self.artist_name_encoder.generate_embeddings(artist_name_sentences)

    def get_metadata_embeddings(self):
        """
        Returns metadata embeddings for each song in the dataset.
        """
        self._generate_metadata_embeddings_by_category()
        metadata_embeddings_df = self.metadata_df.copy()
        metadata_embeddings_df['title_embedding'] = metadata_embeddings_df['title'].apply(lambda x: self.title_encoder.aggregate_embeddings(x))
        metadata_embeddings_df['release_embedding'] = metadata_embeddings_df['release'].apply(lambda x: self.release_encoder.aggregate_embeddings(x))
        metadata_embeddings_df['artist_name_embedding'] = metadata_embeddings_df['artist_name'].apply(lambda x: self.artist_name_encoder.aggregate_embeddings(x))
        for col in self.columns_to_append:
            metadata_embeddings_df[col] = self.metadata_df[col]
        metadata_embeddings_df['metadata_embedding'] = metadata_embeddings_df.apply(
            lambda row: np.concatenate([
                row['title_embedding'],
                row['release_embedding'],
                row['artist_name_embedding'],
                np.array([row[col] for col in self.columns_to_append])
            ]),
            axis=1
        )
        return metadata_embeddings_df[['track_id', 'metadata_embedding']]
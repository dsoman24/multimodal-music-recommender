import os
import pandas as pd
from tqdm import tqdm
import models.util.encoding as encoding
from models.util.encoding import TextEncoder
import numpy as np
import models.util.clustering as clustering
import matplotlib.pyplot as plt

DATA_DIR_NAME = 'data'
INTERMEDIATE_DATA_DIR_NAME = 'intermediate_output'
LABEL_EMBEDDING_DIR_NAME = 'label_embeddings'
EMBEDDING_FILE_PREFIX = 'embedding_'

SEED = 42

class DataProvider:
    """
    This class is the first stage of the model architecture training pipeline.

    The primary purpose is to generate the 'ground truth' training classes.
    """

    def __init__(
            self,
            data_dir=DATA_DIR_NAME,
            label_embedding_technique="w2v",
            embedding_config={},
            load_embeddings_from_file=False,
            save_embeddings_to_file=False,
            clustering_method="kmeans",
            debug=False
        ):
        """
        Args:
            data_dir: directory where pickle files of intermediate data is held.
            label_embedding_technique: accepts `'w2v'` or `'multihot'`
        """
        self.data_dir = data_dir
        self.debug=debug
        if label_embedding_technique not in ('w2v', 'multihot', 'roberta'):
            label_embedding_technique = 'w2v'
        self.label_encoder = TextEncoder(
            data_dir,
            label_embedding_technique,
            embedding_config,
            load_from_file=load_embeddings_from_file,
            save_to_file=save_embeddings_to_file,
            file_dir_name=LABEL_EMBEDDING_DIR_NAME,
            debug=debug
        )
        if clustering_method not in ('kmeans', 'dbscan'):
            clustering_method = 'kmeans'
        self.label_embedding_technique = label_embedding_technique
        self.clustering_method = clustering_method
        self.labels_df = None
        self.tagged_metadata_df = None
        self.untagged_metadata_df = None
        self.lyrics_df = None
        # TODO: add audio_df
        self.audio_df = None
        self.user_data_df = None
        self.train_test_split_mask = None

    def load_data(self, test_size=0.2):
        intermediate_data_dir = os.path.join(self.data_dir, INTERMEDIATE_DATA_DIR_NAME)
        self._print_debug("Reading labels.")
        self.labels_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'labels.pkl'))
        self._print_debug("Reading tagged metadata.")
        self.tagged_metadata_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'tagged_metadata.pkl'))
        self._print_debug("Reading untagged metadata.")
        self.untagged_metadata_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'untagged_metadata.pkl'))
        self._print_debug("Reading lyrics.")
        self.lyrics_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'lyrics.pkl'))
        self._print_debug("Reading user data.")
        self.user_data_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'user_data.pkl'))

        # filter out records that are not in all train dataframes
        # train dataframes are labels_df, tagged_metadata_df, lyrics_df
        # TODO: when audio_df is added, add it to the common track ids logic
        common_track_ids = set(self.labels_df['track_id']) & set(self.tagged_metadata_df['track_id']) & set(self.lyrics_df['song_id'])
        self.labels_df = self.labels_df[self.labels_df['track_id'].isin(common_track_ids)]
        self.tagged_metadata_df = self.tagged_metadata_df[self.tagged_metadata_df['track_id'].isin(common_track_ids)]
        self.lyrics_df = self.lyrics_df[self.lyrics_df['song_id'].isin(common_track_ids)]
        assert self.labels_df.shape[0] == self.tagged_metadata_df.shape[0] == self.lyrics_df.shape[0], "Dataframes do not have the same number of records."
        num_records = self.labels_df.shape[0]
        self.train_test_split_mask = self._train_test_split(num_records, test_size=test_size)

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def _train_test_split(self, num_records, test_size=0.2):
        """
        Returns a boolean mask for splitting the data into training and testing sets.

        The mask is True for training records and False for testing records.
        """
        return np.random.choice([True, False], num_records, p=[1-test_size, test_size])

    def cluster(self, df, col, config={}):
        """
        Returns cluster labels.
        """
        self._print_debug(f"Clustering {col}.")
        if self.clustering_method == 'kmeans' and 'n_classes' not in config:
           config['n_classes'] = 10
        if self.clustering_method == 'dbscan':
            if 'eps' not in config:
                config['eps'] = 0.5
            if 'min_samples' not in config:
                config['min_samples'] = 5
        clusters = clustering.cluster_encodings(df, col, method=self.clustering_method, config=config)
        return clusters.labels_

    def generate_training_classes(self, cluster_config={}):
        """
        Constructs training classes on the mbtag labels via embeddings and clustering and are added to the `labels_df`.

        Embeddings are created according to `self.label_embedding_technique`.

        Clusters are created according to `self.clustering_method`.

        Creates 'cluster' column, which is the training class.
        """

        self._print_debug("Generating training classes.")
        label_lists = []
        for label_list in list(self.labels_df['mbtag']):
            label_lists.append(label_list)
        self.label_encoder.generate_embeddings(label_lists)
        self._print_debug('Aggregating embeddings for all tracks.')
        self.labels_df['mbtag_embedding'] = self.labels_df['mbtag'].apply(lambda x: self.label_encoder.aggregate_embeddings(x))
        self.labels_df['cluster'] = self.cluster(self.labels_df, 'mbtag_embedding', cluster_config)

    def plot_cluster_distribution(self):
        counts = self.labels_df[['cluster']].value_counts()
        counts = counts.reset_index()
        counts.columns = ['cluster', 'count']
        plt.bar(x=counts['cluster'], height=counts['count'])
        plt.title('Number of elements per cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of elements')
        plt.show()
import os
import pandas as pd
from tqdm import tqdm
import models.util.encoding as encoding
from models.util.encoding import LabelEncoder
import models.util.clustering as clustering
import matplotlib.pyplot as plt

DATA_DIR_NAME = 'data'
INTERMEDIATE_DATA_DIR_NAME = 'intermediate_output'
EMBEDDING_DIR_NAME = 'embeddings'
LABEL_EMBEDDING_DIR_NAME = 'label_embeddings'
EMBEDDING_FILE_PREFIX = 'embedding_'


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
        self.label_encoder = LabelEncoder(
            data_dir,
            label_embedding_technique,
            embedding_config,
            load_from_file=load_embeddings_from_file,
            save_to_file=save_embeddings_to_file,
            debug=debug
        )
        if clustering_method not in ('kmeans', 'dbscan'):
            clustering_method = 'kmeans'
        self.label_embedding_technique = label_embedding_technique
        self.clustering_method = clustering_method
        self.labels_df = None
        self.tagged_metadata_df = None
        self.lyrics_df = None
        self.user_data_df = None

    def load_data(self):
        intermediate_data_dir = os.path.join(self.data_dir, INTERMEDIATE_DATA_DIR_NAME)
        self._print_debug("Reading labels.")
        self.labels_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'labels.pkl'))
        self._print_debug("Reading metadata.")
        self.tagged_metadata_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'tagged_metadata.pkl'))
        self._print_debug("Reading lyrics.")
        self.lyrics_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'lyrics.pkl'))
        self._print_debug("Reading user data.")
        self.user_data_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'user_data.pkl'))

    def _print_debug(self, message):
        if self.debug:
            print(message)

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
        Constructs training classes on the mbtag labels via embeddings and clustering and are added to the labels_df.

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
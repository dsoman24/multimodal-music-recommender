import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
import preprocessing.util.encoding as encoding
import models.util.clustering as clustering

class DataProvider:
    """
    This class is the first stage of the model architecture training pipeline.

    The primary purpose is to generate the 'ground truth' training classes.
    """

    def __init__(
            self,
            data_dir=os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'),
            label_embedding_technique="w2v",
            label_w2v_embedding_size=100,
            label_w2v_window_size=5,
            label_w2v_min_count=1,
            debug=False
        ):
        """
        Args:
            data_dir: directory where pickle files of intermediate data is held.
            label_embedding_technique: accepts `'w2v'` or `'multihot'`
        """
        if label_embedding_technique not in ('w2v', 'multihot'):
            label_embedding_technique = 'w2v'
        self.label_embedding_technique = label_embedding_technique
        intermediate_data_dir = os.path.join(data_dir, 'intermediate_output')
        self.debug=debug
        self._print_debug("Reading labels...")
        self.labels_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'labels.pkl'))
        self._print_debug("Reading metadata...")
        self.metadata_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'metadata.pkl'))
        self._print_debug("Reading music_analysis...")
        self.music_analysis_df = pd.read_pickle(os.path.join(intermediate_data_dir, 'music_analysis.pkl'))
        # w2v embeddings config
        self.label_w2v_embedding_size = label_w2v_embedding_size
        self.label_w2v_window_size = label_w2v_window_size
        self.label_w2v_min_count = label_w2v_min_count

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def generate_training_classes(self, num_classes=10):
        """
        Constructs `self.num_classes` training classes on the mbtag labels via embeddings and kmeans clustering.

        Embeddings are created according to `self.label_embedding_technique`.

        Creates 'cluster' column, which is the training class.
        """
        if self.label_embedding_technique == 'multihot':
            self._print_debug("Encoding labels into multihot.")
            labels_multihot, decoding_labels = encoding.encode_multihot(self.labels_df, 'mbtag')
            self._print_debug("Running kmeans on multihot encoded labels.")
            kmeans = clustering.cluster_encodings(labels_multihot, 'multi_hot', num_classes)
            labels_multihot['cluster'] = kmeans.labels_
            self.labels_df = labels_multihot
        elif self.label_embedding_technique =='w2v':
            label_lists = []
            for label in list(self.labels_df['mbtag']):
                label_lists.append(label)
            label_embeddings = encoding.w2v_embedding(
                sentences=label_lists,
                vector_size=self.label_w2v_embedding_size,
                window_size=self.label_w2v_window_size,
                min_count=self.label_w2v_min_count,
            )
            self.labels_df['mbtag_embedding'] = self.labels_df['mbtag'].apply(lambda x: encoding.compute_average_embedding(x, label_embeddings))
            kmeans = clustering.cluster_encodings(self.labels_df, 'mbtag_embedding', num_classes)
            self.labels_df['cluster'] = kmeans.labels_

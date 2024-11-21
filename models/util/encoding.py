import os
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel

DATA_DIR_NAME = 'data'
INTERMEDIATE_DATA_DIR_NAME = 'intermediate_output'
EMBEDDING_DIR_NAME = 'embeddings'
LABEL_EMBEDDING_DIR_NAME = 'label_embeddings'
EMBEDDING_FILE_PREFIX = 'embedding_'

class LabelEncoder:

    def __init__(
            self,
            data_dir=DATA_DIR_NAME,
            label_embedding_technique='w2v',
            config={},
            load_from_file=False,
            save_to_file=False,
            aggregation_method='mean',
            debug=False
        ):
        self.data_dir = data_dir
        self.label_embedding_technique = label_embedding_technique
        if label_embedding_technique not in ('w2v', 'multihot', 'roberta'):
            label_embedding_technique = 'w2v'
        self.aggregation_method = aggregation_method
        if aggregation_method not in ('mean', 'sum'):
            aggregation_method = 'mean'
        self.config = config
        self.debug = debug
        self.load_from_file = load_from_file
        self.save_to_file = save_to_file
        self.embedding_dict = None


    def _print_debug(self, message):
        if self.debug:
            print(message)

    def generate_embeddings(self, sentences):
        """
        Generates embeddings dictionary and assigns the self.embeddings attribute.

        Optionally reads from file if self.load_from_file is True.
        """
        if self.load_from_file:
            self.embedding_dict = self._read_label_embedding_dict()
        if self.embedding_dict is None:
            self._print_debug("Generating label embeddings, not reading from file.")
            if self.label_embedding_technique == 'w2v':
                self.embedding_dict = self._generate_w2v_embeddings(sentences)
            elif self.label_embedding_technique == 'roberta':
                self.embedding_dict =  self._generate_roberta_embeddings(sentences)
            elif self.label_embedding_technique == 'multihot':
                self.embedding_dict =  self._generate_multihot_embeddings(sentences)
            else:
                raise ValueError(f"Invalid label embedding technique: {self.label_embedding_technique}")
            if self.save_to_file and self.embedding_dict is not None:
                self._save_label_embedding_dict()

    def _generate_w2v_embeddings(self, sentences):
        """
        Uses Word2Vec to compute embeddings of each label per sentence (list of labels).
        """
        self._print_debug("Generating Word2Vec embeddings.")
        model = Word2Vec(
            sentences=sentences,
            vector_size=self.config.get('vector_size', 100),
            window=self.config.get('window_size', 5),
            min_count=self.config.get('min_count', 1),
            workers=self.config.get('workers', 4),
            sg=self.config.get('sg', 1)  # Skip-gram; use sg=0 for CBOW
        )
        return {label: model.wv[label] for label in model.wv.index_to_key}

    def _generate_roberta_embeddings(self, sentences):
        self._print_debug("Generating RoBERTa embeddings.")
        tokenizer = RobertaTokenizer.from_pretrained(self.config.get('model_name', 'roberta-base'))
        model = RobertaModel.from_pretrained(self.config.get('model_name', 'roberta-base'))
        embedding_dict = {}
        unique_labels = set(label for sentence in sentences for label in sentence)
        for label in unique_labels:
            inputs = tokenizer(label, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embedding_dict[label] = embedding
        return embedding_dict

    def _generate_multihot_embeddings(self, sentences):
        """
        Returns multi-hot encoded labels and unique labels.
        """
        self._print_debug("Generating multi-hot encoded labels.")
        unique_labels = tuple(set(label for sentence in sentences for label in sentence))
        embedding_dict = {}
        for i, label in enumerate(unique_labels):
            embedding_dict[label] = np.eye(len(unique_labels))[i]
        return embedding_dict

    def aggregate_embeddings(self, sentence):
        """
        Returns an aggregate embedding for a list of labels (sentence).
        """
        vectors = [self.embedding_dict[label] for label in sentence if label in self.embedding_dict]
        if not vectors:
            return np.zeros(next(iter(self.embedding_dict.values())).shape)
        if self.aggregation_method == 'mean':
            return np.mean(vectors, axis=0)
        elif self.aggregation_method == 'sum':
            return np.sum(vectors, axis=0)
        return None

    def _save_label_embedding_dict(self):
        file_dir = os.path.join(
            self.data_dir,
            INTERMEDIATE_DATA_DIR_NAME,
            LABEL_EMBEDDING_DIR_NAME,
        )
        os.makedirs(file_dir, exist_ok=True)
        full_path = os.path.join(file_dir, f'{EMBEDDING_FILE_PREFIX}{self.label_embedding_technique}.pkl')
        self._print_debug(f"Saving label embedding dictionary to {full_path}.")
        pd.to_pickle(self.embedding_dict, full_path)

    def _read_label_embedding_dict(self):
        path = os.path.join(
            self.data_dir,
            INTERMEDIATE_DATA_DIR_NAME,
            LABEL_EMBEDDING_DIR_NAME,
            f'{EMBEDDING_FILE_PREFIX}{self.label_embedding_technique}.pkl'
        )
        self._print_debug(f"Reading label embedding dictionary from {path}.")
        if not os.path.exists(path):
            self._print_debug(f"File {path} not found.")
            return None
        return pd.read_pickle(path)

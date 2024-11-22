import os
import pickle

import numpy as np

# This class handles the fusion step of the model architecture pipeline.

DATA_DIR_NAME = 'data'
OUTPUT_DIR_NAME = 'fusion_output'

class FusionStep:

    def __init__(
            self,
            fusion_method='concat',
            data_dir=DATA_DIR_NAME,
            output_dir=OUTPUT_DIR_NAME,
            debug=False
        ):
        self.fusion_method = fusion_method
        if fusion_method not in (
            'concat',
            'mean',
            'max',
            'min',
            'mean-truncate',
            'max-truncate',
            'min-truncate',
            'mean-pad',
            'max-pad',
            'min-pad'
        ):
            self._print_debug(f"Invalid fusion method: {fusion_method}. Defaulting to 'concat'.")
            fusion_method = 'concat'
        # list of np matrices
        self.components = []
        # np 2d matrix with training features for the fusion model.
        self.fused_vectors = None
        self.output_path = os.path.join(data_dir, output_dir)
        self.debug = debug

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def load_component(self, component):
        """
        A component is one of the sets of vectors to be fused. This method loads one component.
        """
        self.components.append(component)

    def load_components(self, components):
        """
        A component is one of the sets of vectors to be fused. This method loads multiple components.
        """
        for component in components:
            self.load_component(component)

    def fuse(self):
        if self.fusion_method == 'concat':
            self._concatenate()
        elif self.fusion_method.startswith('mean'):
            self._mean()
        elif self.fusion_method.startswith('max'):
            self._max()
        elif self.fusion_method.startswith('min'):
            self._min()

    def _pad(self):
        max_cols = max(component.shape[1] for component in self.components)
        padded_components = []
        for component in self.components:
            if component.shape[1] < max_cols:
                padding = max_cols - component.shape[1]
                component = np.pad(component, ((0, 0), (0, padding)), mode='constant', constant_values=0)
            padded_components.append(component)
        return padded_components

    def _truncate(self):
        min_cols = min(component.shape[1] for component in self.components)
        truncated_components = [component[:, :min_cols] for component in self.components]
        return truncated_components

    def _pad_or_truncate(self):
        if self.fusion_method.endswith('-pad'):
            return self._pad()
        elif self.fusion_method.endswith('-truncate'):
            return self._truncate()
        return self.components

    def _concatenate(self):
        """
        Concatenates the vectors of the components.
        """
        self.fused_vectors = np.concatenate(self.components, axis=1)

    def _mean(self):
        """
        Averages the vectors of the components.
        """
        components = self._pad_or_truncate()
        self.fused_vectors = np.mean(components, axis=0)

    def _max(self):
        """
        Takes the maximum value of the vectors of the components.
        """
        components = self._pad_or_truncate()
        self.fused_vectors = np.max(components, axis=0)

    def _min(self):
        """
        Takes the minimum value of the vectors of the components.
        """
        components = self._pad_or_truncate()
        self.fused_vectors = np.min(components, axis=0)

    def save_fused(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_path, 'fused_vectors.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self.fused_vectors, f)
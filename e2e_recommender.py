import numpy as np
import pandas as pd
from models.data_provider import DataProvider
from models.metadata_model import MetadataModel
from models.fusion import FusionStep
from models.train import FusionModelTrainer
from models.recommender import Recommender, RecommenderEvaluator
import os
import hashlib
import json
import time


E2E_OUTPUT_PATH = 'e2e_output'

class E2ERecommender:

    def __init__(self, config, debug=False):
        self.debug = debug
        self.data_provider_config = config['data_provider']
        self.data_provider = DataProvider(
            label_embedding_technique=self.data_provider_config['label_embedding_technique'],
            embedding_config=self.data_provider_config['embedding_config'],
            save_embeddings_to_file=self.data_provider_config['save_embeddings_to_file'],
            load_embeddings_from_file=self.data_provider_config['load_embeddings_from_file'],
            clustering_method=self.data_provider_config['clustering_method'],
            debug=self.debug
        )
        self.data_provider.load_data(test_size=self.data_provider_config.get('test_size', 0.2))
        self.metadata_model_config = config['metadata_model']
        self.fusion_config = config['fusion']
        self.train_config = config['train']
        self.recommender_config = config['recommender']
        self.num_classes = 0
        self.log = []
        self.fingerprint = hashlib.md5(str(time.time()).encode()).hexdigest()
        self.add_to_log({
            'e2e_config': config
        })

    def add_to_log(self, item):
        self.log.append(item)

    def training_classes_step(self):
        # 1. Generate training classes
        self.data_provider.generate_training_classes(cluster_config=self.data_provider_config['cluster_config'])
        self.num_classes = int(self.data_provider.labels_df[['cluster']].nunique().iloc[0])

    def multimodal_features_step(self):
        # 2. Generate multimodal features
        metadata_model = MetadataModel(
            metadata_df=self.data_provider.tagged_metadata_df,
            configs={
                'title_encoder': self.metadata_model_config['title_encoder_config'],
                'release_encoder': self.metadata_model_config['release_encoder_config'],
                'artist_name_encoder': self.metadata_model_config['artist_name_encoder_config'],
            },
            debug=self.debug
        )
        embeddings_df = metadata_model.get_metadata_embeddings()
        # TODO: add lyrics embeddings here and to return array
        metadata_embedding = np.array(embeddings_df['metadata_embedding'].tolist())
        return [metadata_embedding]

    def fusion_step(self, multimodal_features):
        # 3. Fusion Step
        fusion_step = FusionStep(
            fusion_method=self.fusion_config['fusion_method'],
            debug=self.debug
        )
        fusion_step.load_components(multimodal_features)
        fusion_step.fuse()
        return fusion_step.fused_vectors

    def train_fusion_model_step(self, fused_embedding):
        train_test_split_mask = self.data_provider.train_test_split_mask
        train_data = fused_embedding[train_test_split_mask]
        train_labels = np.array(self.data_provider.labels_df['cluster'][train_test_split_mask])

        test_data = fused_embedding[~train_test_split_mask]
        test_labels = np.array(self.data_provider.labels_df['cluster'][~train_test_split_mask])
        trainer = FusionModelTrainer(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            hidden_sizes=self.train_config.get('hidden_sizes', [128, 256, 512]),
            num_classes=self.num_classes,  # make sure this is equal to the number of training classes generated
            config=self.train_config,
            debug=False
        )
        trainer.train()
        self.add_to_log({
            'train_accuracies': trainer.train_accuracies,
            'test_accuracies': trainer.test_accuracies
        })
        return trainer

    def recommender_system_step(self, trainer):
        # 5. Recommender System
        test_predicted = trainer.get_test_inferences().tolist()
        test_track_ids = self.data_provider.tagged_metadata_df['song_id'][~self.data_provider.train_test_split_mask].tolist()
        test_predictions_df = pd.DataFrame(data=[test_track_ids, test_predicted], index=['song_id', 'prediction']).T

        user_data_df = self.data_provider.user_data_df.sample(frac=self.recommender_config.get('sample_frac', 0.001))

        recommender = Recommender(
            user_data_df=user_data_df,
            predictions_df=test_predictions_df,
            debug=self.debug
        )

        recommender_evaluator = RecommenderEvaluator(recommender)
        results = recommender_evaluator.evaluate(n=self.recommender_config.get('n_recommendations', 500))
        self.add_to_log(results)

    def execute(self):
        """
        Executes the entire pipeline
        """
        # 1. Generate training classes
        print("Training Classes Step")
        self.training_classes_step()

        # 2. Generate multimodal features
        print("Multimodal Features Step")
        metadata_embedding = self.multimodal_features_step()

        # 3. Fusion Step
        print("Fusion Step")
        fused_embedding = self.fusion_step(multimodal_features=metadata_embedding)

        # 4. Train Fusion Model
        print("Train Fusion Model Step")
        trainer = self.train_fusion_model_step(fused_embedding)
        # save training plot

        # 5. Recommender System
        print("Recommender System Step")
        self.recommender_system_step(trainer)

        # Save the log
        self.save_log()
        # save training plot
        trainer.save_plot(path=os.path.join(E2E_OUTPUT_PATH, f"execution_{self.fingerprint}", 'training_plot.png'))

    def save_log(self):
        # Create a unique fingerprint for this execution

        output_dir = os.path.join(E2E_OUTPUT_PATH, f"execution_{self.fingerprint}")

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the log to a file
        log_path = os.path.join(output_dir, 'log.json')
        with open(log_path, 'w') as log_file:
            json.dump(self.log, log_file, indent=4)

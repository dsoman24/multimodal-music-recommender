import os
from preprocessing.util.tag_processor import TagProcessor
from preprocessing.util.user_data_processor import UserDataProcessor
from preprocessing.util.lyrics_processor import LyricsProcessor
from preprocessing.util.metadata_processor import MetadataProcessor

class Processor:
    """
    This file reads data from datasets and processes them for use in the model.

    This primarily involves joining tables and writing them to pkl files.
    """

    def __init__(self,
                 data_dir='data',
                 output_dir_name='intermediate_output',
                 user_data_file='train_triplets.txt',
                 lyrics_db_file='mxm_dataset.db',
                 debug_messages=False):
        self.data_dir = data_dir
        self.output_dir_name = output_dir_name
        self.debug_messages = debug_messages
        user_data_path = os.path.join(data_dir, user_data_file)
        lyrics_db_path = os.path.join(data_dir, lyrics_db_file)
        self.user_data_processor = UserDataProcessor(user_data_path, debug_messages)
        self.lyrics_processor = LyricsProcessor(lyrics_db_path)
        self.tag_processor = TagProcessor(data_dir, debug_messages)
        self.metadata_processor = MetadataProcessor(data_dir, debug_messages)
        self.tagged_metadata_df = None  # contains track metadata and tags
        self.lyrics_df = None
        self.user_data_df = None
        self.labels_df = None

    def process(self):
        """
        Reads the datasets and processes them.
        """
        self.user_data_processor.process()
        self.tag_processor.process()
        self.metadata_processor.process()
        self.user_data_df = self.user_data_processor.df
        self.lyrics_processor.process_all()
        self.lyrics_df = self.lyrics_processor.get_lyrics_data()
        tag_df = self.tag_processor.df
        metadata_df = self.metadata_processor.df
        self.tagged_metadata_df = metadata_df.merge(tag_df, on='artist_id', how='left')
        self.tagged_metadata_df = self.tagged_metadata_df.dropna(subset=['mbtag'])
        self.labels_df = self.tagged_metadata_df[['mbtag']]

    def _print_debug(self, message):
        if self.debug_messages:
            print(message)

    def save_data(self):
        """
        Saves the processed data to pkl files in the specified output directory.
        """
        output_dir = os.path.join(self.data_dir, self.output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        self.labels_df.to_pickle(os.path.join(output_dir, 'labels.pkl'))
        self.user_data_df.to_pickle(os.path.join(output_dir, 'user_data.pkl'))
        self.tagged_metadata_df.to_pickle(os.path.join(output_dir, 'tagged_metadata.pkl'))
        self.lyrics_df.to_pickle(os.path.join(output_dir, 'lyrics.pkl'))
import pandas as pd
from tqdm import tqdm

class UserDataProcessor:

    def __init__(self, data_dir, debug_messages=False):
        self.data_dir = data_dir
        self.debug_messages = debug_messages

    def _output_debug_message(self, message):
        if self.debug_messages:
            print(message)

    def _read_dataset(self):
        """
        Reads the user triplets dataset.
        """
        with open(self.data_dir, 'r') as file:
            data = []
            for line in tqdm(file, desc="Reading user listening history file"):
                user_id, song_id, play_count = line.strip().split('\t')
                data.append({
                    'user_id': user_id,
                    'song_id': song_id,
                    'play_count': int(play_count)
                })
            self._output_debug_message(f"Read {len(data)} user triplets.")
            return data
        self._output_debug_message("Failed to read user triplets.")
        return []

    def process_user_data(self):
        """
        Processes the user data and returns a list of dictionaries.
        """
        data = self._read_dataset()
        df = pd.DataFrame(data)
        return df

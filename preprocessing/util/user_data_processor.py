import pandas as pd
from tqdm import tqdm

class UserDataProcessor:

    def __init__(self, data_dir, debug_messages=False):
        self.data_dir = data_dir
        self.debug_messages = debug_messages
        self.df = None

    def print_debug(self, message):
        if self.debug_messages:
            print(message)

    def read(self):
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
            self.print_debug(f"Read {len(data)} user triplets.")
            return data
        self.print_debug("Failed to read user triplets.")
        return []

    def process(self):
        """
        Processes the user data and returns a list of dictionaries.
        """
        data = self.read()
        self.df = pd.DataFrame(data)

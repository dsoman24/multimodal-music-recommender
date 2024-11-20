import sqlite3

import pandas as pd
from preprocessing.util.db_processor import DatabaseProcessor


class MetadataProcessor(DatabaseProcessor):

    def __init__(self, data_dir, debug_messages=False):
        super().__init__(data_dir, 'track_metadata.db', debug_messages)

    def read(self):
        conn = None
        cursor = None
        metadata_df = None
        try:
            super().print_debug(f"Reading metadata from {self.path}")
            conn = sqlite3.connect(self.path)
            metadata_df = pd.read_sql_query("SELECT * FROM songs", conn)
        except Exception as e:
            super().print_debug(f"Error reading metadata database: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        if metadata_df is None:
            super().print_debug("No metadata found in database.")
            return None
        return metadata_df

    def process(self):
        """
        Processes the artist term database, returning a DataFrame with artist tags.
        """
        metadata_df = self.read()
        if metadata_df is not None:
            self.df = metadata_df
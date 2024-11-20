import sqlite3

import pandas as pd
from preprocessing.util.db_processor import DatabaseProcessor

class TagProcessor(DatabaseProcessor):

    def __init__(self, data_dir, debug_messages=False):
        super().__init__(data_dir, 'artist_term.db', debug_messages)

    def read(self):
        conn = None
        cursor = None
        mbtags_df = None
        try:
            super().print_debug(f"Reading mbtags from {self.path}")
            conn = sqlite3.connect(self.path)
            mbtags_df = pd.read_sql_query("SELECT * FROM artist_mbtag", conn)
        except Exception as e:
            super().print_debug(f"Error reading mbtag database: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        if mbtags_df is None:
            super().print_debug("No mbtags found in database.")
            return None
        mbtags_df = mbtags_df.groupby('artist_id')['mbtag'].apply(list).reset_index()
        return mbtags_df

    def process(self):
        """
        Processes the artist term database.
        """
        mbtags_df = self.read()
        if mbtags_df is not None:
            self.df = mbtags_df
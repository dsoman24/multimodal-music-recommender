import sqlite3
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm

class LyricsProcessor:

    def __init__(self, db_path) -> None:
        self.db_path = db_path
        self.conn = None
        self.lyrics_df = None
        self.lyrics_pivot = None
        # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.model = RobertaModel.from_pretrained("roberta-base")

    def connect_db(self):
        """Connects to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        print("Database connected.")

    def close_db(self):
        """Closes the SQLite database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def ensure_song_id_column(self):
        """Adds a song_id column to the lyrics table if it doesn't exist."""
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA table_info(lyrics);")
        columns = [column[1] for column in cursor.fetchall()]
        print("Columns in 'lyrics' table:", columns)  # Debug: Print existing columns

        if 'song_id' not in columns:
            cursor.execute("ALTER TABLE lyrics ADD COLUMN song_id TEXT;")
            cursor.execute("UPDATE lyrics SET song_id = track_id;")
            self.conn.commit()
        else:
            print("'song_id' column already exists.")

        cursor.close()


    def process_lyrics_table(self):
        """
        Processes the lyrics table, creating a pivoted dataframe with word counts per track.
        """
        self.ensure_song_id_column()  # ensure song_id is present
        query = "SELECT song_id, word, count FROM lyrics LIMIT 10000"
        self.lyrics_df = pd.read_sql_query(query, self.conn)
        self.lyrics_pivot = self.lyrics_df.pivot_table(
            index='song_id', columns='word', values='count', fill_value=0
        ).reset_index()
        print("Lyrics table processed and pivoted with song_id as index.")

    def display_dataframe(self, df_name):
        """
        Displays the specified dataframe: either 'lyrics_df' or 'lyrics_pivot'.

        Args:
            df_name (str): Name of the dataframe to display ('lyrics_df' or 'lyrics_pivot').
        """
        if df_name == 'lyrics_df' and self.lyrics_df is not None:
            print(self.lyrics_df.head())
        elif df_name == 'lyrics_pivot' and self.lyrics_pivot is not None:
            print(self.lyrics_pivot.head())
        else:
            print(f"Dataframe '{df_name}' is not available or not processed.")

    def get_lyrics_data(self):
        """Returns the processed lyrics pivot table."""
        return self.lyrics_pivot

    #processes the database into df but does not display anything
    def process_all(self):
        """
        Connects to the database, processes the lyrics table, and closes the database connection.
        """
        self.connect_db()
        self.process_lyrics_table()
        self.close_db()
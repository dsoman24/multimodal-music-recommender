import os
import h5py
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sqlite3


class DataProcessor:
    """
    A class to process the Million Song Dataset (MSD) into an easy-to-manipulate format.

    This class does very basic data processing, and is intended to be used as a starting point for more complex
    data cleaning.
    """

    def __init__(self, data_dir, mbtag_db_path=None, read_subset=True, file_extension='.h5', desired_fields=None, debug_messages=False):
        self.data_dir = data_dir
        self.mbtag_db_path = mbtag_db_path
        self.read_subset = read_subset
        self.file_extension = file_extension
        self.desired_fields = desired_fields
        if self.desired_fields is None:
            self.desired_fields = {
                'metadata/songs': [
                    'title',
                    'artist_name',
                    'artist_id',
                    'song_id'
                ],
            }
        self.debug_messages = debug_messages

    def _output_debug_message(self, message):
        if self.debug_messages:
            print(message)

    def _get_hd5_files(self, directory):
        """
        Recursively lists all .h5 or .hd5 files in the given directory and its subdirectories.

        Args:
            directory (str or Path): The root directory to start searching from.

        Returns:
            List[Path]: A list of Path objects pointing to HDF5 files.
        """
        directory = Path(directory)  # Ensure directory is a Path object
        h5_files = list(directory.rglob(f'*{self.file_extension}'))
        self._output_debug_message(f"Found {len(h5_files)} files with extension '{self.file_extension}' in '{directory}'.")
        return h5_files

    def _read_dataset(self):
        """
        Reads the dataset and returns a Pandas DataFrame.
        """
        data = []
        files = self._get_hd5_files(self.data_dir)
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(self.data_dir, file)
            try:
                with h5py.File(file_path, 'r') as h5:
                    # The MSD typically stores songs under the 'metadata/songs' group
                    songs = h5['metadata/songs']
                    num_songs = songs.shape[0]
                    for i in range(num_songs):
                        track = {}
                        for dataset_name in self.desired_fields:
                            dataset = h5[dataset_name]
                            if len(self.desired_fields[dataset_name]) == 0:
                                track[dataset_name] = dataset[dataset_name]
                            else:
                                for field in self.desired_fields[dataset_name]:
                                    try:
                                        # Handle nested fields if necessary
                                        if '/' in field:
                                            group, group_dataset = field.split('/')
                                            track[field] = dataset[field][i].decode('utf-8') if songs[field][i].dtype.kind in {'S', 'O'} else dataset[field][i]
                                        else:
                                            # Decode bytes to string if necessary
                                            if dataset[field].dtype.kind in {'S', 'O'}:
                                                track[field] = dataset[field][i].decode('utf-8') if isinstance(dataset[field][i], bytes) else dataset[field][i]
                                            else:
                                                track[field] = dataset[field][i]
                                    except KeyError:
                                        track[field] = None  # or set a default value
                            data.append(track)
            except Exception as e:
                self._output_debug_message(f"Error reading {file_path}: {e}")

        df = pd.DataFrame(data)
        return df

    def _join_mbtags(self, df):
        """
        Joins the artist mbtags from the MSD dataset.
        """
        if self.mbtag_db_path is None:
            self._output_debug_message("No mbtag database path provided. Skipping mbtag join.")
            return df
        conn = None
        cursor = None
        mbtags_df = None
        try:
            conn = sqlite3.connect(self.mbtag_db_path)
            mbtags_df = pd.read_sql_query("SELECT * FROM artist_mbtag", conn)
        except Exception as e:
            self._output_debug_message(f"Error reading mbtag database: {e}")
            return df
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        if mbtags_df is None:
            self._output_debug_message("No mbtags found in database. Skipping mbtag join.")
            return df
        mbtags_df = mbtags_df.groupby('artist_id')['mbtag'].apply(list).reset_index()
        df = df.merge(mbtags_df, on='artist_id', how='left')
        return df

    def process_dataset_df(self):
        """
        Processes the dataset and returns a Pandas DataFrame.
        """
        df = self._read_dataset()
        df = self._join_mbtags(df)
        return df
import os
import h5py
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def get_hd5_files(directory):
    """
    Recursively lists all .h5 or .hd5 files in the given directory and its subdirectories.

    Args:
        directory (str or Path): The root directory to start searching from.

    Returns:
        List[Path]: A list of Path objects pointing to HDF5 files.
    """
    directory = Path(directory)  # Ensure directory is a Path object
    h5_files = list(directory.rglob('*.h5')) + list(directory.rglob('*.hd5'))
    return h5_files

def read_msd_subset(data_dir, file_extension='.h5', desired_fields=None):
    """
    Reads a subset of the Million Song Dataset and returns a Pandas DataFrame.

    Parameters:
    - data_dir (str): Path to the directory containing MSD HDF5 files.
    - file_extension (str): Extension of the HDF5 files. Default is '.h5'.
    - desired_fields (dict of list of str): Dataset mapped to List of fields to extract for each track.

    Returns:
    - pd.DataFrame: DataFrame where each row corresponds to a track.
    """
    if desired_fields is None:
        desired_fields = {
            'metadata/songs': [
                'track_id',
                'title',
                'artist_name',
                'release',
                'year',
                'tempo',
                'duration',
                'time_signature',
                'key',
                'mode',
                # Add more fields as needed
            ]
    }

    data = []

    # List all files in the directory with the given extension
    files = get_hd5_files(data_dir)
    print(f"Found {len(files)} files with extension '{file_extension}' in '{data_dir}'.")

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, file)
        try:
            with h5py.File(file_path, 'r') as h5:
                # The MSD typically stores songs under the 'metadata/songs' group
                songs = h5['metadata/songs']
                num_songs = songs.shape[0]
                for i in range(num_songs):
                    track = {}
                    for dataset_name in desired_fields:
                        dataset = h5[dataset_name]
                        if len(desired_fields[dataset_name]) == 0:
                            track[dataset_name] = dataset[dataset_name]
                        else:
                            for field in desired_fields[dataset_name]:
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
            print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(data)
    return df

def list_groups(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5:
        print("Available Groups:")
        def print_groups(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"- {name}")
        h5.visititems(print_groups)

def list_datasets(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5:
        dataset_list = []
        print("Available Datasets:")
        def print_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"- {name}")
                dataset_list.append(name)
        h5.visititems(print_datasets)
        return dataset_list

def fields_in_dataset(h5_file_path, group_name):
    with h5py.File(h5_file_path, 'r') as h5:
        group = h5.get(group_name)
        if group is None:
            print(f"No '{group_name}' group found in the HDF5 file.")
            return
        if group.dtype.names is None:
            print(f"No fields found in '{group_name}'.")
            return
        fields = list(group.dtype.names)
        print(f"Available fields in '{group_name}':")
        for field in fields:
            print(f" - {field}")

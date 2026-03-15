# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
import pandas as pd
from pathlib import Path

download_path = Path.cwd()/'sound_classification/UrbanSound8K'
print(f"Audio Data folder: {download_path}")

def metadata_file():
    # Read metadata file
    metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df['absolute_path'] = str(download_path) + '/audio' + df['relative_path']

    # Take relevant columns
    df = df[['relative_path', 'absolute_path', 'classID']]

    return df
    
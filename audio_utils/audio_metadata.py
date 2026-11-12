# ----------------------------
# Project paths, class labels and the UrbanSound8K metadata file
# ----------------------------
import pandas as pd
from pathlib import Path

# Paths are resolved relative to this file, so the scripts work no matter which
# directory they are launched from.
PROJECT_DIR = Path(__file__).resolve().parent.parent
download_path = PROJECT_DIR / 'UrbanSound8K'
MODELS_DIR = PROJECT_DIR / 'trained_models'

# UrbanSound8K classID order
CLASS_LABELS = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]


def metadata_file():
    # Read metadata file
    metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
    df = pd.read_csv(metadata_file)

    # Construct file path by concatenating fold and file name
    df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
    df['absolute_path'] = str(download_path) + '/audio' + df['relative_path']

    # Take relevant columns
    df = df[['relative_path', 'absolute_path', 'fold', 'classID']]

    return df

import torch
from torch.utils.data import DataLoader
from sound_classification_dataset import SoundDS
import audio_utils.audio_metadata as audio_metadata

# Fixed seed so the train/validation split is identical on every run. Training and
# sound_inference_from_model.py both call get_dataloaders() with this default, so a
# model is always evaluated on clips it never trained on.
SPLIT_SEED = 42


# ----------------------------
# Random split of 80:20 between training and validation
# ----------------------------
def split_indices(num_items, val_pct=0.2, seed=SPLIT_SEED):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=generator).tolist()
    num_train = round(num_items * (1 - val_pct))
    return perm[:num_train], perm[num_train:]


def get_dataloaders(df=None, val_pct=0.2, batch_size=16, seed=SPLIT_SEED, max_samples=None):
    """
    Returns (train_dl, val_dl). The training set is augmented and the validation set is not.
    max_samples: optionally subsample the metadata first (quick smoke tests).
    """
    if df is None:
        df = audio_metadata.metadata_file()
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed)
    df = df.reset_index(drop=True)

    train_idx, val_idx = split_indices(len(df), val_pct, seed)
    data_path = audio_metadata.download_path

    train_ds = SoundDS(df.iloc[train_idx], data_path, augment=True)
    val_ds = SoundDS(df.iloc[val_idx], data_path, augment=False)
    print(f"Split (seed={seed}): {len(train_ds)} train / {len(val_ds)} validation items")

    # Create training and validation data loaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl

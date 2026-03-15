import torch
from torch.utils.data import random_split
from sound_classification_dataset import SoundDS
import audio_utils.audio_metadata as audio_metadata


df = audio_metadata.metadata_file()
rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")
# n = random rows
random_n = df.sample(n=10, random_state=42)
print("random_n : ")
print(random_n)

data_path = audio_metadata.download_path
print(f'data_path = {data_path}')

# myds = SoundDS(random_n, data_path)
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

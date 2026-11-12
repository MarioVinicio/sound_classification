# sc_train_final.py
import argparse, os, torch
from torch.utils.data import DataLoader
from sound_classification_dataset import SoundDS
import audio_utils.audio_metadata as audio_metadata
from sound_classification_model import AudioClassifier, device
from sound_classification_training import training

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--save-dir", default=str(audio_metadata.MODELS_DIR))
args = parser.parse_args()

df = audio_metadata.metadata_file()          # all 10 folds, no hold-out
dl = DataLoader(SoundDS(df, audio_metadata.download_path, augment=True),
                batch_size=16, shuffle=True)

model = AudioClassifier().to(device)
training(model, dl, args.epochs)

os.makedirs(args.save_dir, exist_ok=True)
torch.save(model.state_dict(),
           os.path.join(args.save_dir, f"final_model_weights_epoch_{args.epochs}.pth"))
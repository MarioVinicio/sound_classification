import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sound_classification_dataset import SoundDS
import audio_utils.audio_metadata as audio_metadata
from sound_classification_model import AudioClassifier
from sound_classification_training import training

fold_accuracies = []
df = audio_metadata.metadata_file()
num_epochs = 15
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'cross validation device = {device}')
# Progressive training mode - create model once before cross validation
model = AudioClassifier().to(device)

for test_fold in range(1, 11):  # iterate folds 1 through 10

    print(f"\n--- Fold {test_fold} ---")

    # Split by predefined fold column
    train_df = df[df["fold"] != test_fold].reset_index(drop=True)
    test_df  = df[df["fold"] == test_fold].reset_index(drop=True)

    data_path = audio_metadata.download_path
    print(f'data_path = {data_path}')

    # DataLoaders
    train_ds = SoundDS(train_df, data_path)
    test_ds  = SoundDS(test_df, data_path)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    # Cross validation mode
    """
    Fresh model each folder
    model = AudioClassifier().to(device)
    """

    # Train
    training(model, train_dl, num_epochs)

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = (inputs - inputs.mean()) / inputs.std()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"all_labels : {all_labels} - all_preds: {all_preds}")
    acc = accuracy_score(all_labels, all_preds)
    fold_accuracies.append(acc)
    print(f"fold_accuracies: {fold_accuracies}")
    print(f"Fold {test_fold} accuracy: {acc:.4f}")

# Summary
fold_accuracies = np.array(fold_accuracies)
print(f"\nMean accuracy : {fold_accuracies.mean():.4f}")
print(f"Std deviation : {fold_accuracies.std():.4f}")

save_dir = os.getcwd() + "/sound_classification/trained_models"
print("Saving model to:", save_dir)

# Option 1 — weights only (recommended)
# torch.save(model.state_dict(), os.path.join(save_dir, f"cross_val_model_weights_epoch_{num_epochs}.pth"))
torch.save(model.state_dict(), os.path.join(save_dir, f"progressive_training_model_weights_epoch_{num_epochs}.pth"))

# Option 2 — entire model
# torch.save(model, os.path.join(save_dir, f"cross_val_model_full_epoch_{num_epochs}.pth"))
torch.save(model, os.path.join(save_dir, f"progressive_training_model_full_epoch_{num_epochs}.pth"))

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.boxplot(fold_accuracies, tick_labels=["10-Fold CV"])
plt.title("10-Fold Cross-Validation Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.axhline(fold_accuracies.mean(), color="red", linestyle="--", label=f"Mean: {fold_accuracies.mean():.4f}")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("cv_boxplot.png")
plt.show()
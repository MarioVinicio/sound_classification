import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sound_classification_dataset import SoundDS
import audio_utils.audio_metadata as audio_metadata
from sound_classification_model import AudioClassifier, device
from sound_classification_training import training


# ----------------------------
# 10-fold cross validation on the predefined UrbanSound8K folds.
# A fresh model is trained for every fold, so the test fold is never seen during
# training. (Reusing one model across folds - "progressive training" - leaks every
# test fold into training through earlier folds and inflates the accuracy.)
# ----------------------------
def evaluate(model, test_dl):
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
    return accuracy_score(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser(description="10-fold cross validation of AudioClassifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)),
                        help="test folds to run (default: 1..10)")
    parser.add_argument("--max-samples", type=int, default=None, help="random subset of clips (smoke tests)")
    parser.add_argument("--save-dir", default=str(audio_metadata.MODELS_DIR), help="where to save weights")
    parser.add_argument("--no-show", action="store_true", help="save the boxplot without opening a window")
    args = parser.parse_args()

    df = audio_metadata.metadata_file()
    if args.max_samples is not None:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
    data_path = audio_metadata.download_path
    num_epochs = args.epochs
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'cross validation device = {device}')

    fold_accuracies = []
    for test_fold in args.folds:
        print(f"\n--- Fold {test_fold} ---")

        # Split by predefined fold column
        train_df = df[df["fold"] != test_fold]
        test_df = df[df["fold"] == test_fold]

        # DataLoaders (augmentation only on the training data)
        train_dl = DataLoader(SoundDS(train_df, data_path, augment=True), batch_size=16, shuffle=True)
        test_dl = DataLoader(SoundDS(test_df, data_path, augment=False), batch_size=16, shuffle=False)

        # Fresh model for every fold
        model = AudioClassifier().to(device)
        training(model, train_dl, num_epochs)

        acc = evaluate(model, test_dl)
        fold_accuracies.append(acc)
        print(f"Fold {test_fold} accuracy: {acc:.4f}")

        # Weights of the model whose held-out fold is `test_fold`. Evaluate it later with:
        #   python3 sound_inference_from_model.py <file> --fold <test_fold>
        weights_file = os.path.join(args.save_dir, f"cross_val_model_weights_fold{test_fold}_epoch_{num_epochs}.pth")
        torch.save(model.state_dict(), weights_file)
        print("Saved:", weights_file)

    # Summary
    fold_accuracies = np.array(fold_accuracies)
    print(f"\nFold accuracies: {np.round(fold_accuracies, 4).tolist()}")
    print(f"Mean accuracy : {fold_accuracies.mean():.4f}")
    print(f"Std deviation : {fold_accuracies.std():.4f}")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.boxplot(fold_accuracies, tick_labels=[f"{len(fold_accuracies)}-Fold CV"])
    plt.title(f"{len(fold_accuracies)}-Fold Cross-Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.axhline(fold_accuracies.mean(), color="red", linestyle="--", label=f"Mean: {fold_accuracies.mean():.4f}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "cv_boxplot.png")
    plt.savefig(plot_path)
    print("Saved:", plot_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

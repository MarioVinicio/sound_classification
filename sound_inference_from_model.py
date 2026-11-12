import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import audio_utils.audio_metadata as audio_metadata
from sound_classification_dataset import SoundDS
from sound_classification_model import load_model, device
from sound_classification_inference import inference
from sound_classification_split import get_dataloaders, SPLIT_SEED

# Models compared by default
DEFAULT_MODELS = [
    "mymodel_weights_epoch_30.pth",
    "cross_val_model_weights_epoch_15.pth",
    "progressive_training_model_weights_epoch_15.pth",
    "new_train/sc_test/final_model_weights_epoch_15.pth",
    "new_train/sc_test/final_model_weights_epoch_20.pth",
    "new_train/sc_test/final_model_weights_epoch_25.pth",
    "new_train/sc_test/final_model_weights_epoch_30.pth",
    "new_train/sc_test/final_model_weights_epoch_35.pth",
    "new_train/sc_test/final_model_weights_epoch_40.pth",
    "new_train/sc_test/final_model_weights_epoch_45.pth",
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved model weights.",
        epilog="NOTE: a score is only meaningful on data the model never trained on. Models trained "
               f"with sound_classification_training.py (seed={SPLIT_SEED}) should be evaluated on the "
               "default validation split; cross-validation fold models with --fold <their test fold>. "
               "Models trained before the seeded split was introduced (e.g. mymodel_weights_epoch_30) "
               "and the progressive model (trained on all folds) have no held-out data, so their "
               "scores here are optimistic.")
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS,
                        help="weight files (name inside trained_models/ or a path)")
    parser.add_argument("--fold", type=int, default=None,
                        help="evaluate on UrbanSound8K fold N instead of the seeded validation split")
    args = parser.parse_args()

    if args.fold is not None:
        df = audio_metadata.metadata_file()
        test_ds = SoundDS(df[df["fold"] == args.fold], audio_metadata.download_path, augment=False)
        eval_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
        eval_name = f"fold {args.fold}"
    else:
        _, eval_dl = get_dataloaders()
        eval_name = f"validation split (seed={SPLIT_SEED})"

    print(f"device = {device}")
    for name in args.models:
        path = Path(name)
        if not path.is_file():
            path = audio_metadata.MODELS_DIR / name

        print()
        print("=" * 40 + " INFERENCE " + "=" * 40)
        print(f"{path.name}  on  {eval_name}")
        if not path.is_file():
            print(f"Skipped: {path} not found")
            continue

        model = load_model(path)
        inference(model, eval_dl)


if __name__ == "__main__":
    main()

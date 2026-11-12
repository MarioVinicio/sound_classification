import argparse
import torch
import audio_utils.audio_metadata as audio_metadata
from audio_utils.audio_metadata import CLASS_LABELS
from audio_utils.audio_util_load import AudioUtil
from sound_classification_dataset import preprocess
from sound_classification_model import load_model, device

DEFAULT_MODEL = "progressive_training_model_weights_epoch_15.pth"


def predict_single(model, file_path, device):
    # Same preprocessing as SoundDS, without training augmentation
    sgram = preprocess(AudioUtil.open(file_path), augment=False)

    # --- Add batch dimension and normalize ---
    input_tensor = sgram.unsqueeze(0).to(device)        # (1, 2, 64, 344)
    input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def main():
    parser = argparse.ArgumentParser(description="Classify one wav file")
    parser.add_argument("--csv-line", type=int, default=935,
                        help="line number in UrbanSound8K.csv as shown in an editor (line 1 is the header)")
    parser.add_argument("--wav", default=None, help="path to any wav file (overrides --csv-line)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="weights file inside trained_models/")
    args = parser.parse_args()

    class_id = None
    if args.wav:
        file_path = args.wav
    else:
        df = audio_metadata.metadata_file()
        # Editor line N -> DataFrame row N-2 (one line for the header, one because rows start at 0)
        row = df.iloc[args.csv_line - 2]
        file_path = row["absolute_path"].strip()
        class_id = int(row["classID"])

    print(f'wav file = {file_path}')
    print(f'inference single wav device = {device}')
    print()
    print("=" * 40 + " INFERENCE " + "=" * 40)
    print(args.model)
    model = load_model(audio_metadata.MODELS_DIR / args.model)

    if class_id is not None:
        print(f"classID : {class_id} — {CLASS_LABELS[class_id]}")
    pred_idx = predict_single(model, file_path, device)
    print(f"Predicted class: {pred_idx} — {CLASS_LABELS[pred_idx]}")


if __name__ == "__main__":
    main()

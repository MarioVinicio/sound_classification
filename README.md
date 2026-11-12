# Sound Classification Framework

End-to-end urban sound classification pipeline built with PyTorch. Ingests raw `.wav` files from the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset, applies a preprocessing and augmentation pipeline, trains a CNN classifier, and supports inference from saved model weights.

## Classes

| ID | Label |
|----|-------|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

## Model Architecture

`AudioClassifier` — 4-block CNN with BatchNorm and Kaiming initialization, followed by `AdaptiveAvgPool2d` and a fully connected linear layer outputting 10 class logits.

```
Input: (batch, 2, 64, 344)  ← 2-channel Mel Spectrogram
  Conv2d(2→8,  5×5, stride 2) + ReLU + BN
  Conv2d(8→16, 3×3, stride 2) + ReLU + BN
  Conv2d(16→32,3×3, stride 2) + ReLU + BN
  Conv2d(32→64,3×3, stride 2) + ReLU + BN
  AdaptiveAvgPool2d(1×1)
  Linear(64 → 10)
Output: (batch, 10)
```

Supports automatic device selection: Apple Silicon (MPS), CUDA, or CPU.

## Preprocessing Pipeline

Applied on-the-fly per batch via a custom PyTorch `Dataset` + `DataLoader`. The same `preprocess()` function (in `sound_classification_dataset.py`) is used for training, evaluation, single-file and microphone inference:

1. Load `.wav` file via `librosa`
2. Resample to 44,100 Hz
3. Convert mono → stereo (duplicate channel)
4. Pad or truncate to fixed 4-second duration → shape `(2, 176,000)`
5. *(training only)* Time-shift augmentation (random left/right shift)
6. Convert to Mel Spectrogram → shape `(2, 64, 344)`
7. *(training only)* SpecAugment: frequency masking + time masking

During training, short clips are padded at a random offset. For validation and inference the clip is centered, so predictions are deterministic.

## Training

- Optimizer: Adam (`lr=0.001`)
- Loss: CrossEntropyLoss
- Scheduler: OneCycleLR with linear annealing
- Inputs normalized per batch (mean/std)
- Train/validation split: 80/20 with a fixed seed (`SPLIT_SEED = 42`), so a model is always evaluated on clips it never trained on
- 10-fold cross validation on the predefined UrbanSound8K folds, with a fresh model per fold
- Trained weights saved to `trained_models/`

## Project Structure

```
sound_classification/
├── audio_utils/                         # Audio preprocessing utilities, paths, class labels
├── trained_models/                      # Saved model weights (.pth)
├── sound_classification_dataset.py      # Shared preprocess() + PyTorch Dataset
├── sound_classification_split.py        # Seeded train/validation DataLoader split
├── sound_classification_model.py        # AudioClassifier architecture, device, load_model()
├── sound_classification_training.py     # Training loop + model export
├── sound_classification_inference.py    # Accuracy on a DataLoader
├── sound_inference_from_model.py        # Evaluate saved .pth files
├── sc_cross_validation.py               # 10-fold cross validation
├── sc_inference_single_wav.py           # Classify a single wav file
├── sc_capture_env_audio.py              # Live classification from the microphone
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `sounddevice` (microphone capture).

Download the UrbanSound8K dataset into `UrbanSound8K/`: https://urbansounddataset.weebly.com/urbansound8k.html

## Usage

Scripts resolve paths relative to the project, so they can be run from any directory.

```bash
# Train the model (add --max-samples 200 for a quick smoke test, --save-dir to change the output folder)
python3 sound_classification_training.py --epochs 30

# 10-fold cross validation (--folds 1 2 to run a subset, --no-show to skip the plot window)
python3 sc_cross_validation.py --epochs 15

# Evaluate saved weights on the seeded validation split...
python3 sound_inference_from_model.py mymodel_weights_epoch_30.pth
# ...or a cross-validation fold model on its held-out fold
python3 sound_inference_from_model.py cross_val_model_weights_fold1_epoch_15.pth --fold 1

# Classify a single file (by UrbanSound8K.csv line number, or any wav)
python3 sc_inference_single_wav.py --csv-line 935
python3 sc_inference_single_wav.py --wav path/to/file.wav

# Live classification from the microphone (Ctrl+C to stop)
python3 sc_capture_env_audio.py
```

## Reference

Based on the tutorial series by Ketan Doshi:
[Audio Deep Learning Made Simple — Towards Data Science](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5/)
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

Applied on-the-fly per batch via a custom PyTorch `Dataset` + `DataLoader`:

1. Load `.wav` file via `torchaudio`
2. Convert mono → stereo (duplicate channel)
3. Resample to 44,100 Hz
4. Pad or truncate to fixed 4-second duration → shape `(2, 176,400)`
5. Time-shift augmentation (random left/right shift)
6. Convert to Mel Spectrogram → shape `(2, 64, 344)`
7. SpecAugment: frequency masking + time masking

## Training

- Optimizer: Adam (`lr=0.001`)
- Loss: CrossEntropyLoss
- Scheduler: OneCycleLR with linear annealing
- Inputs normalized per batch (mean/std)
- Train/validation split: 80/20
- Trained weights saved to `trained_models/`

## Project Structure

```
sound_classification/
├── audio_utils/                         # Audio preprocessing utilities
├── trained_models/                      # Saved model weights (.pth)
├── sound_classification_dataset.py      # Custom PyTorch Dataset with transforms
├── sound_classification_split.py        # Train/validation DataLoader split
├── sound_classification_model.py        # AudioClassifier architecture
├── sound_classification_training.py     # Training loop + model export
├── sound_classification_inference.py    # Inference using DataLoader
├── sound_inference_from_model.py        # Inference from saved .pth file
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

Download the UrbanSound8K dataset: https://urbansounddataset.weebly.com/urbansound8k.html

## Usage

```bash
# Train the model
python3 sound_classification_training.py

# Run inference with a DataLoader
python3 sound_classification_inference.py

# Run inference from a saved model file
python3 sound_inference_from_model.py
```

## Reference

Based on the tutorial series by Ketan Doshi:
[Audio Deep Learning Made Simple — Towards Data Science](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5/)
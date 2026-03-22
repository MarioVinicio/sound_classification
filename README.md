# Sound Classification Framework

## Description
This Sound Classification Framework processess audio samples from UrbanSound8K dataset and trains a Deep Learning Convolutional Neural Network. The model produced by the CNN can classify urban sounds from the following classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.
## Technical Overview
End-to-end audio sound classification pipeline using deep learning in Python and PyTorch.
Ingests raw .wav files from the UrbanSound8K dataset (10 classes, ~8K samples) and applies a preprocessing pipeline: mono-to-stereo conversion, sample rate standardization to 44.1kHz, fixed-length padding/truncation to 4s, and time-shift augmentation.
Audio is transformed into Mel Spectrograms (shape: 2×64×344) and further augmented via SpecAugment (frequency and time masking).
A custom PyTorch Dataset and DataLoader handle batched, on-the-fly preprocessing during training.
The classifier uses a 4-block CNN to extract feature maps, followed by global average pooling and a fully connected linear layer outputting logits for 10 classes.
Training uses an adaptive learning rate scheduler to improve convergence. Achieves baseline accuracy suitable for benchmarking urban sound recognition tasks.

Source: https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504/

## Requirements
- Python 3.x
- PyTorch
- Download 'UrbanSound8K' dataset folder from - https://urbansounddataset.weebly.com/urbansound8k.html

## Usage
```python
# How to train and use the model
python3 sound_classification_training.py
python3 sound_inference_from_model.py
```

## Model
- Architecture: ...
- Training epochs: 30
- Accuracy: ...
import torch
import audio_utils.audio_metadata as audio_metadata
from audio_utils.audio_metadata import CLASS_LABELS
from sound_classification_dataset import preprocess as preprocess_audio, DURATION_MS
from sound_classification_model import load_model, device

# --- Config ---
SAMPLE_RATE   = 44100
DURATION_SECS = DURATION_MS / 1000   # must match training (4s)
CHANNELS      = 1
MODEL_PATH    = audio_metadata.MODELS_DIR / "progressive_training_model_weights_epoch_15.pth"


# --- Record from mic ---
def record(duration=DURATION_SECS, sample_rate=SAMPLE_RATE):
    import sounddevice as sd   # imported here so preprocess/predict work without a mic library
    print(f"🎙  Recording for {duration:g} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()  # block until recording is done
    print("✅ Recording complete.")
    return torch.tensor(audio.T)   # shape: (1, num_samples)


# --- Preprocess (same pipeline as SoundDS, without training augmentation) ---
def preprocess(waveform, sample_rate):
    return preprocess_audio((waveform, sample_rate), augment=False)


# --- Predict ---
def predict(model, sgram):
    input_tensor = sgram.unsqueeze(0).to(device)              # (1, 2, 64, 344)
    input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()

    with torch.no_grad():
        output  = model(input_tensor)
        probs   = torch.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

    label      = CLASS_LABELS[predicted.item()]
    confidence = conf.item() * 100
    return label, confidence


# --- Main loop ---
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    print(f"Model: {MODEL_PATH.name} on {device}")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            print("Starting new record...")
            waveform = record()
            sgram    = preprocess(waveform, SAMPLE_RATE)
            label, confidence = predict(model, sgram)
            print(f"🔊 Predicted: {label} ({confidence:.1f}% confidence)\n")
    except KeyboardInterrupt:
        print("\nStopped.")

import torch
import os
import sounddevice as sd
from sound_classification_model import AudioClassifier
from audio_utils.audio_util_load import AudioUtil

# --- Config ---
SAMPLE_RATE   = 44100
DURATION_SECS = 4          # must match your training (4s)
CHANNELS      = 1
SAVE_DIR      = os.getcwd() + "/sound_classification/trained_models"
MODEL_PATH    = f"{SAVE_DIR}/progressive_training_model_weights_epoch_15.pth"

CLASS_LABELS = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load model ---
model = AudioClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- Record from mic ---
def record(duration=DURATION_SECS, sample_rate=SAMPLE_RATE):
    print(f"🎙  Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()  # block until recording is done
    print("✅ Recording complete.")
    return torch.tensor(audio.T)   # shape: (1, num_samples)

# --- Preprocess (mirror your SoundDS pipeline) ---
def preprocess(waveform, sample_rate):
    aud = (waveform, sample_rate)
    aud = AudioUtil.rechannel(aud, 2)           # mono → stereo
    aud = AudioUtil.resample(aud, 44100)        # standardize sample rate
    aud = AudioUtil.pad_trunc(aud, 4000)        # pad/truncate to 4s
    # skip time_shift and spectro_augment — inference only
    sgram = AudioUtil.spectro_gram_tensor(aud, n_mels=64, n_fft=1024, hop_len=None)
    return sgram

# --- Predict ---
def predict(sgram):
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
    print("Press Ctrl+C to stop.\n")
    while True:
        # input("Press Enter to record...")
        print("Starting new record...")
        waveform = record()
        sgram    = preprocess(waveform, SAMPLE_RATE)
        label, confidence = predict(sgram)
        print(f"🔊 Predicted: {label} ({confidence:.1f}% confidence)\n")
import os
import torch
import audio_utils.audio_metadata as audio_metadata
from sound_classification_model import AudioClassifier
from audio_utils.audio_util_load import AudioUtil

def predict_single(model, file_path, device):
    # --- Preprocessing (mirror your SoundDS pipeline) ---
    aud     = AudioUtil.open(file_path)
    aud     = AudioUtil.rechannel(aud, 2)
    aud     = AudioUtil.resample(aud, 44100)
    aud     = AudioUtil.pad_trunc(aud, 4000)
    # aud     = AudioUtil.time_shift(aud, 0.4) - training argumentation
    sgram   = AudioUtil.spectro_gram_tensor(aud, n_mels=64, n_fft=1024, hop_len=None)
    # sgram   = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2) - training argumentation

    # --- Add batch dimension and normalize ---
    input_tensor = sgram.unsqueeze(0).to(device)        # (1, 2, 64, 344)
    input_tensor = (input_tensor - input_tensor.mean()) / input_tensor.std()

    # --- Inference ---
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Class labels (UrbanSound8K order)
class_labels = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]
# UrbanSound8K.csv wav file index
wav_csv_index = 935

# --- Run it ---
df = audio_metadata.metadata_file()
file_path = df.iloc[wav_csv_index-2]["absolute_path"].strip()
print(f'wav file = {file_path}')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'inference single wav device = {device}')
myModel = AudioClassifier().to(device)

# Load model for inference
save_dir = os.getcwd() + "/sound_classification/trained_models"
print()
print("==================================================INFERENCE=======================================")
print("progressive_training_model_weights_epoch_15.pth")
myModel.load_state_dict(torch.load(os.path.join(save_dir, 'progressive_training_model_weights_epoch_15.pth'), weights_only=True))

classID = df.iloc[wav_csv_index-2]["classID"]
print(f"classID : {classID}")
pred_idx  = predict_single(myModel, file_path, device)
print(f"Predicted class: {pred_idx} — {class_labels[pred_idx]}")


from torch.utils.data import Dataset
from audio_utils.audio_util_load import AudioUtil

# ----------------------------
# Preprocessing settings shared by training and inference
# ----------------------------
SAMPLE_RATE = 44100
CHANNELS = 2
DURATION_MS = 4000
SHIFT_PCT = 0.4


# ----------------------------
# Turn a loaded (signal, sample_rate) tuple into a mel spectrogram.
# Used by SoundDS and by the inference scripts so all of them run the same pipeline.
# augment=True  -> training: random padding offset, time shift, SpecAugment
# augment=False -> evaluation/inference: deterministic (centered padding, no masking)
# ----------------------------
def preprocess(aud, augment=False):
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    aud = AudioUtil.resample(aud, SAMPLE_RATE)
    aud = AudioUtil.rechannel(aud, CHANNELS)
    aud = AudioUtil.pad_trunc(aud, DURATION_MS, random_pad=augment)
    if augment:
        aud = AudioUtil.time_shift(aud, SHIFT_PCT)
    sgram = AudioUtil.spectro_gram_tensor(aud, n_mels=64, n_fft=1024, hop_len=None)
    if augment:
        sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    return sgram


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path, augment=True):
        # reset_index so __getitem__ can use positional indices 0..len-1
        self.df = df.reset_index(drop=True)
        self.data_path = str(data_path)
        self.augment = augment

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + '/audio' + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        sgram = preprocess(aud, augment=self.augment)

        return sgram, class_id

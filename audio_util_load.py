import math, random
import torch
# import torchaudio
from torchaudio import transforms
from IPython.display import Audio

import audio_metadata

import sklearn
import sklearn.preprocessing
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        # sig, sr = torchaudio.load(audio_file)
        # return (sig, sr)
        # ===== Using librosa =========================
        sig, sr = librosa.load(audio_file, sr=None)
        # Convert to torch tensor and add channel dimension if needed
        sig = torch.from_numpy(sig)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)  # Add channel dimension
        return (sig, sr)

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            # resig = sig[:1, :]
            resig = sig[:1]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return (resig, sr)
    
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # ----------------------------
    # Generate a Spectrogram 1
    # Uses torchaudio library
    # ----------------------------
    @staticmethod
    def spectro_gram_tensor(aud, n_mels=64, n_fft=1024, hop_len=None):
        """
        Uses Tensor as input format 
        aud : class 'torch.Tensor'
        """
        sig, sr = aud
        # sig, sr = AudioUtil.rechannel(aud, 1)
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # ----------------------------
    # Generate a Spectrogram 2
    # Uses librosa library
    # ----------------------------
    @staticmethod
    def spectro_gram_numpy(audio_file_numpy):
        """
        Uses numpy array as input format 
        audio_file_numpy : class 'numpy.ndarray'
        """

        # Fix: Make your audio 1-D before the STFT: If your audio file is stereo (2 channels), convert to mono:
        audio_mono = librosa.to_mono(audio_file_numpy)
        # Compute the short-time Fourier transform
        sgram = librosa.stft(audio_mono)
        # Convert the complex-valued STFT to magnitude spectrogram
        sgram_mag = np.abs(sgram)
        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram_mag)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=44100)

        # amplitude to dB
        amp_to_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        
        return (amp_to_db)
    

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        # Convert numpy array to torch tensor if needed
        if isinstance(spec, np.ndarray):
            spec = torch.from_numpy(spec)
        
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = int(max_mask_pct * n_mels)
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = int(max_mask_pct * n_steps)
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


    # ----------------------------
    # Tensor to Numpy
    # ----------------------------
    @staticmethod
    def tensor_to_numpy(tensor):
        # import numpy as np
        # Check if the tensor is on GPU
        if tensor.is_cuda:
            # Move tensor to CPU if it's on GPU
            tensor = tensor.cpu()
        # Convert the tensor to a NumPy array
        numpy_array = tensor.numpy()

        return numpy_array

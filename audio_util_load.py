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
        # print(f" sig.shape = {sig.shape}")
        max_len = sr//1000 * max_ms
        # print(f" max_len = {max_len}")

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
        # print(f'amplitude to db = {amp_to_db}')
        # print(f'type(amp_to_db) = {type(amp_to_db)}, amp_to_db.shape = {amp_to_db.shape}')
        # type(amp_to_db) = <class 'numpy.ndarray'>, amp_to_db.shape = (128, 258)
        
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



# This code only runs when the file is executed directly, not when imported
if __name__ == "__main__":
    # Test code, examples, debugging code, etc.

    # ====== START HERE ===========================================================================================================

    audio_pd = audio_metadata.metadata_file()['absolute_path'][1670] #street_music (voice) - 1670
    # audio_pd = audio_metadata.metadata_file()['absolute_path'][1677] #engine_idling - 1677
    # audio_pd = audio_metadata.metadata_file()['absolute_path'][2244] #street_music (no voice) - 2244
    print(audio_pd)
    audio_file = AudioUtil.open(audio_pd)
    print(audio_file)
    print(len(audio_file[0][0]))
    print(f"Shape of the Signal Tensor: {audio_file[0].shape}")
    print("========================================================================================================================")

    audio_file = AudioUtil.rechannel(audio_file, 2)
    # audio_file = AudioUtil.rechannel(audio_file, 1)
    print(audio_file)
    print(audio_file[0].shape)
    print("========================================================================================================================")


    audio_file_resample = AudioUtil.resample(audio_file, 44100)
    # audio_file_resample = AudioUtil.resample(audio_file, 48000)
    print(audio_file_resample)
    print(audio_file_resample[0].shape)
    print("========================================================================================================================")

    audio_file_pad = AudioUtil.pad_trunc(audio_file_resample, 3000)
    print(audio_file_pad)
    print(audio_file_pad[0].shape)
    print("========================================================================================================================")

    audio_file_time_shift = AudioUtil.time_shift(audio_file_pad, 0)
    print(audio_file_time_shift)
    print(audio_file_time_shift[0].shape)

    print("========================================================================================================================")

    # Convert Tensor to Numpy

    print(f'audio_file_time_shift = {audio_file_time_shift}')
    print(f'type(audio_file_time_shift[0]) = {type(audio_file_time_shift[0])}, audio_file_time_shift[0].shape = {audio_file_time_shift[0].shape}')
    audio_file_numpy = AudioUtil.tensor_to_numpy(audio_file_time_shift[0])
    print(f'audio_file_numpy = {audio_file_numpy}')
    print(f'type(audio_file_numpy) = {type(audio_file_numpy)}, audio_file_numpy.shape = {audio_file_numpy.shape}')

    print("==================================================== Spectrogram 1 ====================================================================")

    spectro = AudioUtil.spectro_gram_tensor(audio_file_time_shift)
    print(f'spectro = {spectro}')
    print(f'type(spectro) = {type(spectro)}, spectro.shape = {spectro.shape}')
    # convert this format:
    # type(spectro) = <class 'torch.Tensor'>, spectro.shape = torch.Size([2, 64, 258])
    # to this format: 
    # type(amp_to_db) = <class 'numpy.ndarray'>, amp_to_db.shape = (128, 258)

    numpy_spectro = AudioUtil.tensor_to_numpy(spectro)
    # print(f'numpy_spectro = {numpy_spectro}')
    # print(f'type(numpy_spectro) = {type(numpy_spectro)}, numpy_spectro.shape = {numpy_spectro.shape}')
    # # type(numpy_spectro) = <class 'numpy.ndarray'>, numpy_spectro.shape = (2, 64, 258)

    # print(f'numpy_spectro[0][0] = {numpy_spectro[0][0]}')
    # print(f'len(numpy_spectro[0][0]) = {len(numpy_spectro[0][0])}')

    # print(f'numpy_spectro[0] = {numpy_spectro[0]}')
    # print(f'len(numpy_spectro[0]) = {len(numpy_spectro[0])}')

    # numpy_spectro_format = np.append(numpy_spectro[0], numpy_spectro[1], axis=0)
    # print(f'len(numpy_spectro_format) = {len(numpy_spectro_format)}')
    # print(f'numpy_spectro_format.shape = {numpy_spectro_format.shape}')

    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(numpy_spectro[0], sr=44100, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Tensor')
    plt.tight_layout()
    plt.show()
    plt.close()

    print("================================ play audio ========================================================================================")


    # play audio
    # Audio(audio_pd)
    # from playsound import playsound
    # playsound(audio_pd)

    import sounddevice as sd
    from scipy.io import wavfile

    rate, data = wavfile.read(audio_pd)
    # print(f'size of wav : {len(data)}')

    import os

    # file_path = "your_audio.wav"
    size_bytes = os.path.getsize(audio_pd)
    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 ** 2)

    print("Size (bytes):", size_bytes)
    print("Size (KB):", size_kb)
    print("Size (MB):", size_mb)

    sd.play(data, rate)
    sd.wait()     # Wait until playback finishes

    print("============================== print waveform =========================================================================================")

    # Print wav

    # x-axis has been converted to time using our sample rate. 
    # matplotlib plt.plot(y), would output the same figure, but with sample 
    # number on the x-axis instead of seconds

    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio_file_numpy, sr=44100)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


    print("================== Spectrogram 2 ======================================================================================================")

    # Spectrogram

    amp_to_db = AudioUtil.spectro_gram_numpy(audio_file_numpy)
    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(amp_to_db, sr=44100, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Numpy')
    plt.tight_layout()
    plt.show()
    plt.close()


    print("========================================================================================================================")

    #Spectrogram is a 2D numpy array
    print(type(amp_to_db), amp_to_db.shape)
    # <class 'numpy.ndarray'> (128, 258)

    print("========== (Mel Frequency Cepstral Coefficients) ======================================================================================")

    # Load the audio file
    samples, sample_rate = librosa.load(audio_pd, sr=None)
    # samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)
    print(f' - samples : {samples}')
    print(f' - sample_rate : {sample_rate}')
    print(f' - librosa.feature.mfcc : {librosa.feature.mfcc}')
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate)
    print("HERE ========================================================================================================================")
    # Center MFCC coefficient dimensions to the mean and unit variance
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    print("HERE 2 ========================================================================================================================")
    # Display 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC (for Human Speech)')
    plt.tight_layout()
    plt.show()
    plt.close()

    print (f'MFCC is of type {type(mfcc)} with shape {mfcc.shape}')
    # MFCC is of type <class 'numpy.ndarray'> with shape (20, 134)



    print("========== Data Augmentation: Time and Frequency Masking ===============================================================================")

    spectro_masking = AudioUtil.spectro_augment(spectro,0.1,2,1)

    # convert to numpy to print with specshow()
    numpy_spectro_masking = AudioUtil.tensor_to_numpy(spectro_masking)
    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(numpy_spectro_masking[0], sr=44100, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Time and Frequency Masking')
    plt.tight_layout()
    plt.show()
    plt.close()
import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal

class AudioProcessor:
    def __init__(self, sample_dir, processed_dir, visualized_dir, compared_dir):
        self.sample_dir = sample_dir
        self.processed_dir = processed_dir
        self.visualized_dir = visualized_dir
        self.compared_dir = compared_dir
    
    def load_audio(self, file_name, directory=None):
        if directory is None:
            directory = self.sample_dir
        file_path = os.path.join(directory, file_name)
        audio, sr = librosa.load(file_path)
        return audio, sr
    
    # white noise
    # def add_noise(self, audio, snr_decibels):
    #     signal_power = np.mean(audio ** 2)
    #     noise_power = signal_power / (10 ** (snr_decibels / 10))
    #     noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    #     noisy_audio = audio + noise
    #     return noisy_audio
        
    # pink noise
    def add_noise(self, audio, snr_decibels):
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_decibels / 10))

        # Generate Pink Noise
        length = len(audio)
        pink_noise = np.zeros(length)

        # Generate White Gaussian Noise
        white_noise = np.random.normal(0, 1, length)

        # Apply 1/f filter to the white noise to obtain Pink Noise
        b = np.array([1, -0.97])
        a = np.array([1, -0.97])
        pink_noise = signal.lfilter(b, a, white_noise)

        # Scale the Pink Noise to the desired noise power
        pink_noise *= np.sqrt(noise_power / np.mean(pink_noise ** 2))

        # Add Pink Noise to the original audio signal
        noisy_audio = audio + pink_noise

        return noisy_audio
    # def add_noise(self, audio, snr_decibels, preemphasis_factor=0.97):
    #     signal_power = np.mean(audio ** 2)
    #     noise_power = signal_power / (10 ** (snr_decibels / 10))

    #     # Apply pre-emphasis filter to the original audio signal
    #     preemphasis_filter = np.array([1, -preemphasis_factor])
    #     preemphasized_audio = signal.lfilter([1], preemphasis_filter, audio)

    #     # Generate Pink Noise
    #     length = len(preemphasized_audio)
    #     pink_noise = np.zeros(length)

    #     # Generate White Gaussian Noise
    #     white_noise = np.random.normal(0, 1, length)

    #     # Apply 1/f filter to the white noise to obtain Pink Noise
    #     b = np.array([1, -0.97])
    #     a = np.array([1, -0.97])
    #     pink_noise = signal.lfilter(b, a, white_noise)

    #     # Scale the Pink Noise to the desired noise power
    #     pink_noise *= np.sqrt(noise_power / np.mean(pink_noise ** 2))

    #     # Add Pink Noise to the pre-emphasized audio signal
    #     noisy_audio = preemphasized_audio + pink_noise

    #     return noisy_audio
        
    def save_audio(self, audio, sr, file_name):
        file_path = os.path.join(self.processed_dir, file_name)
        sf.write(file_path, audio, sr)
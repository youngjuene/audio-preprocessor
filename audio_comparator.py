import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AudioComparator:
    def __init__(self, compared_dir, fig_size=(12, 8)):
        self.compared_dir = compared_dir
        self.fig_size = fig_size

    def compare_audio(self, audio_files, sample_number):
        num_audios = len(audio_files)
        fig = plt.figure(figsize=self.fig_size)
        grid = plt.GridSpec(num_audios + 1, 4, wspace=0.4, hspace=0.4)
        axes = []
        for i in range(num_audios + 1):
            axes.append([])
            for j in range(4):
                axes[i].append(fig.add_subplot(grid[i, j], projection='3d' if j == 3 else None))

        fig.suptitle(f'Audio Comparison - Sample {sample_number}', y=0.95)

        for i, (audio, sr, file_name) in enumerate(audio_files):
            # Plot waveform
            axes[i][0].plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
            axes[i][0].set_title(f'Waveform - {file_name}')
            axes[i][0].set_xlabel('Time (s)')
            axes[i][0].set_ylabel('Amplitude')

            # Plot spectrogram
            spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz', ax=axes[i][1])
            axes[i][1].set_title(f'Spectrogram - {file_name}')
            axes[i][1].set_xlabel('Time')
            axes[i][1].set_ylabel('Frequency (Hz)')

            # Plot mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', ax=axes[i][2])
            axes[i][2].set_title(f'Mel Spectrogram - {file_name}')
            axes[i][2].set_xlabel('Time')
            axes[i][2].set_ylabel('Mel Frequency')

            # Plot 3D time-frequency analysis
            times = librosa.times_like(spectrogram)
            frequencies = librosa.fft_frequencies(sr=sr)
            axes[i][3].plot_surface(times, frequencies, spectrogram.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            axes[i][3].set_title(f'Time-Frequency Analysis - {file_name}')
            axes[i][3].set_xlabel('Time (s)')
            axes[i][3].set_ylabel('Frequency (Hz)')
            axes[i][3].set_zlabel('Amplitude (dB)')
            axes[i][3].view_init(elev=30, azim=-45)

        # Plot the original sample file
        original_file = [file for file in audio_files if sample_number in file][0]
        original_audio, original_sr = librosa.load(os.path.join('sample', original_file[2]))
        original_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
        times = librosa.times_like(original_spectrogram)
        frequencies = librosa.fft_frequencies(sr=original_sr)

        axes[-1][0].plot(np.linspace(0, len(original_audio) / original_sr, len(original_audio)), original_audio)
        axes[-1][0].set_title(f'Waveform - Original')
        axes[-1][0].set_xlabel('Time (s)')
        axes[-1][0].set_ylabel('Amplitude')

        librosa.display.specshow(original_spectrogram, sr=original_sr, x_axis='time', y_axis='hz', ax=axes[-1][1])
        axes[-1][1].set_title('Spectrogram - Original')
        axes[-1][1].set_xlabel('Time')
        axes[-1][1].set_ylabel('Frequency (Hz)')

        original_mel_spectrogram = librosa.feature.melspectrogram(y=original_audio, sr=original_sr)
        original_mel_spectrogram_db = librosa.power_to_db(original_mel_spectrogram, ref=np.max)
        librosa.display.specshow(original_mel_spectrogram_db, x_axis='time', y_axis='mel', ax=axes[-1][2])
        axes[-1][2].set_title('Mel Spectrogram - Original')
        axes[-1][2].set_xlabel('Time')
        axes[-1][2].set_ylabel('Mel Frequency')

        axes[-1][3].plot_surface(times, frequencies, original_spectrogram.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        axes[-1][3].set_title('Time-Frequency Analysis - Original')
        axes[-1][3].set_xlabel('Time (s)')
        axes[-1][3].set_ylabel('Frequency (Hz)')
        axes[-1][3].set_zlabel('Amplitude (dB)')
        axes[-1][3].view_init(elev=30, azim=-45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.compared_dir, f'comparison_sample_{sample_number}.png'))
        plt.close()
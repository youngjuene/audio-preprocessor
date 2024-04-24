import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class AudioVisualizer:
    def __init__(self, visualized_dir, fig_size=(28, 24)):
        self.visualized_dir = visualized_dir
        self.fig_size = fig_size

    def visualize_audio(self, audios, srs, file_name):
        fig = plt.figure(figsize=self.fig_size)
        grid = plt.GridSpec(4, 3, wspace=0.2, hspace=0.3)
        ax_waveform = [fig.add_subplot(grid[i, 0]) for i in range(4)]
        ax_spectrogram = [fig.add_subplot(grid[i, 1]) for i in range(4)]
        ax_mel_spectrogram = [fig.add_subplot(grid[i, 2]) for i in range(4)]

        for i, (audio, sr) in enumerate(zip(audios, srs)):
            # Plot waveform
            librosa.display.waveshow(audio, sr=sr, ax=ax_waveform[i])
            ax_waveform[i].set_title(f'Waveform ({i})')
            ax_waveform[i].set_xlabel('Time')
            ax_waveform[i].set_ylabel('Amplitude')

            # Plot spectrogram
            n_fft = 1024
            hop_length = 512
            win_length = 1024
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz', ax=ax_spectrogram[i])
            ax_spectrogram[i].set_title(f'Spectrogram ({i})')
            ax_spectrogram[i].set_xlabel('Time')
            ax_spectrogram[i].set_ylabel('Frequency')

            # Plot mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', ax=ax_mel_spectrogram[i])
            ax_mel_spectrogram[i].set_title(f'Mel Spectrogram ({i})')
            ax_mel_spectrogram[i].set_xlabel('Time')
            ax_mel_spectrogram[i].set_ylabel('Mel Frequency')


        plt.savefig(os.path.join(self.visualized_dir, f'{file_name}.png'), bbox_inches='tight', dpi=150)
        plt.close()
        

def visualize_time_frequency_3d(audio_file, sr, ax, output_dir='visualized_plots_3d', n_fft=1024, hop_length=512):
    audio, _ = librosa.load(audio_file, sr=sr)

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[:spectrogram.shape[0]]

    # Ensure shape compatibility
    min_len = min(frequencies.shape[0], times.shape[0])
    spectrogram = spectrogram[:min_len, :min_len]

    X, Y = np.meshgrid(times[:min_len], frequencies[:min_len])
    ax.plot_surface(X, Y, spectrogram, rstride=1, cstride=1, cmap='RdYlBu', edgecolor='none')
    ax.set_title('Time-Frequency Analysis', pad=20)
    ax.set_xlabel('Time (s)', labelpad=15)
    ax.set_ylabel('Frequency (Hz)', labelpad=15)
    ax.set_zlabel('Amplitude (dB)', labelpad=15)

    # Adjust viewing angle and set dynamic z-axis limits
    ax.view_init(elev=40, azim=-60)
    z_min = np.floor(np.min(spectrogram))
    z_max = np.ceil(np.max(spectrogram))
    ax.set_zlim(z_min, z_max)

    plt.tight_layout()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot with the same name as the audio file but with a ".png" extension
    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_file))[0] + ".png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
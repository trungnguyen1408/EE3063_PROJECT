import numpy as np
import librosa
import librosa.display
import soundfile as sf
from IPython.display import Audio
import matplotlib.pyplot as plt

def load_audio_file(file_path):
  data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
  return data, sample_rate

def plot_time_series(data):
  fig = plt.figure(figsize=(14, 8))
  plt.title('Raw wave')
  plt.ylabel('Amplitude')
  plt.plot(np.linspace(0, 1, len(data)), data)
  plt.show()

# Get name of audio file
def file_path(s, file):
  return file + str(s) + '.wav'

# Data augmentation
# Noise Injection
def noise(save_path, data, sample_rate, noise_factor=0.01):
  # save_path: the path to the save folder

  noise = np.random.randn(len(data))
  augmented_data = data + noise_factor * noise

    # Cast back to same data type
  augmented_data = augmented_data.astype(type(data[0]))
    #plot_time_series(augmented_data)
  sf.write(str(save_path) + 'noise.wav', augmented_data, sample_rate)
  return augmented_data

# Shifting the sound
def shift_sound(save_path, data, sample_rate, shift_max=0.8):
  # save_path: the path to the save folder

    shift = np.random.randint(sample_rate * shift_max)
    direction = np.random.randint(0, 2)

    if direction == 1:
      shift = - shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
      augmented_data[:shift] = 0
    else:
      augmented_data[shift:] = 0

    #plot_time_series(augmented_data)
    sf.write(str(save_path) + 'shift.wav', augmented_data, sample_rate)
    return augmented_data

# Change pitch
def change_pitch(save_path, data, sample_rate, pitch_factor=5):
  # save_path: the path to the save folder

    augmented_data = librosa.effects.pitch_shift(data, sample_rate, pitch_factor)
    #plot_time_series(augmented_data)

    sf.write(str(save_path) + 'pitch.wav', augmented_data, sample_rate)
    return augmented_data
import librosa
import numpy as np

# Load audio file
audio, sr = librosa.load(r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\1.wav")

# Define the target peak level in decibels (0 dB is common)
target_peak_dB = 0.0

# Find the maximum amplitude
max_amplitude = np.max(np.abs(audio))

# Calculate the scaling factor
scaling_factor = 10 ** (-(max_amplitude - target_peak_dB) / 20)

# Normalize the audio
normalized_audio = audio * scaling_factor

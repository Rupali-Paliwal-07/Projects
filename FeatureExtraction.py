import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_file = r"C:\Users\rupali\Desktop\BE Project\Dataset\Acridotheres fuscus\resample_1.wav"
audio, sr = librosa.load(audio_file)

# Compute the spectrogram
spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

# Convert to decibels
db_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(db_spectrogram, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.title('Spectrogram')
plt.show()

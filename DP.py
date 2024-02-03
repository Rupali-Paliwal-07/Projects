import librosa

# Define the path to your .wav file
file_path = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\1.wav"

# Load the audio data
audio_data, sample_rate = librosa.load(file_path, sr=None)  # Set sr=None to use the native sample rate

# Print the shape and sample rate
print("Audio Data Shape:", audio_data.shape)
print("Sample Rate:", sample_rate)

import librosa
import os
import noisereduce as nr
import soundfile as sf

# Load audio file
input_audio_path = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\2.wav"
audio, sr = librosa.load(input_audio_path)

# Ask the user for the folder path to save the denoised audio
output_folder = input("Enter the folder path to save the denoised audio: ")

# Perform noise reduction
reduced_audio = nr.reduce_noise(y=audio, sr=sr)

# Save the denoised audio in the specified folder
output_audio_path = os.path.join(output_folder, 'denoised_audio.wav')
sf.write(output_audio_path, reduced_audio, sr)

# Print a completion message
print(f"Noise reduction completed. Denoised audio saved as '{output_audio_path}'.")

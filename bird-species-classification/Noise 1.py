import os
import librosa
import noisereduce as nr
import soundfile as sf

# Ask the user for the folder path containing the .wav files
folder_path = input("Enter the folder path containing .wav files: ")

# Perform noise reduction for each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        # Load the audio
        audio, sr = librosa.load(os.path.join(folder_path, filename))

        # Perform noise reduction
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)

        # Save the denoised audio with a new filename
        new_filename = f"denoised_audio_{filename.replace('.wav', '')}.wav"
        sf.write(os.path.join(folder_path, new_filename), reduced_audio, sr)

        # Delete the original .wav file
        os.remove(os.path.join(folder_path, filename))

# Print a completion message
print("Noise reduction completed for all .wav files in the folder.")

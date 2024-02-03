import librosa
import soundfile as sf
import os

# List of input audio file paths
input_files = [
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\6.wav",
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\1.wav",
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\2.wav",
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\3.wav",
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\4.wav",
    r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\5.wav",
]

# Desired duration in seconds
desired_duration = 30

# Output folder for trimmed audio files
output_folder = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\Trimmed to 30 sec"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for input_file in input_files:
    # Load the audio file
    audio, sample_rate = librosa.load(input_file, sr=None)

    # Trim or extend the audio to the desired duration
    if len(audio) < sample_rate * desired_duration:
        # If audio is shorter, extend with silence
        shortage = sample_rate * desired_duration - len(audio)
        padding = [0] * (shortage // 2)
        audio = padding + list(audio) + padding
    else:
        # If audio is longer, trim to the desired duration
        start = (len(audio) - sample_rate * desired_duration) // 2
        end = start + sample_rate * desired_duration
        audio = audio[start:end]

    # Create an output file path
    output_file = os.path.join(output_folder, os.path.basename(input_file))

    # Write the trimmed or extended audio to the output file
    sf.write(output_file, audio, sample_rate)

    print(f"Audio trimmed/extended and saved as {output_file}")

print("All audio files processed.")

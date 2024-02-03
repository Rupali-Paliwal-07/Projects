import librosa
import soundfile as sf
import os

# Folder containing input audio files
input_folder = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Aethopyga gouldiae"

# Desired duration in seconds
desired_duration = 30

# Create a "Trimmed" folder within the input folder if it doesn't exist
output_folder = os.path.join(input_folder, "Trimmed")
os.makedirs(output_folder, exist_ok=True)

# List all .wav files in the input folder
input_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

for input_file in input_files:
    # Get the full path of the input file
    input_file_path = os.path.join(input_folder, input_file)

    # Load the audio file
    audio, sample_rate = librosa.load(input_file_path, sr=None)

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

    # Create an output file path within the "Trimmed" folder
    output_file = os.path.join(output_folder, input_file)

    # Write the trimmed or extended audio to the output file
    sf.write(output_file, audio, sample_rate)

    print(f"Audio trimmed/extended and saved as {output_file}")

print("All audio files processed and saved in the 'Trimmed' folder.")

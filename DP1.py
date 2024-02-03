import librosa
import soundfile as sf

# Load the audio file
input_file = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\1.wav"
output_file = r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Acridotheres fuscus\8.wav"
desired_duration = 30  # Desired duration in seconds

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

# Write the trimmed or extended audio to a new file
sf.write(output_file, audio, sample_rate)

print(f"Audio trimmed/extended to {desired_duration} seconds and saved as {output_file}")

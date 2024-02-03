import librosa
import soundfile as sf
import os

def trim_concentrated_audio(input_folder, output_folder):
    # Process all files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.mp3'):  # Process only .mp3 files, modify extension if needed
                input_file = os.path.join(root, file)

                # Load audio file and extract the most concentrated frequency
                audio, sample_rate = librosa.load(input_file, sr=None)
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
                mean_magnitudes = magnitudes.mean(axis=1)
                most_concentrated_pitch_index = mean_magnitudes.argmax()
                most_concentrated_pitch = pitches[most_concentrated_pitch_index, 0]

                # Calculate time index for 10 seconds duration
                desired_duration = 10  # Desired duration in seconds
                time_index = int(desired_duration * sample_rate)

                # Find the start and end index around the most concentrated frequency for 10 seconds
                start_index = max(0, int(most_concentrated_pitch) - time_index // 2)
                end_index = min(len(audio), start_index + time_index)

                # If the identified segment is shorter than 10 seconds, adjust the start and end indexes
                if end_index - start_index < time_index:
                    start_index = max(0, end_index - time_index)

                # Trim audio to the identified segment (10 seconds)
                trimmed_audio = audio[start_index:end_index]

                # Define the output file path
                output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_trimmed.mp3")

                # Write the trimmed audio to a new file
                sf.write(output_file, trimmed_audio, sample_rate)

                print(f"Audio {file} trimmed to precisely 10 seconds around the most concentrated frequency and saved as {output_file}")

if __name__ == "__main__":
    input_folder = input("Enter the path of the input audio folder: ")
    output_folder = input("Enter the path to save the trimmed audio files: ")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    trim_concentrated_audio(input_folder, output_folder)

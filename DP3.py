import librosa
import soundfile as sf
import os

# List of input folders containing audio files
input_folders = [r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Stachyridopsis ambigua",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Sturnia malabarica",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Todiramphus chloris",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Chloropsis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Chloropsis jerdoni",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Cuculus micropterus",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Cyornis magnirostris",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Cyornis poliogenys",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Cyornis unicolor",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Cypsiurus balasiensis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Dicaeum cruentatum",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Dicaeum minullum",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Dicrurus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Glaucidium cuculoides",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Halcyon coromanda",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Iole cacharensis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Locustella davidi",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Loriculus vernalis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Macronus gularis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Motacilla citreola",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Mystery mystery",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Pellorneum ruficeps",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Phylloscopus forresti",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Phylloscopus inornatus",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Pluvialis squatarola",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Pnoepyga pusilla",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Polyplectron bicalcaratum",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Pomatorhinus ruficollis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Psilopogon lineatus",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Psittacula eupatria",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Rimator malacoptilus",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Riparia chinensis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Sphenocichla humei"]

# Desired duration in seconds
desired_duration = 30

for input_folder in input_folders:
    # Create a "Trimmed" subfolder within the input folder if it doesn't exist
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

        # Create an output file path within the "Trimmed" subfolder
        output_file = os.path.join(output_folder, input_file)

        # Write the trimmed or extended audio to the output file
        sf.write(output_file, audio, sample_rate)

        print(f"Audio trimmed/extended and saved as {output_file} in {input_folder}")

print("All audio files processed and saved in the 'Trimmed' subfolders of the input folders.")

import os
import librosa
import soundfile as sf
from scipy import signal

# Define a list of folder paths
folder_paths = [
r"C:\Users\rupali\Desktop\BE Project\Dataset\Psilopogon lineatus",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Psittacula eupatria",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Rimator malacoptilus",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Riparia chinensis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Sphenocichla humei",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Stachyridopsis ambigua",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Sturnia malabarica",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Todiramphus chloris",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Aethopyga gouldiae",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Alcippe cinerea",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Arachnothera magna",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Arborophila torqueola",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Argya longirostris",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Centropus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Chelidorhynx hypoxanthus",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Chloropsis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Chloropsis jerdoni",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Cuculus micropterus",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis magnirostris",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis poliogenys",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis unicolor",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Cypsiurus balasiensis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicaeum cruentatum",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicaeum minullum",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicrurus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Glaucidium cuculoides",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Halcyon coromanda",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Iole cacharensis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Locustella davidi",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Loriculus vernalis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Macronus gularis",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Motacilla citreola",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Mystery mystery",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Pellorneum ruficeps",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Phylloscopus forresti",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Phylloscopus inornatus",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Pluvialis squatarola",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Pnoepyga pusilla",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Polyplectron bicalcaratum",
r"C:\Users\rupali\Desktop\BE Project\Dataset\Pomatorhinus ruficollis",
]

# Define the target sample rate (e.g., 44100 Hz)
target_sr = 44100

for folder_path in folder_paths:
    # Iterate through the .wav files in the folder
    for i, file_name in enumerate(sorted(os.listdir(folder_path))):
        if file_name.endswith(".wav"):
            # Load the audio with the original sample rate
            audio, original_sr = librosa.load(os.path.join(folder_path, file_name))

            # Resample the audio using scipy.signal.resample
            resampled_audio = signal.resample(audio, int(len(audio) * target_sr / original_sr))

            # Create a new filename for the resampled audio (e.g., 'resample_i')
            resampled_file_name = f'resample_{i + 1}.wav'

            # Specify the full file path for the output, including the output filename
            output_path = os.path.join(folder_path, resampled_file_name)

            # Specify the format as WAV when saving
            sf.write(output_path, resampled_audio, target_sr, format='wav')

            print(f"Resampled '{file_name}' and saved as '{resampled_file_name}'")

            # Delete the original file
            os.remove(os.path.join(folder_path, file_name))

print("Resampling of all files in the folders and renaming is completed.")

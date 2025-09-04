import os
import librosa
import noisereduce as nr
import soundfile as sf

# List of folders to process
folders_to_process = [
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Todiramphus chloris",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Aethopyga gouldiae",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Alcippe cinerea",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Arachnothera magna",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Arborophila torqueola",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Argya longirostris",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Centropus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Chelidorhynx hypoxanthus",
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
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Sphenocichla humei",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Stachyridopsis ambigua",
r"C:\Users\rupali\Desktop\BE Project\BIRDS FINA\iBC53\Sturnia malabarica",


]

for folder_path in folders_to_process:
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

    # Print a completion message for the current folder
    print(f"Noise reduction completed for all .wav files in folder: {folder_path}")

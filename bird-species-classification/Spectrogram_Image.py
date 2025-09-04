import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Function to generate and save spectrogram images
def generate_spectrograms(input_folder, output_folder):
    # Make sure the output folder exists; create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List of input .wav files
    input_files = os.listdir(input_folder)

    for file_name in input_files:
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            # Load the audio
            audio, sr = librosa.load(file_path)

            # Generate the spectrogram
            plt.figure(figsize=(8, 4))
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))

            # Set a proper title for the spectrogram
            plt.title('Spectrogram of ' + os.path.splitext(file_name)[0])

            # Save the spectrogram image
            output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
            plt.savefig(output_file)

            # Close the figure to release memory
            plt.close()
            print(f'Spectrogram saved for {file_name}')


# List of input and output folder pairs
folder_pairs = [
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Aethopyga gouldiae", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Aethopyga gouldiae"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Alcippe cinerea", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Alcippe cinerea"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Arachnothera magna", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Arachnothera magna"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Arborophila torqueola", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Arborophila torqueola"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Argya longirostris", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Argya longirostris"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Centropus andamanensis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Centropus andamanensis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Chelidorhynx hypoxanthus", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Chelidorhynx hypoxanthus"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Chloropsis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Chloropsis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Chloropsis jerdoni", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Chloropsis jerdoni"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Cuculus micropterus", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Cuculus micropterus"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis magnirostris", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Cyornis magnirostris"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis poliogenys", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Cyornis poliogenys"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Cyornis unicolor", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Cyornis unicolor"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Cypsiurus balasiensis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Cypsiurus balasiensis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicaeum cruentatum", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Dicaeum cruentatum"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicaeum minullum", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Dicaeum minullum"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Dicrurus andamanensis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Dicrurus andamanensis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Glaucidium cuculoides", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Glaucidium cuculoides"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Halcyon coromanda", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Halcyon coromanda"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Iole cacharensis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Iole cacharensis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Locustella davidi", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Locustella davidi"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Loriculus vernalis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Loriculus vernalis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Macronus gularis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Macronus gularis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Motacilla citreola", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Motacilla citreola"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Mystery mystery", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Mystery mystery"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Pellorneum ruficeps", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Pellorneum ruficeps"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Phylloscopus forresti", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Phylloscopus forresti"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Phylloscopus inornatus", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Phylloscopus inornatus"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Pluvialis squatarola", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Pluvialis squatarola"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Pnoepyga pusilla", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Pnoepyga pusilla"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Polyplectron bicalcaratum", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Polyplectron bicalcaratum"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Pomatorhinus ruficollis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Pomatorhinus ruficollis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Psilopogon lineatus", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Psilopogon lineatus"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Psittacula eupatria", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Psittacula eupatria"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Rimator malacoptilus", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Rimator malacoptilus"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Riparia chinensis", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Riparia chinensis"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Sphenocichla humei", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Sphenocichla humei"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Stachyridopsis ambigua", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Stachyridopsis ambigua"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Sturnia malabarica", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Sturnia malabarica"),
    (r"C:\Users\rupali\Desktop\BE Project\Dataset\Todiramphus chloris", r"C:\Users\rupali\Desktop\BE Project\Audio_Image\Todiramphus chloris"),


    # Add more pairs as needed
]

for input_folder, output_folder in folder_pairs:
    # Call the function to generate and save spectrogram images
    generate_spectrograms(input_folder, output_folder)

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrograms(input_files, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_file in input_files:
        # Load the audio
        audio, sr = librosa.load(input_file)

        # Generate the spectrogram
        plt.figure(figsize=(8, 4))
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))

        # Set a proper title for the spectrogram
        plt.title('Spectrogram of ' + os.path.splitext(os.path.basename(input_file))[0])

        # Save the spectrogram image
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + '.png')
        plt.savefig(output_file)

        # Close the figure to release memory
        plt.close()
        print(f'Spectrogram saved for {input_file} as {output_file}')

if __name__ == "__main__":
    input_files = [
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC79461 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC90559 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC92141 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC98664 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC105278 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC105579 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC187201 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC187204 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC187207 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC189202 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC191850 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC197826 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC268746 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC281889 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC308049 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC312273 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC330918 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC334058 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC334059 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC351880 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC368741 - Eurasian Tree Sparrow - Passer montanus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC368745 - Eurasian Tree Sparrow - Passer montanus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC388245 - Eurasian Tree Sparrow - Passer montanus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC393893 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC402409 - Eurasian Tree Sparrow - Passer montanus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC402410 - Eurasian Tree Sparrow - Passer montanus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC403955 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC404159 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC407511 - Russet Sparrow - Passer cinnamomeus cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC441377 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC444230 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC460279 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC460286 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC461588 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC461589 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC461590 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC461591 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC464267 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472863 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472864 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472865 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472866 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472867 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472868 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472870 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472871 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC472872 - Sind Sparrow - Passer pyrrhonotus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547467 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547476 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547478 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547693 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547694 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547701 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547702 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547722 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC547835 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC577248 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC577680 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC577858 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC577876 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC577878 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC651787 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC651789 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC651790 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC651792 - House Sparrow - Passer domesticus indicus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC785926 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC785928 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC787091 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC788182 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC788183 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35490 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35491 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35492 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35493 - House Sparrow - Passer domesticus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35520 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35521 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC35522 - Russet Sparrow - Passer cinnamomeus_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC73379 - House Sparrow - Passer domesticus parkini_trimmed.mp3",
        r"C:\Users\rupali\Desktop\BE-Project\Bird voices\Trimmed audio\Sparrow\XC73380 - House Sparrow - Passer domesticus parkini_trimmed.mp3",
    ]

    output_folder = input("Enter the path to save the spectrogram images: ")

    generate_spectrograms(input_files, output_folder)

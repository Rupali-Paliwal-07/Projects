import os

# Specify the root directory where you want to delete .wav files
root_directory = r"C:\Users\rupali\Desktop\BE Project\Audio_Image"

# Define a function to delete .wav files
def delete_wav_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")

# Call the function to delete .wav files in the specified directory
delete_wav_files(root_directory)

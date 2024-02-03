import os

# Specify the root directory where you want to delete .png files
root_directory = r"C:\Users\rupali\Desktop\BE Project\Testing_data" # Update with your folder path

# Define a function to delete .png files
def delete_png_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):  # Change the file extension to .png
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")

# Call the function to delete .png files in the specified directory
delete_png_files(root_directory)

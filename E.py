from PIL import Image
import os


def convert_png_to_jpg(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all PNG image files in the input folder
    png_image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    # Convert each PNG image to JPEG format and save to the output folder
    for file in png_image_files:
        png_image_path = os.path.join(input_folder, file)
        img = Image.open(png_image_path)
        jpg_image_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".jpg")
        img.convert('RGB').save(jpg_image_path, 'JPEG')

    print("Images converted from PNG to JPG and saved successfully.")


# Example usage
input_folder_path = input("Enter the path of the folder containing PNG images: ")
output_folder_path = input("Enter the path to save the converted JPG images: ")
convert_png_to_jpg(input_folder_path, output_folder_path)

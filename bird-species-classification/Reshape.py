import cv2
import os

# Define the target shape
target_shape = (299, 299)

# Root directory containing folders with images
root_directory = r"C:\Users\rupali\Desktop\BE Project\Audio_Image"

# Iterate through subdirectories (folders)
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        output_folder_path = os.path.join(r"C:\Users\rupali\Desktop\BE Project\Reshaped_Images", folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        # Iterate through the images in the input directory
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):  # Adjust the file format as needed (e.g., .png, .jpg)
                image_path = os.path.join(folder_path, filename)

                # Read the image
                img = cv2.imread(image_path)

                # Resize the image to the target shape
                resized_img = cv2.resize(img, target_shape)

                # Save the resized image to the output directory
                output_path = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_path, resized_img)

print("Image resizing complete.")

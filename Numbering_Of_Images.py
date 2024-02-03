import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmenters import ContrastNormalization
from PIL import Image

# Input folder containing multiple .png images
input_folder = input("Enter the input folder path (contains .png images): ")

# Output folder
output_folder = input("Enter the output folder path: ")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define data augmentation techniques
seq = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05)),
    ContrastNormalization((0.8, 1.2)),
], random_order=True)

# Get a list of .png files in the input folder
png_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".png")]

# Sort the list of files for consistent numbering
png_files.sort()

# Process each .png image in the input folder
for i, filename in enumerate(png_files):
    input_path = os.path.join(input_folder, filename)

    # Load the input image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Resize the input image to 800x400 pixels if it has different dimensions
    if image.shape != (400, 800):
        print(f"Resizing {filename} to 800x400 pixels.")
        image = cv2.resize(image, (800, 400))

    # Apply data augmentation
    augmented_images = [seq.augment_image(image) for _ in range(10)]  # Generate 10 augmented images

    # Save the augmented images to the output folder using PIL
    for j, augmented_image in enumerate(augmented_images):
        output_filename = f"{i * 10 + j}.png"  # Label images as 0, 1, 2, ...
        output_path = os.path.join(output_folder, output_filename)
        augmented_image_pil = Image.fromarray(augmented_image)
        augmented_image_pil.save(output_path)

        print(f"Saved {output_filename}")

print("Data augmentation complete.")

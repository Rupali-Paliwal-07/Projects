import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmenters import ContrastNormalization
from PIL import Image


# List of input image files
input_files = [
r"C:\Users\rupali\Desktop\NEW Dataset\Spectogram Images\Asian koel\XC31590 - Asian Koel - Eudynamys scolopaceus scolopaceus_trimmed_spectrogram.png",
r"C:\Users\rupali\Desktop\NEW Dataset\Spectogram Images\Asian koel\XC19400 - Asian Koel - Eudynamys scolopaceus scolopaceus_trimmed_spectrogram.png",
r"C:\Users\rupali\Desktop\NEW Dataset\Spectogram Images\Asian koel\XC20267 - Asian Koel - Eudynamys scolopaceus scolopaceus_trimmed_spectrogram.png"







]

# Output folder
output_folder = r"C:\Users\rupali\Desktop\NEW Dataset\Data Augmented Images\Asian koel"
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

# Counter for naming images
counter = 1

# Process each input image
for input_file in input_files:
    # Check if the input file is a .png file
    if not input_file.endswith(".png"):
        print(f"Skipping {input_file}: Input file must be a .png file.")
        continue

    # Load the input image
    image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)

    # Resize the input image to 800x400 pixels if it has different dimensions
    if image.shape != (332,292):
        print(f"Resizing {input_file} to 332x292 pixels.")
        image = cv2.resize(image, (332, 292))

    # Apply data augmentation
    augmented_images = [seq.augment_image(image) for _ in range(10)]  # Generate 10 augmented images

    # Save the augmented images to the output folder with numbering
    for i, augmented_image in enumerate(augmented_images):
        output_filename = f"{counter}.png"
        output_path = os.path.join(output_folder, output_filename)
        augmented_image_pil = Image.fromarray(augmented_image)
        augmented_image_pil.save(output_path)
        counter += 1

print(f"{counter - 1} augmented images saved to {output_folder}")

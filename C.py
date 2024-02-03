import os
import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

# Input folder containing spectrogram images
input_folder = r"C:\Users\rupali\Desktop\BE-Project\Data Set\Sparrow"

# Output folder to save augmented images
output_folder = r"C:\Users\rupali\Desktop\Data ugumented images\Sparrow"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),  # Apply random rotations
    iaa.Affine(scale=(0.5, 1.5)),  # Apply random scaling
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),  # Apply random translations
    iaa.Fliplr(0.5),  # Flip horizontally
    iaa.Flipud(0.5),  # Flip vertically
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add random Gaussian noise
    iaa.Crop(percent=(0, 0.1))  # Crop randomly
])

# List all files in the input folder
image_files = os.listdir(input_folder)

# Augment each image and save to the output folder
for img_file in tqdm(image_files, desc="Augmenting"):
    img_path = os.path.join(input_folder, img_file)
    image = cv2.imread(img_path)

    # Apply augmentation to the image
    augmented_images = [seq(image=image) for _ in range(5)]  # Augment 5 times

    # Save augmented images to the output folder
    for idx, aug_img in enumerate(augmented_images):
        output_path = os.path.join(output_folder, f"{img_file.split('.')[0]}_aug{idx}.jpg")
        cv2.imwrite(output_path, aug_img)

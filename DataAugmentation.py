import os
import cv2
from imgaug import augmenters as iaa

# Function to create the target folder if it doesn't exist
def create_target_folder(target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

# Function to apply data augmentation techniques to an image
def apply_augmentation(image, target_folder, filename, augmenter):
    augmented_images = augmenter(images=[image])

    for i, augmented_image in enumerate(augmented_images):
        output_filename = f"{os.path.splitext(filename)[0]}_{i+1}.png"
        output_path = os.path.join(target_folder, output_filename)
        cv2.imwrite(output_path, augmented_image)

# Input folder containing original spectrogram images
input_folder = input("Enter the path to the input folder: ")

# Target folder to save augmented spectrogram images
output_folder = input("Enter the path to the output folder: ")

# Create the target folder if it doesn't exist
create_target_folder(output_folder)

# Initialize augmenters for data augmentation techniques
seq = iaa.Sequential([
    iaa.Affine(rotate=(-5, 5)),
    iaa.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}),
    iaa.Affine(scale=(0.9, 1.1)),
    iaa.GammaContrast(gamma=(0.7, 1.3)),
    iaa.AddToHueAndSaturation(value=(-10, 10)),
    iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)
])

# Loop through the input folder to process each spectrogram image
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            apply_augmentation(image, output_folder, filename, seq)

print("Data augmentation complete.")

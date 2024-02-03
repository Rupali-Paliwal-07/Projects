import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmenters import ContrastNormalization
from PIL import Image

# Input image file
input_file = input("Enter the input image file (must be a .png file): ")

# Check if the input file is a .png file
if not input_file.endswith(".png"):
    print("Input file must be a .png file.")
else:
    # Output folder
    output_folder = input("Enter the output folder path: ")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the input image
    image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)

    # Resize the input image to 800x400 pixels if it has different dimensions
    if image.shape != (400, 800):
        print("Resizing the input image to 800x400 pixels.")
        image = cv2.resize(image, (800, 400))

    # Define data augmentation techniques
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05)),
        ContrastNormalization((0.8, 1.2)),
    ], random_order=True)

    # Apply data augmentation
    augmented_images = [seq.augment_image(image) for i in range(9, n)]  # Generate n augmented images starting from 9

    # Save the augmented images to the output folder using PIL with names 9, 10, 11, 12, and so on
    for i, augmented_image in enumerate(augmented_images):
        # Use the value of 'i' plus 9 as the filename
        output_filename = f"{i + 9}.png"
        output_path = os.path.join(output_folder, output_filename)
        augmented_image_pil = Image.fromarray(augmented_image)
        augmented_image_pil.save(output_path)

    print(f"{len(augmented_images)} augmented images saved to {output_folder}")

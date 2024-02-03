import os
import cv2
import imgaug.augmenters as iaa
from imgaug.augmenters import ContrastNormalization
from PIL import Image

# Input folder containing multiple .png images
input_folder = input("Enter the input folder path containing .png images: ")

# Verify that the input folder exists
if not os.path.exists(input_folder):
    print("Input folder does not exist.")
else:
    # List all .png files in the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    print("Files in input folder:", input_files)

    # Output folder for saving augmented images
    output_folder = input("Enter the output folder path: ")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define data augmentation techniques
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1), shear=(-10, 10), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.CropAndPad(percent=(-0.1, 0.1)),
        iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0))),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05)),
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation(value=(-10, 10))),
        iaa.AdditiveLaplaceNoise(scale=(0, 0.05)),
        iaa.Affine(rotate=(-10, 10)),
        iaa.LinearContrast((0.7, 1.3)),
        iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)),
    ], random_order=True)

    # Counter for labeling images
    counter = 0

    # Process each image in the input folder
    for filename in input_files:
        input_path = os.path.join(input_folder, filename)

        # Load the input image
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        # Ensure the image dimensions match your specifications (800x400)
        if image.shape == (400, 800):
            # Apply data augmentation
            augmented_images = [seq.augment_image(image) for _ in range(6)]  # Generate 6 augmented images for each input image

            # Save the augmented images to the output folder using PIL
            for i, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder, f"augmented_{counter}.png")
                augmented_image_pil = Image.fromarray(augmented_image)
                augmented_image_pil.save(output_path)

                # Inside the loop, add a print statement to display image dimensions
                print(f"Image dimensions: {image.shape}")

                print(f"Augmented and saved: {output_path}")
                counter += 1

    print(f"{counter} augmented images saved to {output_folder}")

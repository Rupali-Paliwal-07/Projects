import os
import csv

# Function to label and save data
def label_and_save_data(dataset_path, output_path):
    labels = []  # Create an empty list to store labels

    with open(output_path, mode='w', newline='') as csv_file:
        fieldnames = ['image_path', 'class_label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for class_label, class_dir in enumerate(os.listdir(dataset_path)):
            if os.path.isdir(os.path.join(dataset_path, class_dir)):
                # Iterate through class directories
                for image_file in os.listdir(os.path.join(dataset_path, class_dir)):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(dataset_path, class_dir, image_file)
                        labels.append({'image_path': image_path, 'class_label': class_label})
                        writer.writerow({'image_path': image_path, 'class_label': class_label})

    return labels

# Input paths
dataset_path = input("Enter the path of the dataset: ")
output_path = input("Enter the path to save the labels (e.g., labels.csv): ")

# Label and save data
labels = label_and_save_data(dataset_path, output_path)

print(f"Labels have been saved to {output_path}")

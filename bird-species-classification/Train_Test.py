import os
from sklearn.model_selection import train_test_split

# Define the root directory of your dataset containing labeled subfolders
dataset_root = r"C:\Users\rupali\Desktop\BE Project\Data_A_Image_Data"

# Get a list of all labeled subfolders (each subfolder corresponds to a class)
class_folders = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if
                 os.path.isdir(os.path.join(dataset_root, d))]

# Initialize empty lists to hold file paths and corresponding labels
image_files = []
labels = []

# Iterate through the class subfolders
for class_folder in class_folders:
    class_label = os.path.basename(class_folder)  # Get the class label from the subfolder name
    class_images = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Append the file paths and labels to the lists
    image_files.extend(class_images)
    labels.extend([class_label] * len(class_images))

# Split the data into training and testing sets
test_size = 0.3  # You can adjust the test size as needed
X_train, X_test, y_train, y_test = train_test_split(image_files, labels, test_size=test_size, random_state=42)

# Now, X_train and y_train contain training data, and X_test and y_test contain testing data.

import shutil

# Define the paths where you want to save the training and testing data
train_dir = r"C:\Users\rupali\Desktop\BE Project\Training_data" # Specify where you want to save the training data
test_dir =  r"C:\Users\rupali\Desktop\BE Project\Testing_data" # Specify where you want to save the testing data

# Create directories if they don't exist
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Copy the training and testing data to the specified directories
for source, target in zip(X_train, y_train):
    target_dir = os.path.join(train_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy(source, target_dir)

for source, target in zip(X_test, y_test):
    target_dir = os.path.join(test_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy(source, target_dir)

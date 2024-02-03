import os
import shutil
import random


def split_train_test(input_folder, train_folder, test_folder, split_ratio=0.8):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    # Create train and test directories
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # List all files in the input folder
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                 os.path.isfile(os.path.join(input_folder, f))]

    # Shuffle the list of files
    random.shuffle(all_files)

    # Calculate the split index based on the split_ratio
    split_index = int(split_ratio * len(all_files))

    # Divide into train and test sets
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]

    # Copy train files to train folder
    for file in train_files:
        shutil.copy(file, os.path.join(train_folder, os.path.basename(file)))

    # Copy test files to test folder
    for file in test_files:
        shutil.copy(file, os.path.join(test_folder, os.path.basename(file)))

    print("Data split and saved successfully.")


# Example usage
input_folder_path = input("Enter the path of the folder to split: ")
train_data_path = input("Enter the path to save the train data: ")
test_data_path = input("Enter the path to save the test data: ")
split_train_test(input_folder_path, train_data_path, test_data_path)

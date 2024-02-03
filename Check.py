import pandas as pd
from sklearn.model_selection import train_test_split

# Function to split the dataset and save the split data into CSV files
def split_and_save_dataset(csv_file, train_csv, val_csv, test_csv):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Split the data into training, validation, and test sets
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save the split datasets into separate CSV
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    test_data.to_csv(test_csv, index=False)

# Input paths
input_csv = input("Enter the path of the CSV file containing image paths and labels: ")
train_csv = input("Enter the path to save the training data CSV file: ")
val_csv = input("Enter the path to save the validation data CSV file: ")
test_csv = input("Enter the path to save the test data CSV file: ")

# Split the dataset and save the split data into CSV files
split_and_save_dataset(input_csv, train_csv, val_csv, test_csv)

print("Dataset split and saved into CSV files.")

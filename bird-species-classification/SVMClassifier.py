import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load datasets from CSV files
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.drop('class_label', axis=1)  # Features
    y = data['class_label']  # Labels
    return X, y

# Load the training, validation, and testing datasets
train_X, train_y = load_data(r"C:\Users\rupali\Desktop\BE Project\BirdSong - Train_Data.csv")
val_X, val_y = load_data(r"C:\Users\rupali\Desktop\BE Project\BirdSong - Valid_Data.csv")
test_X, test_y = load_data(r"C:\Users\rupali\Desktop\BE Project\BirdSong - Test_Data.csv")

# Create and train an SVM classifier
classifier = SVC(kernel='linear', C=1.0)
classifier.fit(train_X, train_y)

# Make predictions on the validation set
val_predictions = classifier.predict(val_X)

# Calculate validation accuracy
val_accuracy = accuracy_score(val_y, val_predictions)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Make predictions on the test set
test_predictions = classifier.predict(test_X)

# Calculate test accuracy
test_accuracy = accuracy_score(test_y, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

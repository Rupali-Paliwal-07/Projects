import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define the image size
input_shape = (299, 299, 3)

# Define the parameters
batch_size = 32
num_epochs = 20

# Get the paths to the training and testing data folders from the user
train_data_dir = input("Enter the path to the training data folder: ")
test_data_dir = input("Enter the path to the testing data folder: ")

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the dataset
train_data_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_data_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Do not shuffle for evaluation
)

# Get the number of classes
num_classes = len(train_data_generator.class_indices)

# Build a CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjusted to the number of classes in the dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data_generator, epochs=num_epochs)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data_generator)
print(f"Test Accuracy: {test_accuracy}")

# Get the predicted probabilities for ROC-AUC
y_pred_prob = model.predict(test_data_generator)

# Get the true labels
y_true = np.array(test_data_generator.classes)
y_true_one_hot = to_categorical(y_true, num_classes=num_classes)

# Compute ROC-AUC
roc_auc = roc_auc_score(y_true_one_hot, y_pred_prob)

# Compute confusion matrix
y_pred = np.argmax(y_pred_prob, axis=1)
confusion = confusion_matrix(y_true, y_pred)

# Print the ROC-AUC score
print(f"ROC-AUC Score: {roc_auc}")

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion)

# Plot the confusion matrix
labels = list(train_data_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
class_report = classification_report(y_true, y_pred, target_names=labels)
print("Classification Report:")
print(class_report)

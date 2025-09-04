import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load and preprocess image data
def load_and_preprocess_data(image_dir, label):
    images = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(image_path)
                img = img.resize((64, 64))  # Resize the image to a common size
                img = np.array(img)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image: {image_path}, Error: {e}")

    return np.array(images), np.array(labels)

# Function to train and evaluate the Logistic Regression model
def train_and_evaluate_logistic_regression(train_dir):
    try:
        # Load and preprocess training data
        print("Loading and preprocessing training data...")
        bird_images = []
        bird_labels = []

        for i, train_path in enumerate(train_dir):
            images, labels = load_and_preprocess_data(train_path, label=i)
            bird_images.extend(images)
            bird_labels.extend(labels)

        # Split data into training and validation sets with a 70:30 ratio
        train_images, val_images, train_labels, val_labels = train_test_split(
            bird_images, bird_labels, test_size=0.3, random_state=42
        )

        # Ensure there are samples in the training and validation sets
        if len(train_images) == 0 or len(val_images) == 0:
            raise Exception("No images were loaded. Check the dataset paths.")

        # Create and train the Logistic Regression classifier with increased max_iter
        print("Training the Logistic Regression classifier...")
        classifier = LogisticRegression(max_iter=7000)
        classifier.fit(np.array(train_images).reshape(len(train_images), -1), train_labels)

        # Make predictions on the validation set
        print("Making predictions on the validation set...")
        val_predictions = classifier.predict(np.array(val_images).reshape(len(val_images), -1))

        # Calculate accuracy
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

# Specify paths to the training dataset
train_dataset_paths = [
r"C:\Users\rupali\Desktop\BE Project\Training_data\Centropus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Chelidorhynx hypoxanthus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Chloropsis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Chloropsis jerdoni",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Cuculus micropterus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Cyornis magnirostris",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Cyornis poliogenys",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Cyornis unicolor",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Cypsiurus balasiensis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Dicaeum cruentatum",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Dicaeum minullum",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Dicrurus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Glaucidium cuculoides",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Halcyon coromanda",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Iole cacharensis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Locustella davidi",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Loriculus vernalis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Macronus gularis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Motacilla citreola",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Mystery mystery",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Pellorneum ruficeps",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Phylloscopus forresti",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Phylloscopus inornatus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Pluvialis squatarola",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Pnoepyga pusilla",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Polyplectron bicalcaratum",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Pomatorhinus ruficollis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Psilopogon lineatus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Psittacula eupatria",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Rimator malacoptilus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Riparia chinensis",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Sphenocichla humei",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Stachyridopsis ambigua",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Sturnia malabarica",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Todiramphus chloris",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Acridotheres fuscus",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Aethopyga gouldiae",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Alcippe cinerea",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Arachnothera magna",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Arborophila torqueola",
r"C:\Users\rupali\Desktop\BE Project\Training_data\Argya longirostris",

    # Add more training dataset paths for each class
]

test_dataset_paths = [
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Centropus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Chelidorhynx hypoxanthus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Chloropsis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Chloropsis jerdoni",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Cuculus micropterus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Cyornis magnirostris",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Cyornis poliogenys",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Cyornis unicolor",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Cypsiurus balasiensis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Dicaeum cruentatum",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Dicaeum minullum",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Dicrurus andamanensis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Glaucidium cuculoides",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Halcyon coromanda",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Iole cacharensis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Locustella davidi",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Loriculus vernalis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Macronus gularis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Motacilla citreola",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Mystery mystery",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Pellorneum ruficeps",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Phylloscopus forresti",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Phylloscopus inornatus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Pluvialis squatarola",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Pnoepyga pusilla",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Polyplectron bicalcaratum",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Pomatorhinus ruficollis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Psilopogon lineatus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Psittacula eupatria",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Rimator malacoptilus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Riparia chinensis",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Sphenocichla humei",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Stachyridopsis ambigua",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Sturnia malabarica",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Todiramphus chloris",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Acridotheres fuscus",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Aethopyga gouldiae",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Alcippe cinerea",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Arachnothera magna",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Arborophila torqueola",
r"C:\Users\rupali\Desktop\BE Project\Testing_data\Argya longirostris",

]

# Call the train_and_evaluate_logistic_regression function with the training dataset paths
train_and_evaluate_logistic_regression(train_dataset_paths)

import os
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

# Function to train and evaluate the SVM classifier
def train_and_evaluate_svm(train_dirs, test_dirs):
    try:
        # Load and preprocess training data
        print("Loading and preprocessing training data...")
        train_images, train_labels = [], []
        for label, train_dir in enumerate(train_dirs):
            images, labels = load_and_preprocess_data(train_dir, label)
            train_images.extend(images)
            train_labels.extend(labels)
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        # Load and preprocess testing data
        print("Loading and preprocessing testing data...")
        test_images, test_labels = [], []
        for label, test_dir in enumerate(test_dirs):
            images, labels = load_and_preprocess_data(test_dir, label)
            test_images.extend(images)
            test_labels.extend(labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        # Split data into training and validation sets with a 70:30 ratio
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.3, random_state=42
        )

        # Ensure there are samples in the training and testing sets
        if len(train_images) == 0 or len(test_images) == 0:
            raise Exception("No images were loaded. Check the dataset paths.")

        # Create and train the SVM classifier
        print("Training the SVM classifier...")
        classifier = svm.SVC(kernel="linear")
        classifier.fit(train_images.reshape(train_images.shape[0], -1), train_labels)

        # Make predictions on the validation set
        print("Making predictions on the validation set...")
        val_predictions = classifier.predict(val_images.reshape(val_images.shape[0], -1))

        # Calculate accuracy
        accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        # Make predictions on the test set
        print("Making predictions on the test set...")
        test_predictions = classifier.predict(test_images.reshape(test_images.shape[0], -1))

        # Calculate test accuracy
        test_accuracy = accuracy_score(test_labels, test_predictions)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

# Input paths for training and testing datasets
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



    # Add more training dataset paths as needed
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


    # Add more testing dataset paths as needed
]

# Call the train_and_evaluate_svm function
train_and_evaluate_svm(train_dataset_paths, test_dataset_paths)

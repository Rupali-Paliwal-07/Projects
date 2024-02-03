#!/usr/bin/env python
# coding: utf-8

# # ANN MINI-PROJECT 

# # Title: Music Genres Classification 

# # Importing the required packages

# In[1]:


import os
import sys
import scipy
import pickle
import librosa
import numpy as np
import pandas as pd
import seaborn as sns2
import librosa.display
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import Audio
from keras.models import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# # Reading the dataset

# In[2]:


df = pd.read_csv(r"C:\Users\rupali\Desktop\features_3_sec.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


audio_recording =  (r"C:\Users\rupali\Downloads\Data\genres_original\classical\classical.00079.wav")
data, sr = librosa.load(audio_recording)
print(type(data), type(sr))


# In[8]:


data, sr = librosa.load(audio_recording)


# In[9]:


librosa.load(audio_recording, sr=45600)


# In[10]:


import IPython
IPython.display.Audio(data, rate=sr)


# In[11]:


# Data Vizualization 


# In[12]:


# Feature Extraaction 


# In[13]:


class_list = df.iloc[:,-1]
convertor = LabelEncoder()


# In[14]:


y = convertor.fit_transform(class_list)


# In[15]:


y


# In[16]:


print(df.iloc[:,:-1])


# In[17]:



from sklearn.preprocessing import StandardScaler


# In[18]:



fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:,:-1], dtype = float))


# In[ ]:





# In[ ]:





# In[ ]:





# # Train_Test_split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


# In[ ]:


len(y_train)


# In[ ]:


len(y_test)


# In[ ]:


# Building the model


# In[ ]:


from keras.models import Sequential
import keras as k
from keras import layers


# In[ ]:


def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer = optimizer, loss='sparse_categorical_crossentropy',metrics='accuracy')
    return model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs = epochs,batch_size = batch_size)


# In[ ]:


def plotValidate(histroy):
    print("Validation Accuracy", max(histroy.histroy["val_accuracy"]))
    pd.Dataframe(histroy.histroy).plot(figsize=(12,6))
    plt.show()


# # Building a CNN model

# In[ ]:


model = k.models.Sequential([
    k.layers.Dense(512, activation='relu',input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(10, activation='softmax'),

    
])
print(model.summary())
model_histroy=trainModel(model=model, epochs=10, optimizer='adam')







# In[ ]:





# # Accuracy of the model

# In[ ]:


test_loss,test_acc = model.evaluate(X_test, y_test, batch_size = 128)
print("The test loss is :",test_loss)
print("\nThe best accuracy is: ", test_acc*100)


# In[ ]:


#Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)


# In[ ]:


# Save the trained model
model.save('model.h5')


# In[ ]:


#Load the trained model
model = load_model('model.h5')


# In[ ]:


# Define the genre labels
genre_labels = ['classical', 'hiphop', 'jazz', 'metal', 'pop', 'rock','reggae','blues','country','disco']


# In[ ]:


# Load and process the audio file
audio_file = (r"C:\Users\rupali\Downloads\Data\genres_original\classical\classical.00079.wav")
samples, sample_rate = librosa.load(audio_file, sr=None) 


# In[ ]:


# Extract MFCC features
mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20) 


# In[ ]:


# Normalize the features
mfccs_processed = (mfccs - np.mean(mfccs)) / np.std(mfccs)


# In[ ]:


# Expand dimensions to match the model's input shape
input_features = np.expand_dims(mfccs_processed, axis=0)


# In[ ]:


# Load and process the audio file
audio_file = (r"C:\Users\rupali\Downloads\Data\genres_original\classical\classical.00079.wav")
samples, sample_rate = librosa.load(audio_file, sr=None) 
mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20) 
mfccs_processed = (mfccs - np.mean(mfccs)) / np.std(mfccs)
input_features = np.expand_dims(mfccs_processed, axis=0)


# In[ ]:


# Flatten the input features
input_features = np.reshape(input_features, (input_features.shape[0], -1))


# In[ ]:


import os


# In[ ]:


# Take input from the user
audio_path = input("Enter the path of the audio file: ")


# In[ ]:


# Load and process the audio file
data, sr = librosa.load(audio_path)
features = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=58)
input_features = np.mean(features.T, axis=0)
input_features = np.expand_dims(input_features, axis=0)


# In[ ]:


# Make predictions
predictions = model.predict(input_features)
predicted_label_index = np.argmax(predictions[0])
predicted_label = genre_labels[predicted_label_index]


# # Output of the code

# In[ ]:


print("Predicted label:", predicted_label)


# # Analysis of music genres classification

# In[ ]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[ ]:


# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)


# In[ ]:


# Create a confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred_labels)


# In[ ]:


import seaborn as sns
# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:


# Classification report
classification_rep = classification_report(y_test, y_pred_labels, target_names=genre_labels)
print(classification_rep)


# In[ ]:


# Convert true labels to one-hot encoded format
y_true_one_hot = label_binarize(y_test, classes=np.arange(len(genre_labels)))


# In[ ]:


# Compute predicted probabilities for each class
y_pred_prob = model.predict(X_test)


# In[ ]:



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(genre_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_prob[:, i])
    roc_auc[i] = roc_auc_score(y_true_one_hot[:, i], y_pred_prob[:, i])


# In[ ]:


# Compute micro-average ROC curve and ROC area
fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_pred_prob.ravel())
roc_auc_micro = roc_auc_score(y_true_one_hot, y_pred_prob, average='micro')


# In[ ]:


# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(len(genre_labels)):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(genre_labels[i], roc_auc[i]))


# In[ ]:


# Plot micro-average ROC curve
plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle='--', lw=2,
         label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc_micro))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import os
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Path to audio dataset and metadata CSV file
audio_dataset_path = r"C:\Users\rupali\Documents\archive\Data\genres_original"
metadata = pd.read_csv(r"C:\Users\rupali\Documents\archive\Data\features_30_sec.csv")

# Feature extraction function
def features_extractor(file):
    # Extract MFCC features from audio files
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs_features.T, axis=0)

# Extract features for each audio file
extracted_features = []
for index, row in metadata.iterrows():
    try:
        final_class_labels = row['label']
        file_name = os.path.join(os.path.abspath(audio_dataset_path), final_class_labels + '/', str(row["filename"]))
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except Exception as e:
        print(f"Error: {e}")
        continue

# Convert extracted features to pandas DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

# Split dataset into independent & dependent dataset
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Encode categorical labels
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the neural network model
num_labels = y.shape[1]
model = Sequential()
# Add layers to the model (add more layers as needed)
model.add(Dense(1024, input_shape=(40,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), verbose=1)

# Evaluate model on test data
model.evaluate(X_test, y_test, verbose=0)


# Code modifications...

# Train the model and save it for later use
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=1)
model.save('saved_models/audio_classification_model.h5')

# Load the trained model for future use
model = tf.keras.models.load_model('saved_models/audio_classification_model.h5')

# Function to predict genre for a given audio file
def predict_genre(audio_file_path):
    audio, sample_rate = librosa.load(audio_file_path, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_probabilities = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted_probabilities, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)
    return prediction_class[0]



# Predict genre for a given audio file
audio_file_path = input("Enter the path to the audio file: ")
predicted_genre = predict_genre(audio_file_path)
print(f"Predicted genre for the audio file: {predicted_genre}")

# Predict labels for the test set using the loaded model
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)







# In[ ]:





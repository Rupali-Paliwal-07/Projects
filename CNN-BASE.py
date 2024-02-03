#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Modified CNN architecture
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(500, activation='relu'),  # Modified to have 500 neurons
    Dense(10, activation='softmax')  # Assuming 10 classes
])

# Path to the main dataset folder
dataset_folder = input("Enter the path to the main dataset folder: ")

# Data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=35, validation_data=val_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('modified_CNN_model.h5')

# Get predictions for ROC curve
y_pred = model.predict(test_generator)
y_true = test_generator.classes

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
n_classes = len(np.unique(y_true))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, y_pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f}) for class {i}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:





# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Modified CNN architecture
model = Sequential([
    Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(500, activation='relu'),  # Modified to have 500 neurons
    Dense(7, activation='softmax')  # Assuming 10 classes
])

# Path to the main dataset folder
dataset_folder = input("Enter the path to the main dataset folder: ")

# Data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    dataset_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=5, validation_data=val_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Save the trained model
model.save('modified_CNN_model.h5')

# Get predictions for ROC curve
y_pred = model.predict(test_generator)
y_true = test_generator.classes

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
n_classes = len(np.unique(y_true))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, y_pred[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:0.2f}) for class {i}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# ... (Your existing code)

# Train the model
history = model.fit(train_generator, epochs=35, validation_data=val_generator)

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:





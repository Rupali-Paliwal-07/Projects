#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix,classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


df = pd.read_csv("malware dataset.csv")


# In[24]:


print(df.columns)


# In[25]:


# Extract the relevant features
selected_features = ["static_prio", "usage_counter"]
X = df[selected_features]
y = df["classification"]


# In[26]:


# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[27]:


# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[28]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# In[12]:


# Define the input shape based on the number of features
input_shape = (X_train.shape[1], 1)


# In[ ]:





# In[30]:


# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)


# In[31]:


train_loss, train_accuracy = model.evaluate(X_train, y_train)
print("Training Accuracy:", train_accuracy)


# In[20]:


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Testing Accuracy:", test_accuracy)


# In[32]:


#Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)


# In[33]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[12]:


print("Confusion Matrix:")
print(cm)


# In[13]:


classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(classification_rep)


# In[16]:


# Take input from the user
input_data = []
for feature_name in selected_features:
    value = float(input(f"Enter the value for {feature_name}: "))
    input_data.append(value)
input_data = np.array(input_data)
input_data = input_data.reshape(1, -1)
input_data = scaler.transform(input_data)


# In[17]:


# Make prediction for the user input
prediction = model.predict(input_data)
threshold = 0.5  # Set the threshold for classification
prediction_label = "malware present" if prediction >= threshold else "malware absent"


# In[18]:


print(f"The prediction is: {prediction_label}")


# In[ ]:





# In[ ]:





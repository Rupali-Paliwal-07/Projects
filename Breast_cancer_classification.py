#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project 

# # Title : Breast Cancer Classification 

# # Techniques : 
# # SVM,Logistic Regression,Random Forest Classifer

# # 1.SVM Technique

# In[1]:


#######################
# Required Python Packages
#######################

import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[2]:


########################################################################
# Functions name: Support Vector Machine (SVM)
# Description : To easily handle multiple continous & catergorical data
            #   It offers high accuracy compared to other classifiers
# Author : Rupali Paliwal
# Date : 24-03-2024
#######################################################################


# In[3]:


def DemoSVM():
    cancer = datasets.load_breast_cancer()


# In[4]:


def DemoSVM():
    cancer = datasets.load_breast_cancer()

    print("Features of the cancer dataset : ",cancer.feature_names)

    print("Labels of the cancer dataset:",cancer.target_names)

    print("Shape of dataset is:",cancer.data.shape)

    print("First 5 records are: ")
    print(cancer.data[0:5])
    print("Target of dataset :", cancer.target)


# In[5]:



def DemoSVM():
    cancer = datasets.load_breast_cancer()

    print("Features of the cancer dataset : ",cancer.feature_names)

    print("Labels of the cancer dataset:",cancer.target_names)

    print("Shape of dataset is:",cancer.data.shape)

    print("First 5 records are: ")
    print(cancer.data[0:5])

    print("Target of dataset :", cancer.target)

    X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)

    clf = svm.SVC(kernel='linear')

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy of the model is:",metrics.accuracy_score(y_test,y_pred)*100)

    print('SVC classification_report')
    print('___'*20)
    print(classification_report(y_test,y_pred))


# In[ ]:





# In[6]:


import seaborn as sns

def data_visualization(X_train, y_train, feature_names):
    # Combine features and target into a single DataFrame
    df = pd.DataFrame(X_train, columns=feature_names)
    df['target'] = y_train

    # Histogram
    sns.histplot(data=df, x='mean radius', hue='target', kde=True)
    plt.title('Histogram of Mean Radius')
    plt.show()

    # Bar chart
    sns.countplot(data=df, x='target')
    plt.title('Counts of Target Classes')
    plt.show()

    # Line plot
    sns.lineplot(data=df, x='mean radius', y='mean texture', hue='target')
    plt.title('Line Plot: Mean Radius vs Mean Texture')
    plt.show()

def DemoSVM():
    cancer = datasets.load_breast_cancer()
    feature_names = cancer.feature_names

    print("Features of the cancer dataset: ", feature_names)
    print("Labels of the cancer dataset:", cancer.target_names)
    print("Shape of dataset is:", cancer.data.shape)
    print("First 5 records are:")
    print(cancer.data[0:5])
    print("Target of dataset:", cancer.target)

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy of the model is:", metrics.accuracy_score(y_test, y_pred) * 100)

    print('SVC classification_report')
    print('___' * 20)
    print(classification_report(y_test, y_pred))

    # Perform data visualization
    data_visualization(X_train, y_train, feature_names)

def main():
    print("_____ Support Vector Machine_____")
    DemoSVM()

# Application Starter
if __name__ == "__main__":
    main()


# In[ ]:





# # 2.Random Forest Classifier

# In[7]:


#######################
# Required Python Packages
#######################

import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[8]:


#######################
# File Paths
#######################

INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"


# In[9]:


#############################
# Headers
#############################

HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape","MarginalAdhesion", "SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]


# In[10]:


#####################################
#Function name: read_data
#Description : Read the data into pandas dataframe
#Inpt : path of CSV file
#Output : Gives the data
#Author : Rupali Paliwal
#Date : 24/03/2023
######################################

def read_data(path):
    data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")
    return data


# In[11]:


#####################################
# Function name : get_headers
# Description : dataset headers
# Input : dataset
# Output : Returns the headers
# Author : Rupali Paliwal
# Date : 24/03/2023
#####################################

def get_headers(dataset):
    return dataset.columns.values


# In[12]:


#####################################
# Function name : add_headers
# Description : Add the headers to the dataset
# Input : dataset
# Output : Updated dataset
# Author : Rupali Paliwal
# Date : 24/03/2023
#####################################

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset


# In[13]:


#####################################
# Function name : data_file_to_csv
# Input : Nothing
# Output : Write the data to CSV
# Author : Rupali Paliwal
# Date : 24/03/2023
####################################

def data_file_to_csv():
    #Headers
    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCell","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]

#Load the dataset into Pandas data frame
dataset = read_data(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin .csv")

# Add the headers to the loaded dataset
dataset = add_headers(dataset, HEADERS)

# Save the loaded dataset into csv format
dataset.to_csv (OUTPUT_PATH, index = False)
print("File saved..!!!")


# In[14]:


############################################
# Function name : split_dataset
# Description : Split the dataset with train_percentage
# Input : Dataset with related information
# Output : Dataset after spliting
# Author : Rupali Paliwal
# Date : 23/03/2023
############################################

def split_dataset(dataset, train_percentage, feature_headers,target_header):
    # Split dataset into train & test dataset
    train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],dataset[target_header],train_size= train_percentage)
    return train_x,test_x,train_y,test_y


# In[15]:


###########################################
# Function name : handel_missing_values
# Description : Filter missing values from the dataset
# Input : Dataset with missing values
# Output : Dataset by removing missing values
# Author : Rupali Paliwal
# Date : 24-03-2023
############################################

def handel_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header]!= missing_label]


# In[16]:


###############################################
# Functions name: random_forest_classifier
# Description : To train the random forest classifier with features and target data
# Author : Rupali Paliwal
# Date : 24-03-2024
################################################

def random_forest_classifier(features,taraget):
    clf = RandomForestClassifier()
    clf.fit(features, taraget)
    return clf


# In[17]:


#################################################
# Function name: dataset_statistics
# Description : Basic statistics of the dataset
# Input : Dataset
# Output : Description of dataset
# Author : Rupali Paliwal
# Date : 24-03-2024
##################################################

def dataset_statistics(dataset):
    print(dataset.describe())


# In[18]:


###################################################
# Function name: maain
# Description : Main function from where execution starts
# Author : Rupali Paliwal
# Date : 2403-2023
###################################################

def main():
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # GET BASIC STATISTICS OF THE LOADED DATASET
    dataset_statistics(dataset)

    # Filter missing values
    dataset = handel_missing_values(dataset,HEADERS[6],'?')
    train_x,test_x,train_y,test_y = split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])

    # Train and Test dataset size details
    print("Train_x Shape :: ",train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_x Shape :: ", test_y.shape)

    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x,train_y)
    print("Trained model ::", trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0,205):
        print("Actual outcome :: {} and Predicted output :: {}".format(list(test_y)[i],predictions[i]))

    print("Train Accuracy :: ", accuracy_score(train_y,trained_model.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y,predictions))

    print('RF classification_report')
    print('___' * 19)
    print(classification_report(test_y, predictions))


# In[19]:


# Data Visualization
###########################################################################################
# reading the database
data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")

# Bar chart with day against tip
plt.bar(dataset['ClumpThickness'], dataset['SingleEpithelialCellSize'])

plt.title("Bar Chart")

# Setting the X and Y labels
plt.xlabel('ClumpThickness')
plt.ylabel('SingleEpithelialCellSize')

# Adding the legends
plt.show()
 ###########################################################################################
# reading the database
data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")

# histogram of total_bills
plt.hist(data['CancerType'])

plt.title("Histogram")

# Adding the legends
plt.show()
###########################################################################################
# reading the database
data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")
sns.barplot(x='BareNuclei', y='MarginalAdhesion', data=data,
            hue='CancerType')

plt.show()
###########################################################################################
# reading the database
data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")
sns.histplot(x='UniformityCellSize', data=data, kde=True, hue='CancerType')

plt.show()
############################################################################################
# reading the database
data = pd.read_csv(r"C:\Users\rupali\Downloads\breast-cancer-wisconsin.csv")

# plotting the scatter chart
fig = px.line(data, y='BlandChromatin', color='CancerType')

# showing the plot
fig.show()
##############################################################################################






# In[20]:


############################################
#Application Starter
############################################

if __name__ == "__main__":
    main()



# In[ ]:





# # 3. Logistic Regression 

# In[21]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def logistic_regression_classification():
    # Load the breast cancer dataset
    cancer = load_breast_cancer()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

    # Create a logistic regression classifier
    clf = LogisticRegression()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Compute ROC curve and ROC AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Generate the classification report
    report = classification_report(y_test, y_pred)

    return report


def main():
    print("Breast Cancer Classification using Logistic Regression")
    report = logistic_regression_classification()
    print(report)

# Application Starter
if __name__ == "__main__":
    main()


# # Comparisions of three Algorithms

# In[22]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

def compare_algorithms():
    # Load the breast cancer dataset
    cancer = load_breast_cancer()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

    # Create instances of the classifiers
    svm_clf = SVC()
    rf_clf = RandomForestClassifier()
    lr_clf = LogisticRegression()

    # Train the classifiers
    svm_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    lr_clf.fit(X_train, y_train)

    # Make predictions
    svm_pred = svm_clf.predict(X_test)
    rf_pred = rf_clf.predict(X_test)
    lr_pred = lr_clf.predict(X_test)

    # Calculate accuracy scores
    svm_accuracy = accuracy_score(y_test, svm_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    # Create a comparison table
    data = {'Algorithm': ['SVM', 'Random Forest', 'Logistic Regression'],
            'Accuracy': [svm_accuracy, rf_accuracy, lr_accuracy]}
    df = pd.DataFrame(data)

    return df

def main():
    print("Breast Cancer Classification Algorithm Comparison")
    comparison_table = compare_algorithms()
    print(comparison_table)

# Application Starter
if __name__ == "__main__":
    main()


# In[23]:


from IPython.display import Markdown, display

# Create the table data as a list of lists
table_data = [
    ['1', 'SVM','0.923977' ,],
    ['2',  'RFC','0.982456',],
    ['3', 'Logistic Regression', '0.953216'],
]
# Create the table markdown
table_markdown = '|'.join(['{:40}'.format(header) for header in ['Sr.No', 'Algorithm', 'Accuracy']])
table_markdown += '\n' + '|'.join(['-' * 30] * 3)

for row in table_data:
    table_markdown += '\n' + '|'.join(['{:30}'.format(cell) for cell in row])

# Display the table in Jupyter Notebook
display(Markdown("<style>table {font-size: 20px}</style>"))
display(Markdown(table_markdown))


# In[ ]:





import numpy as np  
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split   
from sklearn import metrics 

# Define column names for the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] 

# Read dataset into a pandas dataframe
dataset = pd.read_csv("9-dataset.csv", names=names) 

# Split dataset into features (X) and target (y)
X = dataset.iloc[:, :-1]  # Features (all columns except the last one)
y = dataset.iloc[:, -1]   # Target (the last column)

# Display the first few rows of the features
print("First 5 rows of the dataset:")
print(X.head())

# Split the dataset into training and testing sets (90% training, 10% testing)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)  

# Initialize and train the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5).fit(Xtrain, ytrain)  

# Predict the labels for the test set
ypred = classifier.predict(Xtest) 

# Display the results
print("\n-------------------------------------------------------------------------")
print('%-25s %-25s %-25s' % ('Original Label', 'Predicted Label', 'Correct/Wrong'))
print("-------------------------------------------------------------------------")
for i, (label, pred) in enumerate(zip(ytest, ypred)): 
    print('%-25s %-25s' % (label, pred), end="")
    if label == pred: 
        print(' %-25s' % ('Correct')) 
    else: 
        print(' %-25s' % ('Wrong')) 
print("-------------------------------------------------------------------------")

# Display the confusion matrix
print("\nConfusion Matrix:\n", metrics.confusion_matrix(ytest, ypred))   
print("-------------------------------------------------------------------------")

# Display the classification report
print("\nClassification Report:\n", metrics.classification_report(ytest, ypred))  
print("-------------------------------------------------------------------------")

# Display the accuracy of the classifier
print('Accuracy of the classifier is %0.2f' % metrics.accuracy_score(ytest, ypred)) 
print("-------------------------------------------------------------------------")

# Importing Libraries

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

# Getting the dataset
breast_cancer = sklearn.datasets.load_breast_cancer()

#print(breast_cancer)

X = breast_cancer.data
Y = breast_cancer.target

#print(X)
#print(Y)

#print(X.shape, Y.shape)

# Import data to the Pandas Data Frame

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

data['class'] = breast_cancer.target

data.head()

data.describe()

# print(data['class'].value_counts())

# print(breast_cancer.target_names)

data.groupby('class').mean()

# Train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1, stratify=Y, random_state=1)

classifier = LogisticRegression()   # loading the logistic regression model to the variable "classifier"

# training the model on training data
classifier.fit(X_train,Y_train)

pickle.dump(classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

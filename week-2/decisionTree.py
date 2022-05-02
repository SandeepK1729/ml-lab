import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# loading dataset
iris = load_iris()
X, y = pd.DataFrame(iris.data), pd.DataFrame(iris.target)

# spliting of dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
# traing the model
dtc.fit(X_train, y_train)
# prediction
y_pred = dtc.predict(X_test)
# finding the accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)
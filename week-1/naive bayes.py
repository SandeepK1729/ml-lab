import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = pd.read_csv("../csv_datasets/wine.csv")
y = wine['Wine']
X = wine.drop('Wine', axis = 1)

value=[]
k = 5
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 1 / k)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy= accuracy_score(y_test,y_pred)

    value.append(accuracy)
    
print("average accuracy of the model is ", sum(value) / k)
plt.title("Accuracies Of Naive Bayesian")
plt.bar(range(1, k + 1),value)

plt.show()


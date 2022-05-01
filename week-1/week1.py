import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

wine = pd.read_csv("wine.csv")
X = wine[["Alcohol","Malic.acid","Ash","Acl","Mg","Phenols","Flavanoids","Nonflavanoid.phenols","Proanth","Color.int","Hue","OD","Proline"]]
y = wine["Wine"]

value=[]
k = 5
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 1 / k)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy= accuracy_score(y_test,y_pred)

    value.append(accuracy)
    
print("average of the model",(sum(value))/len(value))
plt.bar( range(k),value)
plt.show()
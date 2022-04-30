import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
gnb = GaussianNB()

wine = pd.read_csv("wine.csv")
X = wine[["Alcohol","Malic.acid","Ash","Acl","Mg","Phenols","Flavanoids","Nonflavanoid.phenols","Proanth","Color.int","Hue","OD","Proline"]]
y = wine["Wine"]

import matplotlib.pyplot as plt
comb=[]
value=[]
for i in range(4):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy=metrics.accuracy_score(y_test,y_pred)

    comb.append('comb'+str(i))
    value.append(accuracy)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
value.append((sum(value))/len(value))
print("average of the model",(sum(value))/len(value))
plt.bar([0,1,2,3,4],value)
plt.show()
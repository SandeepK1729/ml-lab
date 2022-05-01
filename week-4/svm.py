import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, test_size=0.25 )

clf = SVC(kernel = 'linear').fit(X_train,y_train)
y_pred = clf.predict(X_test)            
acc = accuracy_score(y_test, y_pred)    
# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred)   
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica']
                    )

sns.heatmap(cm_df, annot=True)
plt.title('Accuracy using SVM algorithm is {0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris=datasets.load_iris()
k = 4

value = []
for i in range(k):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size= 1 / k)

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    value.append(accuracy)
    
print("average accuracy of the knn model is ", sum(value) / k)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica']
                     )

sns.heatmap(cm_df, annot=True)
plt.title('Accuracy using KNN :{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
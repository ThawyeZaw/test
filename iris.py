import sklearn
import numpy
import pandas as pd
from sklearn import datasets

#NOTE - data understanding
iris = datasets.load_iris()

iris_data = pd.DataFrame(iris.data, columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

iris_data['label'] = iris.target

print(iris_data)

print("\n", iris_data.describe())

#NOTE - data preprocessing

iris_data.dropna(axis=0, inplace=True)

X = iris_data.iloc[:,0:-1]

Y = iris_data.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

#NOTE - Modeling

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

#NOTE - Evaluation
y_predict = model.predict(X_test)
print("\nPrediction: ", y_predict)

from sklearn.metrics import confusion_matrix
print("\n", "Confusion matrix: ")
print(confusion_matrix(Y_test, y_predict))

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

print('\nAccuracy Score: ',accuracy_score(Y_test, y_predict)*100,'%')
print('Precision Macro Score: ',precision_score(Y_test, y_predict,average = 'macro')*100,'%')
print('Recall_Score: ',recall_score(Y_test, y_predict, average = 'macro')*100,'%')
print('F_Score: ',f1_score(Y_test, y_predict, average = 'macro')*100,'%')
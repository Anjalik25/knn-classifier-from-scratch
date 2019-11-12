import pandas as pd
import numpy as np
from KNearestNeighbor import KNearestNeighbors
from sklearn.model_selection import train_test_split

data = pd.read_csv('Iris.csv')

data['Species'].replace({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}, inplace=True)

x = data.iloc[:, 1:5].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
knn = KNearestNeighbors(k=5)
knn.fit(X_train, y_train)

def predict_new():
    sl = float(input("Enter Sepal Length"))
    sw = float(input("Enter Sepal Width"))
    pl = float(input("Enter Petal Length"))
    pw = float(input("Enter Petal Width"))
    X_new = np.array([[sl], [sw], [pl], [pw]]).reshape(1, 4)
    result = knn.predict(X_new)
    if result == 0:
        print("It is Iris-setosa")
    elif result == 1:
        print("It is Iris-virginica")
    else:
        print("It is Iris-versicolor")
    ans=input("Do you want to enter more??(y/n)")
    if ans == 'y' or ans == 'Y':
        predict_new()
    else:
        exit(0)

predict_new()



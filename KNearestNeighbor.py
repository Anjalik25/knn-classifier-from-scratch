import operator
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print(self.X_train)
        print(self.y_train)
        print("Training done")

    def predict(self, X_new):
        distance = {}
        counter = 0

        for i in self.X_train:
            dist = 0
            for n in range(0, X_new.shape[1]):
                dist = dist + ((X_new[0][n] - i[n])**2)
            distance[counter] = dist**(1/2)
            counter = counter + 1

        distance = sorted(distance.items(), key=operator.itemgetter(1))
        self.classify(distance=distance[:self.k])

    def classify(self, distance):
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        return Counter(label).most_common()[0][0]



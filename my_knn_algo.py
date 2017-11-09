from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np 
from scipy.spatial.distance import euclidean

class ScrappyKNN():
	
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		labels = []
		for i in range(len(X_test)):
			#label = np.random.choice(self.y_train)
			#labels.append(label)
			index = self.closest(X_test[i])
			labels.append(self.y_train[index])

		return labels

	def closest(self, sample):
		shortest_distance = euclidean(sample, self.X_train[0])
		index = 0
		for i in range(1,len(self.X_train)):
			distance = euclidean(sample,self.X_train[i])
			if distance < shortest_distance:
				shortest_distance = distance
				index = i
		return index



iris = load_iris()

X = iris.data
y = iris.target

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)

clf = ScrappyKNN()
clf.fit(train_X, train_y)
predictions = clf.predict(test_X)

print accuracy_score(predictions, test_y)
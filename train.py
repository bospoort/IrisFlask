import pickle
import sys
import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

print ('Python version: {}'.format(sys.version))
print ()

# load Iris dataset
iris = load_iris()
print ('Iris dataset shape: {}'.format(iris.data.shape))

# load features and labels
X, Y = iris.data, iris.target

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# train a logistic regression model
clf1 = LogisticRegression(C=1/reg).fit(X, Y)
print (clf1)

accuracy = clf1.score(X, Y)
print ("Accuracy is {}".format(accuracy))

print ("Export the model to model.pkl")
f = open('model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

# load the model back 
print("Import the model from model.pkl ")
f2 = open('model.pkl', 'rb')
clf2 = pickle.load(f2)

# predict a new sample
X_new = [[3.0, 3.6, 1.3, 0.25]]
print ('New sample: {}'.format(X_new))
pred = clf2.predict(X_new)
print('Predicted class is {}'.format(pred))

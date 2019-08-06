import matplotlib.pyplot as plt
import numpy
from sklearn import datasets

#from sklearn.linear_model import Perceptron
from perceptron import Perceptron

iris = datasets.load_iris()


# sepal length, sepal width, petal length, petal width
X = iris.data[:, :2]
#import sys
#sys.exit(0)

y = iris.target
#y = [ 0 if x == 1 else 1 for x in iris.target]

perceptron = Perceptron(n_iter=100)

perceptron.fit(X, y)

def to_plot(weights, bias):
    a, b = weights
    i = bias
    xx = numpy.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    yy = (- a * xx - i) / b
    return xx, yy

fig, axis = plt.subplots()

axis.scatter(X[:, 0], X[:, 1], c=['brg'[c] for c in y], s=120)
#axis.plot(*to_plot(perceptron.coef_[0], perceptron.intercept_[0]))

axis.plot(*to_plot(perceptron.weights, perceptron.bias))

axis.axis([min(X[:, 0]) - .5, max(X[:, 0]) + .5, min(X[:, 1]) - .5, max(X[:, 1]) + .5])
plt.show()

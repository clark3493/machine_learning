import testpathmagic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from perceptron import Perceptron


class PerceptronTest(object):

    def __init__(self):
        """
        Load the Iris dataset
        """
        self.df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                              'machine-learning-databases/iris/iris.data',
                              header=None)
        self.df.tail()
        self.ppn = None

        # select setosa and versicolor
        self.y = self.df.iloc[0:100, 4].values
        self.y = np.where(self.y == 'Iris-setosa', -1, 1)

        # extract sepal length and petal length
        self.X = self.df.iloc[0:100, [0, 2]].values

        self.ppn = Perceptron(eta=0.1, n_iter=10)
        self.ppn.fit(self.X, self.y)

    def plot_raw_data(self):
        X = self.X

        # plot data
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()

    def plot_fit_errors(self):
        plt.plot(range(1, len(self.ppn.errors)+1), self.ppn.errors, marker='o')
        plt.xlabel('epochs')
        plt.ylabel('Number of updates')
        plt.show()

    def plot_decision_region(self, resolution=0.02):

        X = self.X
        y = self.y

        # set up marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.ppn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')

        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()

if __name__ == '__main__':
    pt = PerceptronTest()

    pt.plot_raw_data()
    pt.plot_fit_errors()
    pt.plot_decision_region()
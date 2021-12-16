import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron


def accuracy(labels, predicted):
    accuracy = np.sum(labels == predicted) / len(labels)
    return accuracy

samples, labels = datasets.make_blobs(n_samples=150, n_features=2, centers=2,cluster_std=1.05,random_state=2)

trainingSamples, testSamples, trainingLabels, testLabels = train_test_split(samples, labels, test_size=0.2, random_state=123)

perceptron = Perceptron(learningRate=0.01, numberOfIterations=1000)
perceptron.fit(trainingSamples, trainingLabels)
prediction = perceptron.preditct(testSamples)

print("Perceptron classification accuracy is: ", accuracy(testLabels, prediction))

figure = plt.figure()
ax = figure.add_subplot(1,1,1)
plt.scatter(trainingSamples[:,0], trainingSamples[:,1], marker='o', c=trainingLabels)

sample0_1 = np.amin(trainingSamples[:,0])
sample0_2 = np.amax(trainingSamples[:,0])

sampel1_1 = (-perceptron.weights[0] * sample0_1 - perceptron.bias)/perceptron.weights[1]
sampel1_2 = (-perceptron.weights[0] * sample0_2 - perceptron.bias)/perceptron.weights[1]

ax.plot([sample0_1, sample0_2], [sampel1_1, sampel1_2], "k")

ymin = np.amin(trainingSamples[:, 1])
ymax = np.amax(trainingSamples[:, 1])

ax.set_ylim([ymin - 3, ymax + 3])

plt.show()



    
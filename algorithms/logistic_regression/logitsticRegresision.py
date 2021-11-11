import numpy as np

class LogisticRegression:

    def __init__(self, learningRate=0.001, numberOfIterations=1000):
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations
        self.weights = None
        self.bias = None

    def fit(self, samples, labels):
        # initialize parameters
        numberOfSamples, numberOfFeatures = samples.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0

        # gradient descent
        for _ in range(self.numberOfIterations):
            # approximate the labels with liniear combination of weights and samples with bias
            linearModel = np.dot(samples, self.weights) + self.bias
            # apply sigmoid
            predicted = self._sigmoid(linearModel)
            # compute gradients 
            delivaritiveWeights = (1/numberOfSamples) * np.dot(samples.T, predicted - labels)
            delivaritiveBias = (1/numberOfSamples) * np.sum(predicted - labels)
            # update parameters based on the gradient
            self.weights -= self.learningRate * delivaritiveWeights
            self.bias -= self.learningRate * delivaritiveBias


    def predict(self, samples):
        linearModel = np.dot(samples, self.weights) + self.bias
        predicted = self._sigmoid(linearModel)
        labelPredictedClasses = [1 if i > 0.5 else 0 for i in predicted]
        return labelPredictedClasses


    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
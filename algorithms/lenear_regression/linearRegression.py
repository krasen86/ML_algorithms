import numpy as np

class LinearRegression:
    # learning rate 0.01 
    def __init__(self, lerningRate = 0.01, numberOfIterations = 1000):
        self.lerningRaate = lerningRate
        self.numberOfIterations = numberOfIterations
        self.weights = None
        self.bias = None

    def fit(self, samples, labels):
        # initialize parameters
        numberOfSamples, numberOfFeatures = samples.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0
        # Gradient descent 
        for _ in range(self.numberOfIterations):
            label_predicted = np.dot(samples, self.weights) + self.bias
            # calculate teh deriviates/gradient
            derivativeWeight = (1/numberOfSamples) * np.dot(samples.T, (label_predicted - labels))
            derivativeBias = (1/numberOfSamples) * np.sum(label_predicted -labels)
            # iteratevly update the parameters
            self.weights -= self.lerningRaate * derivativeWeight
            self.bias -= self.lerningRaate * derivativeBias
        


    def predict(self, samples):
        label_predicted = np.dot(samples, self.weights) + self.bias
        return label_predicted




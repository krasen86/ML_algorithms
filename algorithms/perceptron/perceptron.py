import numpy as np

class Perceptron:

    def __init__(self, learningRate=0.01, numberOfIterations=1000):
        self.learningRate = learningRate
        self.numberOfIterations = numberOfIterations
        self.activationFunction = self._unit_step_function
        self.weights = None
        self.bias = None


    def _unit_step_function(self, x):
        return np.where(x>=0, 1, 0)

    
    def fit(self, trainingSamples, trainingLabels):
        numberOfSamples, numberOfFeatures = trainingSamples.shape
        self.weights = np.zeros(numberOfFeatures)
        self.bias = 0
        labels_ = np.array([1 if i > 0 else 0 for i in trainingLabels])
        for _ in range(self.numberOfIterations):
            for index, sample in enumerate(trainingSamples): 
                liniarOutput = np.dot(sample, self.weights) + self.bias
                labelPredicted = self.activationFunction(liniarOutput)

                update = self.learningRate * (labels_[index]-labelPredicted)
                self.weights += update * sample
                self.bias += update

    def preditct(self, samples):
        liniarOutput = np.dot(samples, self.weights) + self.bias
        labelPredicted = self.activationFunction(liniarOutput)
        return labelPredicted

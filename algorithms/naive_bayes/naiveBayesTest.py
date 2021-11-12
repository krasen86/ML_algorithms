import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib as plt

from naiveBayes import NaiveBayes

def accuracy(labels, labelsPredicted):
    accuracy = np.sum(labels==labelsPredicted) / len(labels)
    return accuracy


samples, labels = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
samplesTrain, samplesTest, labelsTrain, labelsTest = train_test_split(samples, labels, test_size=0.2, random_state=123)

naiveBayes = NaiveBayes()
naiveBayes.fit(samplesTrain, labelsTrain)
predictions = naiveBayes.predict(samplesTest)

print("Naive bayes accuracy", accuracy(labelsTest, predictions))
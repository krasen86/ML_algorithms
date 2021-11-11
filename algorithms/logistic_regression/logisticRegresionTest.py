import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logitsticRegresision import LogisticRegression

breastCancerData = datasets.load_breast_cancer()

samples, labels = breastCancerData.data, breastCancerData.target

samplesTrain, samplesTest, labelsTrain, labelsTest = train_test_split(samples, labels, test_size=0.2, random_state=1234)

def accuracy(labels, labelsPredicted):
    accuracy = np.sum(labels == labelsPredicted)/len(labels)
    return accuracy

regressor = LogisticRegression(learningRate=0.0001, numberOfIterations=1000)
regressor.fit(samplesTrain, labelsTrain) 
predictions = regressor.predict(samplesTest)   

print("Logistic Regresion: ", accuracy(labelsTest, predictions))


import numpy as np

class NaiveBayes:

    def fit(self, samples, labels):
        numberOfSamples, numberOfFeatures = samples.shape
        self._classes = np.unique(labels)
        numberOfClasses = len(self._classes)

        # init mean, variance, priors
        self._mean = np.zeros((numberOfClasses, numberOfFeatures), dtype=np.float64)
        self._variance = np.zeros((numberOfClasses, numberOfFeatures), dtype=np.float64)
        self._priors = np.zeros(numberOfClasses, dtype=np.float64)

        for classItem in self._classes:
             sampleClasses = samples[classItem == labels]
             self._mean[classItem,:] = sampleClasses.mean(axis=0)
             self._variance[classItem,:] = sampleClasses.var(axis=0)
             self._priors[classItem] = sampleClasses.shape[0] / float(numberOfSamples)
             

    def predict(self, samples):
        labelsPredict = [self._predict(sample) for sample in samples]
        return np.array(labelsPredict)

    def _predict(self, sample):
        postiriors = []
        # calculates posterior probability for each class
        for index, c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            classConditional = np.sum(np.log(self._probabilityDencity(index, sample)))
            posterior = prior + classConditional
            postiriors.append(posterior)
        # returns only the class with highest probability
        return self._classes[np.argmax(postiriors)]

    def _probabilityDencity(self, classIndex, sample):
        mean = self._mean[classIndex]
        varience = self._variance[classIndex]
        numerator = np.exp(- ((sample-mean)**2)/(2* varience))
        denominator = np.sqrt(2* np.pi * varience)
        return numerator/denominator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from linearRegression import LinearRegression

samples, labels = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size =0.2, random_state=1234)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(samples[:,0], labels, color = "b", marker ="o", s = 30)
#plt.show()

# cost function for lenear regresion
def meanSqrError(actualValues, predictedValue):
    return np.mean((actualValues - predictedValue) ** 2)

# print(train_samples.shape)
# print(train_labels.shape)

regressor = LinearRegression()
regressor.fit(train_samples, train_labels)

# predicted = regressor.predict(test_samples)
# meanSquereError = meanSqrError(test_labels, predicted)




labelsPredictedLine = regressor.predict(samples)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(train_samples, train_labels, color=cmap(0.9), s=10)
m2 = plt.scatter(test_samples, test_labels, color=cmap(0.5), s=10)
plt.plot(samples, labelsPredictedLine, color="black", linewidth=2, label="Prediction")
meanSquereError = meanSqrError(labels, labelsPredictedLine)
print("Mean square error: ",meanSquereError)
plt.show()
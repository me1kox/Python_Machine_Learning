from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# loading in the iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # assign instances to X; only interested in third and fourth column
y = iris.target  # assign outputs to y

# split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# perform preprocessing by standardization
sc = StandardScaler()
sc.fit(X_train) # fit mean and variation of standard scaler object to training dataset
X_train_std = sc.transform(X_train) # standardize training data set
X_test_std = sc.transform(X_test) # standardize test data set

ppn = Perceptron(eta0=0.1, random_state=1) # instantiate perceptron object; set hyperparameters
ppn.fit(X_train_std, y_train) # update weights using training dataset

y_pred = ppn.predict(X_test_std) # run trained perceptron on test data set and collect outputs
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test)) # display accuracy of predictions on test data set

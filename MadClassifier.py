"""
Build my own super simple Machine Learning Classifier!?!
(Actually used code in the Rashka book, @ location 11%.)

I will create an array where if (e.g.) 0.5*X0 + 0.6*X1 >= 5, Y = 1 else Y = -1

"""
# imports
import numpy as np
import matplotlib.pyplot as plt

# numpy experimentation
# data = np.random.randint(0,9,(50,2))
# print(data[:5])
# print((data*0.5)[:5])
# print(data[:1])
# print(data[:,0][:5])
# target = (data[:,0] * 0.3) + (data[:,1] * 0.6)
# print(target[:5], target.shape)
# print(data.shape, target.shape)


# create a dataset array, first create fixed random seed
np.random.seed(seed=200)
predictors = np.random.randint(0,9,(100,2))

# create target variables
def create_targets(predictors, frac0, frac1):
    return np.add(predictors[:,0] * frac0, predictors[:,1] * frac1)

# make target column as categorical
def make_categorical(targets):
    categories = []
    for target in targets:
        if target < 5:
            categories.append(-1)
        else:
            categories.append(1)
    return np.asarray(categories)

# numpy-esque method? (N.B. temp returns True/False)
def make_categorical2(targets, cutoff):
     temp = targets < cutoff
     return ((temp * 2) - 1) * -1

# merge to one dataset
def create_full_data(predictors, targets):
    return np.column_stack((predictors, targets))

targets = create_targets(predictors, 0.5, 0.6)

cat_targets = make_categorical2(targets, 5)

data = create_full_data(predictors, cat_targets)
print(data[:5])


# now the hard part, making a MadClassifier
class Perceptron(object):
    """
    abc
    """
    def __init__(self, eta=0.01, n_iter=25, random_state=4):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+ X.shape[1])
        self.errors_ = []

        for  _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return (np.dot(X, self.w_[1:]) + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >=0, 1, -1)


# test on the dataset
X = predictors
y = cat_targets
print(X.shape, y.shape)

ppn = Perceptron(eta=0.1, n_iter=35)
ppn.fit(X,y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
# An epoch is a measure of the number of times all of the training vectors
# are used once to update the weights.
plt.ylabel('Number of updates')
plt.show()




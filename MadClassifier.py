"""
Build my own super simple Machine Learning Classifier!?!

I will create an array where if 0.3*X0 + 0.4*X1 >= 10, Y = 1 else Y = -1

"""
# imports
import numpy as np

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
np.random.seed(seed=100)
predictors = np.random.randint(0,9,(50,2))

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








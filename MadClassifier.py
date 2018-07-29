"""
Build my own super simple Machine Learning Classifier!?!

I will create an array where if 0.3*X0 + 0.4*X1 >= 10, Y = 1 else Y = -1

"""
# imports
import numpy as np

# create a dataset array
random_array = np.random.randint(0,16,(4,4))
print(random_array)
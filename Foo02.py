"""
test in pm for Github

"""
# imports
import numpy as np

# play with np arrays
print(np.ones((2,2)))

# Create an array of zeros
print(np.zeros((3,3,3),dtype=np.int16))

# Create an array with random values
print(np.random.random((2,2)))

# Create an empty array, as not zero is slightly faster
print(np.empty((3,2)))

# Create a full array, array of given shape and type filled with fill_value
print(np.full((2,2),"Cheese"))
print(np.full((2,2),123))

# Create an array of evenly-spaced values
print(np.arange(10,40,5))

# Create an array of evenly-spaced values
print(np.linspace(0,2,9))

# Create an array from the “standard normal” distribution
print(np.random.randn(2,2))

# Create an array offset from the “standard normal” distribution
def custom_array(rows, cols, offset):
    """
    Create custom array
    :param rows: # rows
    :param cols: # cols
    :param offset: # 'offset' number added to random outputs
    :return: array of offset number + random part
    """
    base = np.full((rows,cols),offset)
    epsilon = np.random.random((rows,cols))

    return base + epsilon

z = custom_array(5, 1, 3)
print(z)



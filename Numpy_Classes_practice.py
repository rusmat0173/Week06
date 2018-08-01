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

print('==== function here ===')
z = custom_array(3, 1, 0)
print(z)

# Create class for practice. This one checks metrics on a single column array
class metrics:
    """
    Good practice to put a doc string
    I want to practice and do metrics on a column vector
    """
    import numpy as np

    def __init__(self, col_vec):
        self.vector = col_vec
        self.type = type(col_vec)
        self.len = len(col_vec)
        self.shape = col_vec.shape
        self.mean = col_vec.mean(axis=0)
        self.median = col_vec.mean(axis=0)
        self.std = col_vec.std(axis=0)

    def double(self):
        return self.vector * 2

    def sqrt(self):
        return self.vector ** 0.5

    def half(self):
        return self.vector * 0.5

my_col = custom_array(5, 1, 1)

print('==== class here ===')
b = metrics(my_col)

print(b.vector)
print()
print(b.type)
print()
print(b.len)
print()
print(b.shape)
print()
print(b.mean)
print()
print(b.median)
print()
print(b.std)
print()
print(b.double())
print()
print(b.sqrt())
print()
print(b.half())

# foo comments






# Import Libraries
import torch
import numpy as np
import torch.nn as nn



# Creating a Tensor from a Numpy Array
ndarray = np.array([1, 2, 3])
t = torch.from_numpy(ndarray)
print("Tensor: ", t)

print(t.shape)
print(t.dtype)
print(t.device)

ndarray = np.array([[0, 1, 2], 
                    [3, 4, 5]])

tensor_2 = torch.from_numpy(ndarray)

# Index first row:
print("First row: ", tensor_2[0])

# Index first column:
print("First column: ", tensor_2[:, 0])

# Transpose the tensor
print("Transpose: \n", tensor_2.T)

# Multiply the Tensor: (Matrix Multiplication)
ndarray = np.array([[2, 2, 2],
                    [3, 3, 3],
                    [6, 7, 8]])

tensor_3 = torch.from_numpy(ndarray)

print("Multiplication: \n", torch.matmul(tensor_2, tensor_3))

# You can also use the @ operator to perform matrix multiplication:
print("Same thing... \n", tensor_2 @ tensor_3)


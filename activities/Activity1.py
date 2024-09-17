import torch
import numpy as np
import torch.nn as nn


def matrix_multiplication(A: np.ndarray, B: np.ndarray) -> torch.Tensor:
    try:
        tensor1 = torch.from_numpy(A)
        tensor2 = torch.from_numpy(B)
        result = torch.matmul(tensor1, tensor2)
        return result
    except:
        return None

    
# Case 1

A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
expected_output = torch.tensor([[19, 22],
                                [43, 50]])
output = matrix_multiplication(A, B)
print("Multiplication Result for Case 1: \n", output)
assert torch.equal(output, expected_output), f"Test Case 1 Failed: {output}"

# Case 2 (DIff dimension but valid mult)

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11,12]])

expected_output = torch.tensor([[58, 64], [139, 154]])
output = matrix_multiplication(A, B)
print("Multiplication Result for Case 2: \n", output)
assert torch.equal(output, expected_output), f"Test Case 2 Failed: {output}"

# Case 3 (Invalid mult, returns nil)

A = np.array([[1, 2]])
B = np.array([[3, 4],
              [5, 6],
              [7, 8]])
output = matrix_multiplication(A, B)
print("Multiplication Result for Case 3: \n", output)
assert output is None, f"Test Case 3 Failed: {output}"

print("All test cases passed!")
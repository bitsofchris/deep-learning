# Understanding Tensors


The meaning of a tesnsor can change on the context and application.

Tensors basically lists of lists, etc.


2D tensor can be thought of as having dimensions `(rows, columns)`. This is analogous to a matrix in linear algebra, where the first dimension represents the number of rows and the second dimension represents the number of columns.

Matrix multiplication

m x n @ n x p = m x p

### Example

Consider a 2D tensor (matrix) with shape `(3, 4)`:

```python
import torch

# Example 2D tensor with shape (3, 4)
tensor_2d = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("Shape of tensor_2d:", tensor_2d.shape)
```

### Output

```
Shape of tensor_2d: torch.Size([3, 4])
```

### Explanation

- **Rows**: The first dimension (3) represents the number of rows.
- **Columns**: The second dimension (4) represents the number of columns.

### Visual Representation

```
[
    [1, 2, 3, 4],   # Row 1
    [5, 6, 7, 8],   # Row 2
    [9, 10, 11, 12] # Row 3
]
```

### Summary

- **2D Tensor**: A 2D tensor is effectively a matrix with dimensions `(rows, columns)`.
- **Shape**: The shape of a 2D tensor is given by the number of rows and columns.

By understanding this, you can effectively work with 2D tensors (matrices) in PyTorch and other linear algebra contexts.

### Tensors as Nested Lists

1. **Scalars**: A single number, no nesting.
    
    - Example: `3.14`
2. **Vectors**: A 1D tensor, a list of numbers.
    
    - Example: `[1.0, 2.0, 3.0]`
3. **Matrices**: A 2D tensor, a list of lists of numbers.
    
    - Example: `[[1.0, 2.0], [3.0, 4.0]]`
4. **3D Tensors**: A list of lists of lists of numbers.
    
    - Example: `[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]`
5. **Higher-Dimensional Tensors**: Continue this pattern with more levels of nesting.


### PyTorch
You specify what dimension to apply operations to.

```
inputs = torch.tensor(
    [[0.21, 0.47, 0.91], # I
     [0.52, 0.11, 0.65], # can't
     [0.03, 0.85, 0.19], # find
     [0.73, 0.64, 0.39], # the
     [0.13, 0.55, 0.68], # light
     [0.22, 0.77, 0.08]] # switch
)

inputs.shape is (6,3)
6 tokens, of vector length 3
```

dim=0 means first dimension
dim=1 means second dimension of tensor


dim=-1 means last dimension of tensor


---
### General Principles
##### Tensor Dimensions
1. **Batch Size**:
    
    - **Dimension 0**: The first dimension often represents the batch size, which is the number of samples processed together in one forward/backward pass.
    - Example: A tensor with shape `(batch_size, ...)` where `batch_size` is the number of samples in the batch.
2. **Sequence Length**:
    
    - **Dimension 1**: In sequence models (e.g., RNNs, Transformers), the second dimension often represents the sequence length, which is the number of time steps or tokens in the sequence.
    - Example: A tensor with shape `(batch_size, sequence_length, ...)`.
3. **Feature Dimensions**:
    
    - **Last Dimensions**: The last dimensions often represent the features or the feature length. These can include channels, height, width, or other feature dimensions.
    - Example: In image data, a tensor with shape `(batch_size, channels, height, width)`.

##### Tensors
1. **Shape**: The shape of a tensor is a tuple of integers representing the size of the tensor along each dimension.
    
    - Example: A tensor with shape `(2, 3, 4)` has 2 elements in the first dimension, 3 elements in the second dimension, and 4 elements in the third dimension.
2. **Rank**: The rank of a tensor is the number of dimensions it has.
    
    - Example: A tensor with shape `(2, 3, 4)` has a rank of 3.
3. **Indexing**: You can access elements of a tensor using indexing. The number of indices you provide should match the rank of the tensor.
    
    - Example: For a tensor `t` with shape `(2, 3, 4)`, `t[0, 1, 2]` accesses the element at the specified indices.

### Common Tensor Shapes in Deep Learning

1. **Scalars**: A single number, shape `()`.
2. **Vectors**: A 1D tensor, shape `(n,)`.
3. **Matrices**: A 2D tensor, shape `(m, n)`.
4. **Batches of Vectors**: A 2D tensor, shape `(batch_size, n)`.
5. **Batches of Matrices**: A 3D tensor, shape `(batch_size, m, n)`.
6. **Images**: A 4D tensor, shape `(batch_size, channels, height, width)`.

### Summary
- **Shape and Rank**: Understand the shape and rank of tensors to work with them effectively.
- **Indexing**: Use indexing to access elements of tensors.
- **Common Shapes**: Familiarize yourself with common tensor shapes in deep learning.
- **Consistency**: Ensure dimensions are consistent across operations.
- **Broadcasting**: Leverage broadcasting for operations on tensors of different shapes.
- **Transpose and Reshape**: Use `transpose` and `reshape` to change tensor dimensions as needed.
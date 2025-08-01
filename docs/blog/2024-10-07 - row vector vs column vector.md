# 2024-10-07 - row vector vs column vector

I fixed my bug from yesterday, my suspicion was relatively close.

The input activation vector I was using `[0, 0]` was causing an unintended broadcast.

In Numpy the `nd.array([0,0])` is an ambiguous 1D array. This was my mistake and lead to the unintended broadcast operation when I added the `bias` to my `np.dot(weights, acitvations)`.

Numpy will try to interpret how to apply an operation based on what you provide.  In my case the ambiguous 1D array should be a column vector.

So turning my inputs into `nd.array([[0], [0]])` creates a column vector and fixed my issue.


### From GitHub Copilot
A row vector and a column vector are both types of matrices with only one row or one column, respectively. They are used to represent vectors in linear algebra.

### Row Vector
- **Definition**: A row vector is a 1 x n matrix, where n is the number of elements.
- **Shape**: (1, n)
- **Example**: 
  ```python
  row_vector = np.array([[1, 2, 3]])  # Shape: (1, 3)
  ```

### Column Vector
- **Definition**: A column vector is an n x 1 matrix, where n is the number of elements.
- **Shape**: (n, 1)
- **Example**: 
  ```python
  column_vector = np.array([[1], [2], [3]])  # Shape: (3, 1)
  ```


**Usage in Operations**:
   - When performing matrix multiplication, the orientation of vectors matters. For example, multiplying a row vector by a column vector results in a scalar (dot product), while multiplying a column vector by a row vector results in a matrix.

### Left Off
Feedforward is working, lets implement training. 

I am working through the book here http://neuralnetworksanddeeplearning.com/, tomorrow maybe I'll just summarize my Chapter 1 notes.
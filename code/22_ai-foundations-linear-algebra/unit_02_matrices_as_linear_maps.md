# Unit 2 — Matrices as Linear Maps

## Question

What does a matrix do to a vector?

## Cold Attempt

Answer before consuming:

1. What is a matrix?
2. What makes a function linear?
3. Why does knowing where basis vectors go determine the whole transformation?
4. What does X @ W_Q mean geometrically?
5. Why is 'matrix = linear map' such an important sentence?

## Consume

- 3Blue1Brown — Chapter 3: Linear Transformations and Matrices
- Savov — sections on matrices and linear transformations

## Practice

By hand:

1. `T(e1) = (2, 1)`, `T(e2) = (0, 3)`. Write the matrix.
2. Apply it to `(4, 5)`.
3. Write the matrix for a 90° counterclockwise rotation.
4. Write the matrix for reflection across `y = x`.

PyTorch:

```python
# Implement y = x @ W + b without nn.Linear.
# Then implement the batched version: X @ W + b.
# Verify against torch.nn.Linear by copying weights.
```

## Output

Notebook section: **`nn.Linear` without `nn.Linear`**. Add shape comments.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 2 — Matrices as Linear Maps

The core question is:

What does a matrix do to a vector?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 3: Linear Transformations and Matrices
- Savov — sections on matrices and linear transformations

Practice:
By hand:

1. `T(e1) = (2, 1)`, `T(e2) = (0, 3)`. Write the matrix.
2. Apply it to `(4, 5)`.
3. Write the matrix for a 90° counterclockwise rotation.
4. Write the matrix for reflection across `y = x`.

PyTorch:

```python
# Implement y = x @ W + b without nn.Linear.
# Then implement the batched version: X @ W + b.
# Verify against torch.nn.Linear by copying weights.
```

Output:
Notebook section: **`nn.Linear` without `nn.Linear`**. Add shape comments.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is a matrix?
2. What makes a function linear?
3. Why does knowing where basis vectors go determine the whole transformation?
4. What does X @ W_Q mean geometrically?
5. Why is 'matrix = linear map' such an important sentence?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


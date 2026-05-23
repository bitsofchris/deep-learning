# Unit 3 — Matrix Multiplication as Composition

## Question

Why is matrix multiplication function composition?

## Cold Attempt

Answer before consuming:

1. Why does order matter in matrix multiplication?
2. Why is AB usually different from BA?
3. What does (AB)v = A(Bv) mean?
4. How does this relate to stacked neural network layers?
5. What are the shape rules for matrix multiplication?

## Consume

- 3Blue1Brown — Chapter 4: Matrix Multiplication as Composition
- Savov — matrix multiplication

## Practice

By hand:

Compute `AB` and `BA`:

```text
A = [[1, 2],
     [0, 1]]

B = [[1, 0],
     [3, 1]]
```

Then explain why they differ.

PyTorch:

```python
# Implement matmul three ways:
# 1. triple nested loops
# 2. torch.einsum("ij,jk->ik", A, B)
# 3. A @ B

# Use small matrices for the loop version.
# Add shape comments.
```

## Output

Notebook section: **Matrix multiplication as composition**. Include one example of order mattering.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 3 — Matrix Multiplication as Composition

The core question is:

Why is matrix multiplication function composition?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 4: Matrix Multiplication as Composition
- Savov — matrix multiplication

Practice:
By hand:

Compute `AB` and `BA`:

```text
A = [[1, 2],
     [0, 1]]

B = [[1, 0],
     [3, 1]]
```

Then explain why they differ.

PyTorch:

```python
# Implement matmul three ways:
# 1. triple nested loops
# 2. torch.einsum("ij,jk->ik", A, B)
# 3. A @ B

# Use small matrices for the loop version.
# Add shape comments.
```

Output:
Notebook section: **Matrix multiplication as composition**. Include one example of order mattering.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. Why does order matter in matrix multiplication?
2. Why is AB usually different from BA?
3. What does (AB)v = A(Bv) mean?
4. How does this relate to stacked neural network layers?
5. What are the shape rules for matrix multiplication?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


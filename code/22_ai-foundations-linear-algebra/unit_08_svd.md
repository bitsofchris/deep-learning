# Unit 8 — SVD

## Question

How does SVD find the important directions in a matrix?

## Cold Attempt

Answer before consuming:

1. What problem does SVD solve?
2. Why is it different from eigendecomposition?
3. What does low-rank approximation mean?
4. Why might singular values tell us something useful about a weight matrix?
5. Where might SVD show up in ML?

## Consume

- Savov — matrix decompositions / SVD
- fast.ai — Computational Linear Algebra, SVD material only
- Optional: visual SVD explainer if intuition feels missing

## Practice

By hand:

1. For a diagonal matrix, identify singular values by inspection.
2. Show that a rank-1 matrix can be written as an outer product.
3. Given singular values `[10, 3, 1, 0.1]`, explain what a rank-1 vs rank-2 approximation preserves.

PyTorch:

```python
# A can be a grayscale image or synthetic matrix.
U, S, Vh = torch.linalg.svd(A)

# Reconstruct using k singular values.
# Try k = 1, 5, 20, 50.
# Plot or inspect reconstruction error.
```

## Output

Notebook section: **SVD as low-rank approximation.** Include a reconstruction plot, singular value spectrum, or written explanation.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 8 — SVD

The core question is:

How does SVD find the important directions in a matrix?

The unit plan is:

Consume:
- Savov — matrix decompositions / SVD
- fast.ai — Computational Linear Algebra, SVD material only
- Optional: visual SVD explainer if intuition feels missing

Practice:
By hand:

1. For a diagonal matrix, identify singular values by inspection.
2. Show that a rank-1 matrix can be written as an outer product.
3. Given singular values `[10, 3, 1, 0.1]`, explain what a rank-1 vs rank-2 approximation preserves.

PyTorch:

```python
# A can be a grayscale image or synthetic matrix.
U, S, Vh = torch.linalg.svd(A)

# Reconstruct using k singular values.
# Try k = 1, 5, 20, 50.
# Plot or inspect reconstruction error.
```

Output:
Notebook section: **SVD as low-rank approximation.** Include a reconstruction plot, singular value spectrum, or written explanation.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What problem does SVD solve?
2. Why is it different from eigendecomposition?
3. What does low-rank approximation mean?
4. Why might singular values tell us something useful about a weight matrix?
5. Where might SVD show up in ML?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


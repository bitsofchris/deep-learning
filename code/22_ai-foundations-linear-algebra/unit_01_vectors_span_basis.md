# Unit 1 — Vectors, Span, Basis

## Question

How do vectors combine to form spaces, and what is a basis?

## Cold Attempt

Answer before consuming:

1. What is a linear combination?
2. What does it mean for vectors to span a space?
3. What is a basis?
4. Can three vectors in R² be linearly independent?
5. Why does basis matter for embeddings?

## Consume

- 3Blue1Brown — Chapter 2: Linear Combinations, Span, and Basis
- Savov — sections on linear combinations, span, basis, and vector spaces

## Practice

By hand:

1. Is `(5, 7)` in the span of `{(1, 1), (2, 3)}`?
2. Find a basis for the plane `x + y + z = 0` in R³.
3. Show that `{(1, 2), (2, 4)}` does not span R².
4. Express `(3, 5, 2)` using the basis:
   - `(1, 0, 0)`
   - `(1, 1, 0)`
   - `(1, 1, 1)`

PyTorch:

```python
# Put basis vectors as columns of A.
# Solve A @ c = target using torch.linalg.solve.
# Verify that A @ c reconstructs target.
```

## Output

One note or diagram: **Basis = a coordinate system for a vector space.** Include an example where the same vector has different coordinates in two bases.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 1 — Vectors, Span, Basis

The core question is:

How do vectors combine to form spaces, and what is a basis?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 2: Linear Combinations, Span, and Basis
- Savov — sections on linear combinations, span, basis, and vector spaces

Practice:
By hand:

1. Is `(5, 7)` in the span of `{(1, 1), (2, 3)}`?
2. Find a basis for the plane `x + y + z = 0` in R³.
3. Show that `{(1, 2), (2, 4)}` does not span R².
4. Express `(3, 5, 2)` using the basis:
   - `(1, 0, 0)`
   - `(1, 1, 0)`
   - `(1, 1, 1)`

PyTorch:

```python
# Put basis vectors as columns of A.
# Solve A @ c = target using torch.linalg.solve.
# Verify that A @ c reconstructs target.
```

Output:
One note or diagram: **Basis = a coordinate system for a vector space.** Include an example where the same vector has different coordinates in two bases.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is a linear combination?
2. What does it mean for vectors to span a space?
3. What is a basis?
4. Can three vectors in R² be linearly independent?
5. Why does basis matter for embeddings?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


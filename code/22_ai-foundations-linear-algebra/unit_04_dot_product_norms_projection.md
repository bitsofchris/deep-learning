# Unit 4 — Dot Product, Norms, Projection

## Question

Why does the dot product measure alignment?

## Cold Attempt

Answer before consuming:

1. What is a dot product?
2. How is it related to angle?
3. What does orthogonality mean?
4. What is projection?
5. Why does cosine similarity normalize away magnitude?

## Consume

- 3Blue1Brown — Chapter 9: Dot Products and Duality
- Savov — dot product, norms, projections

## Practice

By hand:

1. Compute `u · v` for `u = (1, 2, 3)`, `v = (4, -1, 0)`.
2. Find a vector orthogonal to both `(1, 1, 0)` and `(0, 1, 1)`.
3. Project `(3, 4)` onto `(1, 0)`.
4. Project `(3, 4)` onto `(1, 1) / sqrt(2)`.
5. Explain the geometric difference.

PyTorch:

```python
# Implement manually:
# dot product
# L2 norm
# cosine similarity
# projection of v onto u

# Verify against PyTorch where useful.
```

## Output

Short note: **Dot product as alignment: from geometry to attention.**

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 4 — Dot Product, Norms, Projection

The core question is:

Why does the dot product measure alignment?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 9: Dot Products and Duality
- Savov — dot product, norms, projections

Practice:
By hand:

1. Compute `u · v` for `u = (1, 2, 3)`, `v = (4, -1, 0)`.
2. Find a vector orthogonal to both `(1, 1, 0)` and `(0, 1, 1)`.
3. Project `(3, 4)` onto `(1, 0)`.
4. Project `(3, 4)` onto `(1, 1) / sqrt(2)`.
5. Explain the geometric difference.

PyTorch:

```python
# Implement manually:
# dot product
# L2 norm
# cosine similarity
# projection of v onto u

# Verify against PyTorch where useful.
```

Output:
Short note: **Dot product as alignment: from geometry to attention.**

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is a dot product?
2. How is it related to angle?
3. What does orthogonality mean?
4. What is projection?
5. Why does cosine similarity normalize away magnitude?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


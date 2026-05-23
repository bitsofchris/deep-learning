# Unit 6 — Rank, Nullspace, Low-Rank Structure

## Question

What information does a matrix preserve, compress, or destroy?

## Cold Attempt

Answer before consuming:

1. What is rank?
2. What is column space?
3. What is nullspace?
4. What does it mean for a matrix to be low-rank?
5. Why might low-rank structure matter in ML?

## Consume

- 3Blue1Brown — Chapter 7: Inverse Matrices, Column Space, and Null Space
- Savov — rank, column space, nullspace

## Practice

By hand:

Find rank and nullspace:

```text
A = [[1, 2, 3],
     [2, 4, 6],
     [1, 1, 1]]
```

Also:

1. Construct a 3×3 rank-1 matrix.
2. Construct a 3×3 rank-2 matrix.
3. Explain why `rank(AB) <= min(rank(A), rank(B))`.

PyTorch:

```python
# Construct rank-1 matrix as outer product.
u = torch.randn(5, 1)
v = torch.randn(1, 5)
A = u @ v

# Construct rank-k matrix as sum of k outer products.
# Check torch.linalg.matrix_rank.
# Add tiny noise and compare exact/numerical rank.
```

## Output

Short note: **Rank is the number of independent directions a map preserves or creates.** Add one paragraph connecting this to LoRA or compression.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 6 — Rank, Nullspace, Low-Rank Structure

The core question is:

What information does a matrix preserve, compress, or destroy?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 7: Inverse Matrices, Column Space, and Null Space
- Savov — rank, column space, nullspace

Practice:
By hand:

Find rank and nullspace:

```text
A = [[1, 2, 3],
     [2, 4, 6],
     [1, 1, 1]]
```

Also:

1. Construct a 3×3 rank-1 matrix.
2. Construct a 3×3 rank-2 matrix.
3. Explain why `rank(AB) <= min(rank(A), rank(B))`.

PyTorch:

```python
# Construct rank-1 matrix as outer product.
u = torch.randn(5, 1)
v = torch.randn(1, 5)
A = u @ v

# Construct rank-k matrix as sum of k outer products.
# Check torch.linalg.matrix_rank.
# Add tiny noise and compare exact/numerical rank.
```

Output:
Short note: **Rank is the number of independent directions a map preserves or creates.** Add one paragraph connecting this to LoRA or compression.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is rank?
2. What is column space?
3. What is nullspace?
4. What does it mean for a matrix to be low-rank?
5. Why might low-rank structure matter in ML?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


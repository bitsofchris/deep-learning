# Unit 7 — Change of Basis

## Question

How can the same vector or transformation look different in different coordinates?

## Cold Attempt

Answer before consuming:

1. What is a basis?
2. What does it mean to represent a vector in another basis?
3. What does it mean to represent a transformation in another basis?
4. Why does A' = P^{-1} A P make sense?
5. Why might this matter for the residual stream?

## Consume

- 3Blue1Brown — Chapter 13: Change of Basis
- Savov — change of basis
- Optional skim: privileged basis discussion from Transformer Circuits / Anthropic

## Practice

By hand:

1. Express `(3, 4)` in basis `{(1, 1), (1, -1)}`.
2. Convert it back to standard coordinates.
3. Given:

```text
A = [[2, 1],
     [0, 3]]
```

represent it in basis `{(1, 1), (1, -1)}`.

PyTorch:

```python
# P columns = new basis vectors.
# basis_coords = torch.linalg.solve(P, x)
# x_back = P @ basis_coords
# A_prime = P_inv @ A @ P
# Verify equivalent behavior.
```

## Output

Short note: **Why change of basis matters for transformer interpretability.** Keep it to 2–4 paragraphs.

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 7 — Change of Basis

The core question is:

How can the same vector or transformation look different in different coordinates?

The unit plan is:

Consume:
- 3Blue1Brown — Chapter 13: Change of Basis
- Savov — change of basis
- Optional skim: privileged basis discussion from Transformer Circuits / Anthropic

Practice:
By hand:

1. Express `(3, 4)` in basis `{(1, 1), (1, -1)}`.
2. Convert it back to standard coordinates.
3. Given:

```text
A = [[2, 1],
     [0, 3]]
```

represent it in basis `{(1, 1), (1, -1)}`.

PyTorch:

```python
# P columns = new basis vectors.
# basis_coords = torch.linalg.solve(P, x)
# x_back = P @ basis_coords
# A_prime = P_inv @ A @ P
# Verify equivalent behavior.
```

Output:
Short note: **Why change of basis matters for transformer interpretability.** Keep it to 2–4 paragraphs.

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is a basis?
2. What does it mean to represent a vector in another basis?
3. What does it mean to represent a transformation in another basis?
4. Why does A' = P^{-1} A P make sense?
5. Why might this matter for the residual stream?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


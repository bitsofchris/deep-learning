# Unit 5 — Attention Scores

## Question

How is attention just batched dot products plus routing?

## Cold Attempt

Answer before consuming:

1. What are Q, K, and V?
2. Why compute QK^T?
3. What are the shapes of X, W_Q, W_K, Q, K, and QK^T?
4. Why divide by sqrt(d_head)?
5. What does softmax turn scores into?

## Consume

- Neel Nanda — transformer basics / attention section from mech interp prerequisites
- Optional support: revisit 3Blue1Brown Chapter 9 if dot product intuition is weak

## Practice

By hand:

Use:

```text
T = 3 tokens
d_model = 2
d_head = 2
```

Choose tiny `X`, `W_Q`, and `W_K`.

Compute:

1. `Q = XW_Q`
2. `K = XW_K`
3. `QK^T`
4. identify which token attends most to which token

PyTorch:

```python
# X: [T, d_model]
# W_Q, W_K: [d_model, d_head]
# Q = X @ W_Q
# K = X @ W_K
# scores = Q @ K.T / sqrt(d_head)

# Then batch it:
# X: [B, T, d_model]
# scores: [B, T, T]
# Use einsum.
```

## Output

Shape table:

```text
X:
W_Q:
W_K:
Q:
K:
scores:
```

One sentence: **Attention scores measure which token representations are aligned under learned projections.**

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 5 — Attention Scores

The core question is:

How is attention just batched dot products plus routing?

The unit plan is:

Consume:
- Neel Nanda — transformer basics / attention section from mech interp prerequisites
- Optional support: revisit 3Blue1Brown Chapter 9 if dot product intuition is weak

Practice:
By hand:

Use:

```text
T = 3 tokens
d_model = 2
d_head = 2
```

Choose tiny `X`, `W_Q`, and `W_K`.

Compute:

1. `Q = XW_Q`
2. `K = XW_K`
3. `QK^T`
4. identify which token attends most to which token

PyTorch:

```python
# X: [T, d_model]
# W_Q, W_K: [d_model, d_head]
# Q = X @ W_Q
# K = X @ W_K
# scores = Q @ K.T / sqrt(d_head)

# Then batch it:
# X: [B, T, d_model]
# scores: [B, T, T]
# Use einsum.
```

Output:
Shape table:

```text
X:
W_Q:
W_K:
Q:
K:
scores:
```

One sentence: **Attention scores measure which token representations are aligned under learned projections.**

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What are Q, K, and V?
2. Why compute QK^T?
3. What are the shapes of X, W_Q, W_K, Q, K, and QK^T?
4. Why divide by sqrt(d_head)?
5. What does softmax turn scores into?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


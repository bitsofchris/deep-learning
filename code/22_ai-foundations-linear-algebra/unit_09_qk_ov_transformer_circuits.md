# Unit 9 — QK / OV / Transformer Circuits Bridge

## Question

Where do the linear algebra ideas show up in transformer circuits?

## Cold Attempt

Answer before consuming:

1. What does QK do?
2. What does OV do?
3. What is an outer product?
4. What is a composed linear map?
5. What does the residual stream carry?

## Consume

- Transformer Circuits — A Mathematical Framework for Transformer Circuits: residual stream, attention heads, QK circuits, OV circuits, virtual weights
- Support: Neel Nanda walkthrough of Transformer Circuits for only those sections

## Practice

By hand:

1. Compute an outer product:

```text
u = [1, 2, 3]
v = [4, 5, 6]
u v^T = ?
```

2. Show it is rank 1.
3. For tiny matrices, compute:
   - `W_QK = W_Q W_K^T`
   - `W_OV = W_V W_O`

PyTorch:

```python
# outer product
torch.einsum("i,j->ij", u, v)

# QK / OV toy compositions
W_QK = W_Q @ W_K.T
W_OV = W_V @ W_O
```

## Output

One-page stress-test note:

```markdown
# Transformer Circuits Stress Test

## What clicked

## What bounced

## Where linear algebra showed up

## What to revisit after nanoGPT
```

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 9 — QK / OV / Transformer Circuits Bridge

The core question is:

Where do the linear algebra ideas show up in transformer circuits?

The unit plan is:

Consume:
- Transformer Circuits — A Mathematical Framework for Transformer Circuits: residual stream, attention heads, QK circuits, OV circuits, virtual weights
- Support: Neel Nanda walkthrough of Transformer Circuits for only those sections

Practice:
By hand:

1. Compute an outer product:

```text
u = [1, 2, 3]
v = [4, 5, 6]
u v^T = ?
```

2. Show it is rank 1.
3. For tiny matrices, compute:
   - `W_QK = W_Q W_K^T`
   - `W_OV = W_V W_O`

PyTorch:

```python
# outer product
torch.einsum("i,j->ij", u, v)

# QK / OV toy compositions
W_QK = W_Q @ W_K.T
W_OV = W_V @ W_O
```

Output:
One-page stress-test note:

```markdown
# Transformer Circuits Stress Test

## What clicked

## What bounced

## Where linear algebra showed up

## What to revisit after nanoGPT
```

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What does QK do?
2. What does OV do?
3. What is an outer product?
4. What is a composed linear map?
5. What does the residual stream carry?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


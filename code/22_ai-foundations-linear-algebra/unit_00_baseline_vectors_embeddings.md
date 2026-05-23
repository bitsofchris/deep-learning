# Unit 0 — Baseline: Vectors and Embeddings

## Question

What is a vector, and why are embeddings vector representations?

## Cold Attempt

Answer before consuming:

1. What is a vector?
2. What is a token embedding?
3. Why can an embedding be treated as a vector?
4. What does it mean for a vector to have coordinates?
5. What feels fuzzy?

## Consume

- 3Blue1Brown — Essence of Linear Algebra, Chapter 1: Vectors
- Optional: Savov, No Bullshit Guide to Linear Algebra — vector basics

## Practice

By hand:

```text
v = (2, -1, 3)
w = (0, 4, -2)
```

Compute:

1. `v + w`
2. `3v`
3. `v - 2w`
4. `v · w`
5. `||v||`

Optional PyTorch:

```python
import torch

v = torch.tensor([2., -1., 3.])
w = torch.tensor([0., 4., -2.])

v_plus_w = ...
three_v = ...
v_minus_2w = ...
dot_vw = ...
v_norm_manual = ...
```

Do `v_norm_manual` without `torch.linalg.norm`.

## Output

Short note:

```markdown
# What is a vector representation?

## My cold explanation

## What changed after studying

## Why embeddings are vectors

## What is still fuzzy
```

## LLM Kickoff Prompt

Paste this into a new chat when starting this unit:

```text
I am doing a six-week AI foundations sprint. Right now I am in the linear algebra block. My goal is not to passively consume content; it is to build foundations for understanding transformers, PyTorch, attention, residual streams, and eventually mechanistic interpretability.

For this unit, I am learning:

Unit 0 — Baseline: Vectors and Embeddings

The core question is:

What is a vector, and why are embeddings vector representations?

The unit plan is:

Consume:
- 3Blue1Brown — Essence of Linear Algebra, Chapter 1: Vectors
- Optional: Savov, No Bullshit Guide to Linear Algebra — vector basics

Practice:
By hand:

```text
v = (2, -1, 3)
w = (0, 4, -2)
```

Compute:

1. `v + w`
2. `3v`
3. `v - 2w`
4. `v · w`
5. `||v||`

Optional PyTorch:

```python
import torch

v = torch.tensor([2., -1., 3.])
w = torch.tensor([0., 4., -2.])

v_plus_w = ...
three_v = ...
v_minus_2w = ...
dot_vw = ...
v_norm_manual = ...
```

Do `v_norm_manual` without `torch.linalg.norm`.

Output:
Short note:

```markdown
# What is a vector representation?

## My cold explanation

## What changed after studying

## Why embeddings are vectors

## What is still fuzzy
```

I want you to act as a Socratic tutor and pre-check examiner.

Start by asking me the cold-attempt questions below, one at a time or in a small batch. Do not teach or give answers before I answer.

Cold-attempt questions:
1. What is a vector?
2. What is a token embedding?
3. Why can an embedding be treated as a vector?
4. What does it mean for a vector to have coordinates?
5. What feels fuzzy?

After I answer, do three things:

1. Record my answers cleanly under a heading called "My Cold Attempt".
2. Give corrections or missing nuance under a heading called "Corrections / Gaps".
3. Give me a short "What to look for while consuming" section so I know what to pay attention to when I study.

Do not spoon-feed me. Push on vague language. Ask follow-up questions when my answer is hand-wavy. Keep me doing the work.
```


# Notes


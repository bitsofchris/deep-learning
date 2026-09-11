# Unit 7 — GPT: attention from scratch

Lecture: https://www.youtube.com/watch?v=kCc8FmEb1nY (1h56m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

How does a token gather information from the tokens before it, and only those?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. A batch of logits has shape `(B, T, C)` and the targets are `(B, T)`. What has to happen to both before cross entropy can be computed, and what is being averaged over?
2. You want every position `t` to hold the mean of positions `0..t`. Write that as a single matrix multiply. What does the matrix look like, and where does the `-inf` come in when you build it with softmax instead?
3. In one head of self-attention, what are the key, the query and the value, in words? Which two of them are multiplied together, and what shape is the result for one batch row?
4. Why is `q @ k.T` divided by `sqrt(head_size)` before the softmax? What goes wrong, numerically, if you don't?
5. A transformer block has two residual connections and two LayerNorms. Where exactly does each one sit, and why does the residual path matter for a deep stack?
6. Attention is a set operation. What does that mean, and what does the model add so it still knows where each token is?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–38:00 data, character tokenizer, batches of chunks, the bigram baseline, its loss and `generate`, training it
  - 38:00–62:00 port to a script; the mathematical trick: averaging the past with loops, then tril matmul, then softmax with `-inf`; positional encoding
  - 62:00–1:19:11 **the crux**: version 4, self-attention; the six notes (communication, no notion of space, no cross-batch talk, encoder vs decoder, self vs cross, scaling)
  - 1:19:11–1:37:49 inserting a head; multi-head; feed-forward; residual connections; LayerNorm
  - 1:37:49–end scaling up, dropout, encoder/decoder, nanoGPT walkthrough, ChatGPT and RLHF
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`gpt.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | bigram `nn.Module`: `(B,T,C)` logits meet `(B,T)` targets; `generate` crops to `block_size` | 1 |
| 2 | the trick: averaging the past by loop, by tril matmul, by softmax with `-inf` | 2 |
| 3 | **one self-attention Head**: k/q/v, scaled dot product, causal mask, softmax, weighted values | 3 |
| 4 | MultiHeadAttention (concat + proj) and the per-token FeedForward | 4 |
| 5 | the Block with residuals + pre-LN, and the full GPT with token + position embeddings | 4 |
| 6 | it learns: val loss on tinyshakespeare beats anything a bigram could reach | 5 |
| 7 | *stretch:* LayerNorm from scratch, graded against `nn.LayerNorm`; dropout placement | 4–5 |

Boilerplate, given fully written in the notebook: reading the file, the character tokenizer,
train/val split, `get_batch`, `estimate_loss`, the training loop, and the sampling step of
`generate`. Everything else is yours.

Launch from this folder:

```
../../.venv/bin/jupyter lab --notebook-dir=.
```

Rules: no lecture open while coding, no nanoGPT repo. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.
Every training run in this unit is under a minute on CPU.

## Review

`quiz_07_gpt.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own training loop and sampler**, from memory, on a small GPT. Read the text it produces.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 7, "Let's build GPT: from scratch, in code, spelled out." The folder
code/23_zero-to-hero/07_gpt/ in this repo already contains a notebook exercise
(gpt.ipynb) and a grader (test_gpt.py). Do not rebuild any of that.

The core question for this unit is: how does a token gather information from the
tokens before it, and only those?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. A batch of logits has shape (B, T, C) and the targets are (B, T). What has to
   happen to both before cross entropy can be computed, and what is being averaged over?
2. You want every position t to hold the mean of positions 0..t. Write that as a
   single matrix multiply. What does the matrix look like, and where does the -inf
   come in when you build it with softmax instead?
3. In one head of self-attention, what are the key, the query and the value, in
   words? Which two are multiplied together, and what shape is the result for one
   batch row?
4. Why is q @ k.T divided by sqrt(head_size) before the softmax? What goes wrong,
   numerically, if you don't?
5. A transformer block has two residual connections and two LayerNorms. Where
   exactly does each one sit, and why does the residual path matter for a deep stack?
6. Attention is a set operation. What does that mean, and what does the model add so
   it still knows where each token is?

After I answer, do three things:
1. Record my answers under "My Cold Attempt".
2. Corrections or missing nuance under "Corrections / Gaps".
3. A short "What to look for while watching" list.

Then switch to coaching mode for the notebook. Rules for that mode:
- Don't explain. If I'm stuck I'll tell you what I tried and what I expected;
  give me ONE hint, tiered: first a question, then the shape of the idea, and
  only something close to the answer if I ask a third time. One tier per ask.
- Before I run a grader cell for the first time on a milestone, ask me to
  predict what it will print.
- If I ask a PyTorch or Python syntax question, just answer it.
- If I say "just tell me", tell me.
- Push on vague language. Keep me doing the work.
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one
  bullet per idea, only after I have said it back correctly in my own words,
  never when you explained it. Record the wrong model I had and the correction
  that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone /
  what I was doing>` to the unit note's Notes, so a later session can resume
  from it. When resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in quiz_07_gpt.md (+1 day,
+4 days, +2 weeks from today) and rewrite its questions so they target the gaps
that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

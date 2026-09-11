# Unit 5 — makemore part 4: becoming a backprop ninja

Lecture: https://www.youtube.com/watch?v=q8SA3rM6ckI (1h55m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

Can you compute every gradient in the MLP by hand, at the tensor level, and match autograd exactly?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. `loss = -logprobs[range(n), Yb].mean()`. What is `dloss/dlogprobs`, entry by entry? What is it at the entries that were never indexed?
2. `b1` has shape `(64,)` and is added to a `(32, 64)` tensor. What shape is `db1`, and what operation gets you from the `(32, 64)` gradient to it? Why that operation and not another?
3. `counts` is used twice in the forward pass: once to make `counts_sum`, once to make `probs`. How do the two contributions combine into `dcounts`?
4. `c = a @ b`. Write `dL/da` and `dL/db` in terms of `dL/dc`, `a`, `b`. How do you know which side each goes on and which gets transposed, without memorizing?
5. `logit_maxes` is subtracted from the logits before the exp. Does it change the loss? What should its gradient be, and what will you actually get for it, numerically?
6. Softmax followed by cross-entropy has a very short backward expression. What is it, and in words, what does it push up and what does it push down?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–13:01 why bother (the leaky abstraction, the vanishing/exploding stories), and the starter code: the forward pass broken into named intermediates, `cmp`
  - 13:01–65:17 exercise 1, the atomic backward pass. Long. Stop and do milestones 1–5 in lockstep: watch him do a group, pause, do the same group in the notebook, then watch the next group only if you got stuck or want to compare
  - 65:17–96:37 the Bessel's-correction digression, then exercise 2: deriving `dlogits` for softmax + cross-entropy on paper
  - 96:37–110:02 exercise 3: deriving the single-expression batchnorm backward on paper
  - 110:02–end exercise 4: the training loop with manual gradients, and the outro
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`makemore_backprop_ninja.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

Launch from this folder:

```
../../.venv/bin/jupyter lab --notebook-dir=.
```

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | loss → logprobs → probs → counts; the first broadcast sum, the first tensor used twice | 2 |
| 2 | the softmax: norm_logits, logit_maxes, logits; a gradient that "should" be zero | 2 |
| 3 | second linear layer and tanh; matmul backward from shapes alone | 2 |
| 4 | the batchnorm chain, nine tensors; the wall | 2 |
| 5 | first linear layer and the embedding lookup; scatter-add into `C` | 2 |
| 6 | *stretch:* the fused shortcuts, `dlogits` and `dhprebn` in one expression each | 3, 4 |
| 7 | *stretch:* train the MLP using only your gradients, and it learns | 5 |

Boilerplate, given: the dataset, the parameter init, the forward pass split into ~20 named
tensors with `retain_grad`, the `cmp` helper, the grader's training loop. The lesson: every
line of the backward pass.

Rules: no lecture open while coding, no reference notebook. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_05_makemore_backprop_ninja.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own training loop and sampling loop**, from memory,
  with no autograd anywhere.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what
  the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 5, "Building makemore Part 4: Becoming a Backprop Ninja". The folder
code/23_zero-to-hero/05_makemore_backprop_ninja/ in this repo already contains a
notebook exercise (makemore_backprop_ninja.ipynb) and a grader
(test_makemore_backprop_ninja.py). Do not rebuild any of that.

The core question for this unit is: can I compute every gradient in the MLP by
hand, at the tensor level, and match autograd exactly?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. loss = -logprobs[range(n), Yb].mean(). What is dloss/dlogprobs, entry by entry? What is it at the entries that were never indexed?
2. b1 has shape (64,) and is added to a (32, 64) tensor. What shape is db1, and what operation gets you from the (32, 64) gradient to it? Why that operation?
3. counts is used twice in the forward pass: once to make counts_sum, once to make probs. How do the two contributions combine into dcounts?
4. c = a @ b. Write dL/da and dL/db in terms of dL/dc, a, b. How do you know which side each goes on and which gets transposed, without memorizing?
5. logit_maxes is subtracted from the logits before the exp. Does it change the loss? What should its gradient be, and what will you actually get for it, numerically?
6. Softmax followed by cross-entropy has a very short backward expression. What is it, and in words, what does it push up and what does it push down?

After I answer, do three things:
1. Record my answers under "My Cold Attempt".
2. Corrections or missing nuance under "Corrections / Gaps".
3. A short "What to look for while watching" list.

Then switch to coaching mode for the notebook. Rules for that mode:
- Don't explain. If I'm stuck I'll tell you what I tried and what I expected;
  give me ONE hint, tiered: first a question, then the shape of the idea, and
  only something close to the answer if I ask a third time. One tier per ask.
  Never hand me a gradient formula unless I say "just tell me".
- Before I run a grader cell for the first time on a milestone, ask me to
  predict what it will print.
- If I ask a PyTorch syntax question (keepdim, one_hot, index_add_, view vs
  reshape, .T), just answer it.
- If I say "just tell me", tell me.
- Push on vague language. "The gradient flows back" is not an answer; make me
  say what shape it is and what it is multiplied by.
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one
  bullet per idea, only after I have said it back correctly in my own words,
  never when you explained it. Record the wrong model I had and the correction
  that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone /
  what I was doing>` to the unit note's Notes, so a later session can resume
  from it. When resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in
quiz_05_makemore_backprop_ninja.md (+1 day, +4 days, +2 weeks from today) and
rewrite its questions so they target the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

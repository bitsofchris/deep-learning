# Unit 6 — makemore part 5: building a WaveNet

Lecture: https://www.youtube.com/watch?v=t3YJ5hKiMQ0 (56 min)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

How do you fuse context gradually instead of squashing all of it in one layer?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. The lecture-3 MLP concatenates the embeddings of all context characters and hands them to one `Linear`. What goes wrong with that design when the context grows from 3 characters to 8?
2. You have a tensor of shape `(B, T, C)`, batch × positions × channels. You want each new position to hold *two consecutive* input positions side by side. What is the output shape, and what single torch call gets there without copying data? Why does that call produce the right adjacency?
3. `x @ W` where `x` is `(B, T, C_in)` and `W` is `(C_in, C_out)`. Does it run? What comes out, and what did matmul do with the extra dimension?
4. Your BatchNorm1d takes the mean over `dim=0`. Feed it `(B, T, C)` instead of `(B, C)`. Does it crash? What does it silently compute, and which dims *should* the mean be over?
5. What is the minimal interface a class needs so that a `Sequential` container can treat Embedding, Flatten, Linear, BatchNorm1d and Tanh identically? Name the methods.
6. `FlattenConsecutive(2)` followed by a `Linear` applies the same weights to every pair of positions. What standard layer is that equivalent to, and what does the real WaveNet change about it (hint: stride vs. dilation)?

## Consume

The lecture, in sections. Stop after each and try the matching milestones.

- 0:00–9:16 intro, starter code walkthrough, fixing the learning-rate/loss plot
- 9:16–17:11 pytorchifying the code: layers, containers, torch.nn, "fun bugs"
- 17:11–37:41 WaveNet overview, block_size 8, re-running the baseline, implementing the hierarchy (FlattenConsecutive)
- 37:41–46:07 training the WaveNet, the BatchNorm1d bug, re-training with the fix
- 46:07–end scaling up, the experimental harness, dilated causal convolutions, torch.nn, the development process

## Practice

`makemore_wavenet.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

Launch from this folder:

```bash
../../.venv/bin/jupyter lab --notebook-dir=.
```

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | Embedding and Flatten as modules (`__call__`, `parameters`) | 2 |
| 2 | the Sequential container | 2 |
| 3 | FlattenConsecutive(n): fuse n neighbours, the `.view` gymnastics, the squeeze | 3 |
| 4 | BatchNorm1d on a 3-D input, checked against `nn.BatchNorm1d` | 4 |
| 5 | assemble the hierarchical net for block_size 8 | 3–4 |
| 6 | it learns: dev loss under a threshold after a short run | 4 |
| 7 | *stretch:* the Linear after FlattenConsecutive is a strided conv; reshape its weight for `F.conv1d` | 5 |

Boilerplate, given fully written: the block_size-8 dataset build, `Linear` and `Tanh` from lecture 4, `BatchNorm1d.__init__`, the training loop, the chunk-averaged loss plot, eval-mode switching, and the sampling loop.

Rules: no lecture open while coding, no makemore repo. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_06_makemore_wavenet.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own training loop**, from memory, with the lecture-sized model (`n_embd=24, n_hidden=128`), including the eval-mode switch before measuring dev loss.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 6, "Building makemore Part 5: Building a WaveNet". The folder
code/23_zero-to-hero/06_makemore_wavenet/ in this repo already contains a notebook
exercise (makemore_wavenet.ipynb) and a grader (test_makemore_wavenet.py). Do not
rebuild any of that.

The core question for this unit is: how do you fuse context gradually instead of
squashing all of it in one layer?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. The lecture-3 MLP concatenates the embeddings of all context characters and hands
   them to one Linear. What goes wrong with that design when the context grows from
   3 characters to 8?
2. You have a tensor of shape (B, T, C). You want each new position to hold two
   consecutive input positions side by side. What is the output shape, and what
   single torch call gets there without copying? Why does it give the right adjacency?
3. x @ W where x is (B, T, C_in) and W is (C_in, C_out). Does it run? What comes out,
   and what did matmul do with the extra dimension?
4. Your BatchNorm1d takes the mean over dim=0. Feed it (B, T, C) instead of (B, C).
   Does it crash? What does it silently compute, and which dims should the mean be over?
5. What is the minimal interface a class needs so that a Sequential container can
   treat Embedding, Flatten, Linear, BatchNorm1d and Tanh identically? Name the methods.
6. FlattenConsecutive(2) followed by a Linear applies the same weights to every pair
   of positions. What standard layer is that equivalent to, and what does the real
   WaveNet change about it?

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
- If I ask a Python or torch syntax question, just answer it.
- If I say "just tell me", tell me.
- Push on vague language. Keep me doing the work.
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one
  bullet per idea, only after I have said it back correctly in my own words,
  never when you explained it. Record the wrong model I had and the correction
  that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone /
  what I was doing>` to the unit note's Notes, so a later session can resume
  from it. When resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in
quiz_06_makemore_wavenet.md (+1 day, +4 days, +2 weeks from today) and rewrite
its questions so they target the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

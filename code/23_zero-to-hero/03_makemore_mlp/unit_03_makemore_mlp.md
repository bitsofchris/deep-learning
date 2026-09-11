# Unit 3 — makemore part 2: the MLP

Lecture: https://www.youtube.com/watch?v=TCH_1BHY58I (1h15m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

How does a network see more than one character of context without the table exploding?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. A bigram table has 27×27 entries. How many rows would a lookup table need for three characters of context, and why is that the wrong direction?
2. Take the name `emma` and a context window of 3. Write out every (context, target) pair the model trains on. How many are there, and what does the very first context look like?
3. `C` is a (27, 2) table and `X` is a (32, 3) matrix of character indices. What is the shape of `C[X]`, and what does each entry mean?
4. The hidden layer wants one flat vector per example. Two reshapes can produce the same (32, 6) shape with different numbers inside. What decides whether a `.view` is "right", and what does it cost in memory?
5. Write cross-entropy from logits in three or four operations, without calling a library function. What goes wrong numerically if a logit is around +100?
6. Why train on a random minibatch instead of the whole dataset each step, and what is the price? How would you *pick* a learning rate rather than guess it?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–18:35 the Bengio paper, rebuilding the dataset with `block_size`, the embedding table `C` and `C[X]`
  - 18:35–32:49 the hidden layer, tensor storage and views, output layer, the loss by hand, the whole net in one screen
  - 32:49–53:20 `F.cross_entropy` and why, overfitting one batch, minibatches, finding a learning rate
  - 53:20–end train/dev/test and why, bigger hidden layer, visualising the embedding, bigger embedding, sampling
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`makemore_mlp.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | `build_dataset`: the sliding window, the padding, the off-by-one | 1 |
| 2 | `embed`: what `C[X]` means and what shape it has | 1 |
| 3 | `flatten_context`: `(N, T, d)` → `(N, T·d)` without scrambling, without copying | 2 |
| 4 | `forward`: embed → hidden tanh → logits | 2 |
| 5 | `cross_entropy` by hand, equal to torch's, also on nasty logits | 2–3 |
| 6 | `train`: the minibatch loop, and it learns (dev loss threshold) | 3–4 |
| 7 | *stretch:* `sample` names from the model | 4 |

Given as boilerplate: reading the names, `stoi`/`itos`, `init_params` (the parameter dict and its shapes), the shuffled 80/10/10 split, `split_loss`, the `lrs` sweep grid, and the embedding plot. The learning-rate sweep itself is the Output.

Launch from this folder:

```
../../.venv/bin/jupyter lab --notebook-dir=.
```

Rules: no lecture open while coding, no makemore repo. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_03_makemore_mlp.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own learning-rate sweep**, from memory. Pick an lr from the plot, train longer with it, decay at the end, report train/dev/test loss (predict their order first), plot the 2-d embedding, sample ten names.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 3, "Building makemore Part 2: MLP". The folder
code/23_zero-to-hero/03_makemore_mlp/ in this repo already contains a notebook
exercise (makemore_mlp.ipynb) and a grader (test_makemore_mlp.py). Do not
rebuild any of that.

The core question for this unit is: how does a network see more than one
character of context without the table exploding?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. A bigram table has 27x27 entries. How many rows would a lookup table need
   for three characters of context, and why is that the wrong direction?
2. Take the name "emma" and a context window of 3. Write out every
   (context, target) pair the model trains on. How many are there, and what
   does the very first context look like?
3. C is a (27, 2) table and X is a (32, 3) matrix of character indices. What
   is the shape of C[X], and what does each entry mean?
4. The hidden layer wants one flat vector per example. Two reshapes can
   produce the same (32, 6) shape with different numbers inside. What decides
   whether a .view is "right", and what does it cost in memory?
5. Write cross-entropy from logits in three or four operations, without
   calling a library function. What goes wrong numerically if a logit is
   around +100?
6. Why train on a random minibatch instead of the whole dataset each step, and
   what is the price? How would you pick a learning rate rather than guess it?

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

When the unit finishes: fill in the three review dates in quiz_03_makemore_mlp.md
(+1 day, +4 days, +2 weeks from today) and rewrite its questions so they target
the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)


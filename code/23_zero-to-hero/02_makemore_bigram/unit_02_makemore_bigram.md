# Unit 2 — makemore, part 1: the bigram model

Lecture: https://www.youtube.com/watch?v=PaCmpygFfXo (1h58m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

How do you turn "predict the next character" into a number you can minimize?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. A bigram model of names is just a table. What are its rows, what are its columns, and what is in each cell?
2. You have the table of counts. What do you do to one row to turn it into something you can sample from, and why does `keepdim` (or its absence) matter when you do it for every row at once?
3. The model gives each actual next character in the dataset a probability. Combine those into ONE number that is small when the model is good. Why log, why negative, why average?
4. A neural net can't eat the integer `5`. What do you feed it instead, and what does multiplying that by a weight matrix `W` actually do to `W`?
5. The net outputs 27 real numbers per input, some negative. What two operations turn them into a probability distribution, and what do the outputs of the first one resemble?
6. Write the four lines of a gradient-descent step in PyTorch. Which line is easy to forget, and what goes wrong when you forget it?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–36:17 the dataset, counting bigrams into a dict then a 27×27 tensor, the `.` boundary token, sampling from the rows with `torch.multinomial`
  - 36:17–62:57 broadcasting and row normalization, the negative log likelihood as the loss, smoothing with fake counts
  - 62:57–95:49 the neural approach: the `(xs, ys)` dataset, one-hot encoding, one linear layer, softmax as exp-then-normalize
  - 95:49–end vectorized loss, `backward()` and the update, everything together, one-hot as row selection, the `W**2` penalty as smoothing, sampling from the net
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`makemore_bigram.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | the bigram count table with the boundary token | 1 |
| 2 | rows to probabilities (broadcasting); seeded sampling must match a reference | 1–2 |
| 3 | negative log likelihood: the number to minimize | 2 |
| 4 | one-hot → linear layer → softmax by hand, checked against torch's values and gradient | 3 |
| 5 | the gradient-descent loop; the loss lands within 0.05 of the counting model's | 4 |
| 6 | the net as a learned count table; `log(P)` in gives the counting model's names out | 4 |
| 7 | *stretch:* smoothing and the `W**2` penalty are the same idea | 2, 4 |

Launch from this folder:

```
../../.venv/bin/jupyter lab --notebook-dir=.
```

Rules: no lecture open while coding, no makemore repo, no lecture notebook. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_02_makemore_bigram.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own train-and-sample loop**, from memory, printing the neural net's names next to the counting model's names on the same seed, with a comment on why they do or don't match.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 2, makemore part 1 (the bigram model). The folder
code/23_zero-to-hero/02_makemore_bigram/ in this repo already contains a notebook
exercise (makemore_bigram.ipynb) and a grader (test_makemore_bigram.py). Do not
rebuild any of that.

The core question for this unit is: how do you turn "predict the next character"
into a number you can minimize?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. A bigram model of names is just a table. What are its rows, its columns, and
   what is in each cell?
2. You have the table of counts. What do you do to one row to turn it into
   something you can sample from, and why does keepdim matter when you do it for
   every row at once?
3. The model gives each actual next character a probability. Combine those into
   ONE number that is small when the model is good. Why log, why negative, why
   average?
4. A neural net can't eat the integer 5. What do you feed it instead, and what
   does multiplying that by a weight matrix W actually do to W?
5. The net outputs 27 real numbers per input, some negative. What two operations
   turn them into a probability distribution, and what do the outputs of the
   first one resemble?
6. Write the four lines of a gradient-descent step in PyTorch. Which line is easy
   to forget, and what goes wrong when you forget it?

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
- If I ask a PyTorch or Python syntax question (broadcasting rules, indexing,
  what keepdim does mechanically), just answer it.
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
quiz_02_makemore_bigram.md (+1 day, +4 days, +2 weeks from today) and rewrite
its questions so they target the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

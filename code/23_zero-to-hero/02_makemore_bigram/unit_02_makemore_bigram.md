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

`quiz_02_makemore_bigram.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log. Reviews are scheduled in the vault note `Quiz Schedule - Foundations` (the OpenAugi board puts a due review on that day's board); run one with Prompt B in `../PROMPT.md`.

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
Then append three rows to the OpenAugi quiz schedule, the vault note
~/Documents/ZK Home/_private/3-MOCs and Projects/Quiz Schedule - Foundations.md
(unit 02, quiz path, review 1/2/3, due +1d/+4d/+14d from today) so the daily
board fires each review.
```


### My Cold Attempt (2026-09-22, after watching once)

1. Rows and columns are each a letter plus the special token. Row = first letter, column = the letter that comes after it. Cell = the count of that pair.
2. "Use some function" to turn a row into a distribution you can sample from. `keepdim` matters "because otherwise it would be collapsing certain dimensions."
3. It's the loss, the negative log likelihood. If we predict the correct next letter the loss is small. Don't know why log. Negative "because we want the number to go down." Not sure why average.
4. One-hot: a vector where each position is a class. Multiplying by W gives "a vector of weights or probabilities for that category, if there's one layer."
5. Softmax: exponent, then a normalization step. Didn't follow "what do the outputs of the first one resemble."
6. Zero grads (easy to forget; otherwise gradients accumulate across steps), backward, then step against the gradient scaled by the learning rate on each parameter. Named three lines, not four.

### Corrections / Gaps

- **Q1 is right.** Rows are the *previous* char, columns the *next* char, plus `.` for start/end. Hold onto "row = previous"; the notebook will ask which axis is which.
- **Q2, two gaps.** (a) "Some function" is just *divide the row by its sum*; there is no library call. (b) `keepdim` is not about collapsing in general. `N.sum(1)` has shape `(27,)`; `N.sum(1, keepdim=True)` has shape `(27, 1)`. Broadcasting aligns trailing dims, so `N / N.sum(1)` divides *columns* by row sums (each column j divided by the sum of row j), which is silently wrong and still runs. With `(27, 1)` each row is divided by its own sum. Be able to say which wrong thing gets computed, not just that it "collapses."
- **Q3 is the unit question and was the weakest.** You had "loss = NLL" as a name but not the reason for any of its three pieces:
  - *Why log:* the likelihood of the whole dataset is a *product* of per-character probabilities, which underflows to 0 for thousands of names. Log turns the product into a sum.
  - *Why negative:* log of a probability is between −∞ and 0, so higher likelihood means log closer to 0. Negating makes "good" small and gives a quantity to *minimize*. "Because we want it to go down" is the effect, not the reason.
  - *Why average:* the sum grows with dataset size, so it can't be compared across datasets or batches. The mean is per-character, and the uniform model's mean is exactly `log(27)`.
- **Q4 half right.** The one-hot part is fine. But `one_hot @ W` is not "a vector of probabilities." It is *row 5 of W* selected out, unchanged. The matmul is an indexing operation. The row is logits, not probabilities, until Q5's two steps happen.
- **Q5 partial.** The two operations are `exp` then divide by the row sum. What `exp(logits)` resembles is the *count table* from milestone 1: positive numbers that get normalized the same way. That is the link between the counting model and the net, and it is the whole point of milestones 4 and 6.
- **Q6 mostly right, one line missing.** Four lines: forward (compute loss), `W.grad = None`, `loss.backward()`, `W.data += -lr * W.grad`. You had the zero-grad consequence right. Forgotten-line effect at `lr=50`: the accumulated gradient makes the update grow every step, so the loss wobbles or blows up.

### What to look for while watching

You've watched once. On the rewatch or skim of these timestamps, check specifically:

- ~40:00 the `keepdim` broadcasting bug. Karpathy shows the wrong version. Say out loud what the wrong version divides by.
- ~50:00 the likelihood section. Watch the product of probabilities collapse, then the log, then the negation, then the mean. Each step should answer one of the three "why"s.
- ~1:05:00 one-hot times W. Watch him point at the one row of W that comes out.
- ~1:15:00 "we interpret the logits as log-counts." Then `exp` gives counts and the divide gives probs. This is Q5's "resemble."
- ~1:30:00 the training loop. Count the lines. Notice `W.grad = None` before `backward`.
- ~1:50:00 smoothing versus the `W**2` regularization term. Only for milestone 7.

## Notes

### Coaching log (things I worked out from my own questions)

- **M2, the denominator.** Wrong model: divide every count by the grand total of N ("the full corpus"). Correction, in my words: given letter i, I'm predicting j, so each row has to sum to 1 on its own; sum the counts row-wise and divide each count in a row by that row's sum. The sampling loop draws from one row alone, which is why the whole-table total is wrong.
- **M2, keepdim.** Wrong model: `keepdim` "stops it collapsing dimensions"; and "nothing fails without it so it doesn't matter." Correction: `N.sum(1)` without keepdim is a `(27,)` vector, which broadcasts as a *row* `(1, 27)` and divides column j by row j's sum, silently wrong. `keepdim=True` gives `(27, 1)`, which broadcasts as a column so row i is divided by its own sum. Verified by `(N / N.sum(1)).sum(1)` not being ones. (Note: I still need to be able to say this back without prompting; the grader prediction was skipped on M1 and M2.)
- **M2, side observation.** Row sums equal column sums for every character on this dataset. Not because the table is square; because every character occurrence inside `.word.` is a *previous* exactly once and a *next* exactly once. Said I got it; not said back in full.
- **M3, likelihood.** Wrong model: "NLL has something to do with the probability we already have"; no idea what likelihood meant. Correction, in my words: how likely the model found everything that actually happened, i.e. take the model's probability for each true next character and multiply them together. (Log, negative, and average still open.)
- **M3, why not the raw product.** Said back after running it: multiplying hundreds of thousands of probabilities below 1 keeps shrinking and underflows to exactly 0, so the raw likelihood is useless as a training number. (Still vague on *how* log fixes it: said "log makes it a number that gets bigger," missing that log turns the product into a sum.)
- **M3, what gets scored.** Wrong model: `nll` needs `xs` too, and "somehow compare the probability to the fact we saw it." Correction, in my words: `probs` already has one row per example (the caller did `P[xs]`); for each row k I look up column `ys[k]`, the probability the model gave to what actually came next, and that picked number *is* the score; there is nothing to compare it to. Then log each, add them, negate. (Average still open. `ys` is already indices, not strings.)
- **M3, average.** Found by the grader (off by a factor of n), fixed with `.mean()` (sum / len(ys) is identical). Not yet said back *why* the average: owed one sentence using the uniform model = log(27).
- **M4, one-hot @ W.** Wrong model (cold attempt Q4): `one_hot @ W` gives "a vector of probabilities for that category." Correction, in my words: a one-hot with a 1 at position 5 times W takes all 27 values at row 5 of W. (Not yet said: why the other rows vanish, and that this makes the matmul equal to `W[xs]`.)
- **M7, the W**2 penalty.** Wrong model: copied `alpha * mean(W**2)` without an intuition. Correction, in my words: mean(W**2) is 0 only when every weight is 0 and grows as weights move away from 0; adding alpha times it to the loss makes non-zero weights cost loss, so gradient descent pushes the weights toward 0. Also: a W row with all-equal values gives a uniform output row. (Still mixing up weights vs table: weights go to 0, the table goes to 1/27. And said "exp(0)=1, divide by 1" instead of divide by the row sum of 27.)
- **M7, smoothing placement.** Wrong model: add k to the row sum. Correction (grader: "still contains zeros"): k goes into every cell of N before normalizing; adding it only to the denominator leaves zero counts at zero. (Pending: say back why.)

### What surprised me / not clear (closed 2026-09-24)

A single layer of weights can be the same thing as a probability lookup table. Softmax, and the relationship between e and the logarithm, took real work to understand. It was a good reminder that backprop starts at the loss.

Closed after milestone 7, with all seven milestones passing. The from-memory train-and-sample cell was skipped by choice; the quiz covers that loop instead. Never said back in full: why the loss is averaged, what `neural_probs(log(P))` returns and why, and the "smoothing == regularization" sentence. I never predicted a grader result before running it.

# Unit 4 — makemore part 3: activations, gradients, BatchNorm

Lecture: https://www.youtube.com/watch?v=P6sfmUTpUmc (1h56m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

Why does a deep net train badly at init, and what do you do about it?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. A 27-way classifier has seen zero data. What loss *should* it report on its first batch, and what does it mean if the number is much bigger?
2. A hidden unit's tanh output is 0.9999 for every example in the batch. What is the gradient flowing back through it, and what happens to that unit's incoming weights over training?
3. `y = x @ W`, with `x` standard normal and `W` drawn from `randn`. As a function of the number of inputs, how does the std of `y` compare to the std of `x`? What do you multiply `W` by to cancel that, and why does tanh need an extra factor on top?
4. Write, in words or math, what BatchNorm does to a `(B, n_hidden)` tensor in training mode. Why does it then need two learnable vectors?
5. At test time you feed one example. Why can't the training-mode BatchNorm handle that, and what does the layer use instead?
6. A `Linear` layer feeding a BatchNorm has a bias. What does that bias do? Then: how do you tell from the size of the parameter updates that the learning rate is roughly right?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–27:53 starter code, fixing the initial loss, fixing the saturated tanh
  - 27:53–40:40 the init scale: Kaiming init, and measuring it
  - 40:40–78:35 batch normalization, its summary, resnet50 walkthrough, first summary
  - 78:35–1:06:04 pytorch-ifying the code, the four visualizations, the no-nonlinearity case
  - 1:06:04–end bringing back BatchNorm, the real summary
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`makemore_batchnorm.ipynb`. Six milestones, graded in the notebook, stops at first failure.

Launch from this folder:

```
../../.venv/bin/jupyter lab --notebook-dir=.
```

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | the loss at init: what it should be, and an init that lands on it | 1 |
| 2 | the saturated tanh: measure saturation, dead units, the dying local gradient | 1 |
| 3 | Kaiming scale, derived and then measured through a stack of layers | 2 |
| 4 | `BatchNorm1d` from scratch, matching `torch.nn.BatchNorm1d` in train and eval | 3 |
| 5 | `Linear` / `Tanh` in the pytorch-ified style; the deep model learns | 4 |
| 6 | *stretch:* update/data ratio, and "the bias before BatchNorm does nothing", measured | 4–5 |

Rules: no lecture open while coding, no makemore repo or lecture notebook. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

The dataset, the one-layer forward pass, the deep-model assembly, the training loop and the four diagnostic plots are given in the notebook. Everything the grader checks, you wrote.

## Review

`quiz_04_makemore_batchnorm.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own training run + eval**, from memory: build the deep model from your layers, train, switch to eval mode, report dev loss. Then once more without BatchNorm and without the tanh gain, and look at the four plots for each.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 4, "Building makemore Part 3: Activations & Gradients, BatchNorm". The folder
code/23_zero-to-hero/04_makemore_batchnorm/ in this repo already contains a notebook
exercise (makemore_batchnorm.ipynb) and a grader (test_makemore_batchnorm.py). Do not
rebuild any of that.

The core question for this unit is: why does a deep net train badly at init, and what
do you do about it?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt questions,
one or two at a time. Do not teach or answer before I answer:

1. A 27-way classifier has seen zero data. What loss should it report on its first
   batch, and what does it mean if the number is much bigger?
2. A hidden unit's tanh output is 0.9999 for every example in the batch. What is the
   gradient flowing back through it, and what happens to that unit's incoming weights?
3. y = x @ W, x standard normal, W from randn. How does the std of y compare to the
   std of x as a function of the number of inputs? What do you multiply W by to cancel
   that, and why does tanh need an extra factor on top?
4. In words or math, what does BatchNorm do to a (B, n_hidden) tensor in training
   mode, and why does it then need two learnable vectors?
5. At test time you feed one example. Why can't training-mode BatchNorm handle that,
   and what does the layer use instead?
6. A Linear layer feeding a BatchNorm has a bias. What does that bias do? And how do
   you tell from the size of the parameter updates that the learning rate is roughly right?

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
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one bullet per idea, only after I have said it back correctly in my own words, never when you explained it. Record the wrong model I had and the correction that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone / what I was doing>` to the unit note's Notes, so a later session can resume from it. When resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in quiz_04_makemore_batchnorm.md (+1 day, +4 days, +2 weeks from today) and rewrite its questions so they target the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

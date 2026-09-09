# Unit 1 — micrograd: backprop from scratch

Lecture: https://www.youtube.com/watch?v=VMj-3S1tku0
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

How does a number learn which direction to move?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. What is a derivative, in one sentence, without the word "slope"?
2. `c = a * b`. If I nudge `a` by a tiny amount, how much does `c` move? What about `c = a + b`?
3. A chain of operations produces `L`. Why can you get `dL/da` for every `a` in the chain without ever writing down the whole formula?
4. Why must gradients be *added* into a node rather than assigned, and when does the difference show up?
5. In what order must you walk the graph when backpropagating, and why?
6. What is the one line of a training step that actually makes the network better?

## Consume

- The lecture, in sections. Suggested stops:
  - 0:00–25:00 derivatives, the `Value` object, the expression graph
  - 25:00–52:00 manual backprop, the chain rule on the graph
  - 52:00–1:20 `_backward` closures, topological sort, the accumulation bug
  - 1:20–end tanh vs. primitive ops, Neuron/Layer/MLP, the training loop
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`micrograd.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | a number that remembers its inputs | 1 |
| 2 | one hop of gradient, local derivative × incoming | 2 |
| 3 | full backward pass in topological order | 3 |
| 4 | gradient accumulation on reused nodes | 3 |
| 5 | tanh as a single op | 4 |
| 6 | *stretch:* exp, pow, and ops built from them | 4 |
| 7 | *stretch:* Neuron / Layer / MLP, and it learns | 4 |

Rules: no lecture open while coding, no real micrograd repo. Predict before every grader run.
Stuck on an idea for 20 min → one tier of `HINTS.md`. Stuck on syntax → ask.

## Output

- Notebook section at the bottom: **your own training loop**, from memory, on the toy data.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 1, micrograd. The folder code/23_zero-to-hero/01_micrograd/ in this repo
already contains a notebook exercise (micrograd.ipynb), a grader
(test_micrograd.py), and tiered hints (HINTS.md). Do not rebuild any of that.

The core question for this unit is: how does a number learn which direction to move?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. What is a derivative, in one sentence, without the word "slope"?
2. c = a * b. If I nudge a by a tiny amount, how much does c move? What about c = a + b?
3. A chain of operations produces L. Why can you get dL/da for every a without writing down the whole formula?
4. Why must gradients be added into a node rather than assigned, and when does the difference show up?
5. In what order must you walk the graph when backpropagating, and why?
6. What is the one line of a training step that actually makes the network better?

After I answer, do three things:
1. Record my answers under "My Cold Attempt".
2. Corrections or missing nuance under "Corrections / Gaps".
3. A short "What to look for while watching" list.

Then switch to coaching mode for the notebook. Rules for that mode:
- Don't explain. If I'm stuck I'll tell you what I tried and what I expected;
  give me the next hint tier from HINTS.md only, and only the one I need.
- Before I run a grader cell for the first time on a milestone, ask me to
  predict what it will print.
- If I ask a Python syntax question, just answer it.
- If I say "just tell me", tell me.
- Push on vague language. Keep me doing the work.
```

## Notes


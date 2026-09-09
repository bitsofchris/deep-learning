# Zero to Hero — Karpathy's neural networks playlist, rebuilt from scratch

One folder per lecture of the
[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
playlist. Watching is not learning; each lecture becomes a from-scratch exercise
with a grader in front of me.

## Launch

From this folder, copy-paste:

```bash
../../.venv/bin/jupyter lab --notebook-dir=.
```

Or from anywhere in the repo, `./nb code/23_zero-to-hero/01_micrograd`.

## Lectures

| # | Lecture | Folder | Status |
|---|---------|--------|--------|
| 1 | micrograd — backprop and autograd | `01_micrograd/` | in progress |
| 2 | makemore 1 — bigram model | `02_makemore_bigram/` | todo |
| 3 | makemore 2 — MLP | `03_makemore_mlp/` | todo |
| 4 | makemore 3 — activations, gradients, BatchNorm | `04_makemore_batchnorm/` | todo |
| 5 | makemore 4 — becoming a backprop ninja | `05_makemore_backprop_ninja/` | todo |
| 6 | makemore 5 — WaveNet | `06_makemore_wavenet/` | todo |
| 7 | GPT from scratch | `07_gpt/` | todo |
| 8 | GPT tokenizer | `08_tokenizer/` | todo |

## The loop (borrowed from `22_ai-foundations-linear-algebra`)

Each lecture folder has one **unit note** (`unit_NN_<name>.md`) and one
**notebook**. The note has seven sections, in order:

1. **Question** — the single question the lecture answers.
2. **Cold Attempt** — answer 5–6 questions from memory *before* watching. Vague answers are the gaps.
3. **Consume** — the lecture, split into sections, with a stop after each.
4. **Practice** — the notebook: stubs per milestone, grader cell after each, stops at first failure.
5. **Output** — something shippable: a section in the notebook plus a paragraph of what surprised you.
6. **LLM Kickoff Prompt** — paste into a new chat. It quizzes you *before* it teaches, then coaches on hints only.
7. **Notes** — running notes. Stay in the same note across days.

Hints live in `HINTS.md`, tiered, one idea per tier. Read one tier at a time.

## Files per lecture

```
NN_<name>/
  unit_NN_<name>.md     the note above; start here
  <name>.ipynb          the exercise; a cell of stubs, then a grade(...) cell, per milestone
  test_<name>.py        the grader; from test_<name> import grade
  HINTS.md              tiered hints, never inlined in the notebook
```

## Starting the next lecture

`PROMPT.md` is the prompt that generates a new lecture folder in this layout.
Fill in the four brackets, including your own cold-attempt paragraph, and paste
it into a fresh chat. It asks the model to verify its own grader against a
private reference before shipping.

## Rules

- Don't open the lecture while coding. Don't open the real repo.
- The grader is the source of truth. One failing milestone at a time.
- Predict out loud before every grader run.
- Stuck on an *idea* for 20 min → one hint tier, or ask. Stuck on *syntax* → ask immediately.

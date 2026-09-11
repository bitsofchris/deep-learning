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

Playlist order. Each folder has a unit note, a notebook of stubs with a grader
cell per milestone, and the grader module.

| # | Video | Folder | Status |
|---|-------|--------|--------|
| 1 | The spelled-out intro to neural networks and backpropagation: building micrograd | `01_micrograd/` | done, in review |
| 2 | The spelled-out intro to language modeling: building makemore | `02_makemore_bigram/` | scaffolded |
| 3 | Building makemore Part 2: MLP | `03_makemore_mlp/` | scaffolded |
| 4 | Building makemore Part 3: Activations & Gradients, BatchNorm | `04_makemore_batchnorm/` | scaffolded |
| 5 | Building makemore Part 4: Becoming a Backprop Ninja | `05_makemore_backprop_ninja/` | scaffolded |
| 6 | Building makemore Part 5: Building a WaveNet | `06_makemore_wavenet/` | scaffolded |
| 7 | Let's build GPT: from scratch, in code, spelled out. | `07_gpt/` | scaffolded |
| 8 | State of GPT (talk, watch-only) | `08_state_of_gpt/` | note only |
| 9 | Let's build the GPT Tokenizer | `09_tokenizer/` | scaffolded |
| 10 | Let's reproduce GPT-2 (124M) (densest components only) | `10_gpt2_reproduce/` | scaffolded |

Shared data in `data/`: `names.txt` (makemore) and `tinyshakespeare.txt` (GPT).

## The loop (borrowed from `22_ai-foundations-linear-algebra`)

Each lecture folder has one **unit note** (`unit_NN_<name>.md`) and one
**notebook**. The note has seven sections, in order:

1. **Question** — the single question the lecture answers.
2. **Cold Attempt** — answer 5–6 questions from memory *before* watching. Vague answers are the gaps.
3. **Consume** — the lecture, split into sections, with a stop after each.
4. **Practice** — the notebook: stubs per milestone, grader cell after each, stops at first failure.
5. **Review** — three spaced retrieval quizzes from memory, checked against the Coaching log.
6. **Output** — something shippable: a section in the notebook plus a paragraph of what surprised you.
7. **LLM Kickoff Prompt** — paste into a new chat. It quizzes you *before* it teaches, then coaches on hints only.
8. **Notes** — running notes. Stay in the same note across days.

Hints come from the coaching chat, one tier at a time, only when asked. No hints file.

## Files per lecture

```
NN_<name>/
  unit_NN_<name>.md     the note above; start here
  <name>.ipynb          the exercise; a cell of stubs, then a grade(...) cell, per milestone
  test_<name>.py        the grader; from test_<name> import grade(..., upto=, skip=)
  quiz_NN_<name>.md     spaced-retrieval quiz at +1d, +4d, +2w; dates filled when the unit ends
```

The unit note's Notes section carries a **Coaching log** (one bullet per idea,
written only after you've said it back correctly) and a `**Paused <date>.**`
line for resuming mid-unit.

## Starting the next lecture

Watch it, write a 3–8 sentence recall paragraph from memory, then paste
`PROMPT.md` (brackets filled) into Claude Code opened at the repo root. It
builds the next folder in this layout and verifies its own grader before
handing it over. Then answer the Cold Attempt in the new unit note, paste its
kickoff prompt into a fresh chat, and launch the notebook.

## Rules

- Don't open the lecture while coding. Don't open the real repo.
- The grader is the source of truth. One failing milestone at a time.
- Predict out loud before every grader run.
- Stuck on an *idea* for 20 min → ask the coaching chat for a hint. Stuck on *syntax* → ask immediately.

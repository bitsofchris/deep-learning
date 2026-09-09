# Zero to Hero — Karpathy's neural networks playlist, rebuilt from scratch

**What:** One folder per lecture of the
[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
playlist. Each folder is a *from-scratch exercise*: a skeleton with
`raise NotImplementedError` stubs, a grader that stops at the first failure,
and a tiered hints file. Boilerplate is given; the ideas are left for me.

**Why:** Watching is not learning. Re-deriving each lecture's core idea with
a grader in front of me is the only way it sticks.

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

## Workflow per lecture

1. Watch the lecture (or a section of it).
2. **Write a recall paragraph from memory before rewatching anything.** Put it
   in `NN_<lecture>/RECALL.md`.
3. Paste `PROMPT.md` into a fresh chat with the recall paragraph filled in.
   It produces `<name>.py` (skeleton), `test_<name>.py` (grader), `HINTS.md`.
4. Work the milestones. Before each grader run, say out loud what you expect.
5. Stuck on an *idea* for 20 min → read one hint tier, or ask in chat.
   Stuck on *Python syntax* → ask immediately.
6. When the grader passes end to end, write a few lines in `RECALL.md` about
   what surprised you. That gap is the lesson.

## Running things

```bash
# from the repo root
source .venv/bin/activate
python code/23_zero-to-hero/01_micrograd/test_micrograd.py
```

For tinkering in a notebook, use the launcher in the repo root:

```bash
./nb code/23_zero-to-hero/01_micrograd     # opens Jupyter Lab in that folder
./nb                                       # opens Jupyter Lab at the repo root
```

## Rules of engagement

- Don't open the lecture while coding. Don't open the real repo.
- The grader is the source of truth. One failing test at a time.
- Predict before you run.

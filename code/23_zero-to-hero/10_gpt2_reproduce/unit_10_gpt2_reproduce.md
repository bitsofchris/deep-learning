# Unit 10 — Let's reproduce GPT-2 (124M): the parts that are ideas

Lecture: https://www.youtube.com/watch?v=l8pRSuU81PU (4h01m)
Stay in this note until the unit is done. Don't move on to fake progress.

This lecture is too big to rebuild and most of it is engineering. This unit scaffolds only the
components with the highest conceptual density per line; everything else is handed over working
or marked "watch, don't rebuild."

## Question

What separates a toy transformer from a real one, and which of those differences are *ideas*
versus *engineering*?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. GPT-2 computes q, k and v with **one** `Linear` of width `3*n_embd` instead of three. Why is that the same computation as lecture 7's, and what shape does a tensor have to be in before all the heads can attend at once?
2. The token embedding `wte` and the output head `lm_head` share a single weight matrix. Why does that make sense, and roughly how many of GPT-2's 124M parameters does it save?
3. Every weight is initialised with std 0.02, except the Linears that write into the residual stream, which are scaled down by `1/sqrt(2*n_layer)`. What grows if you don't do that, and why `2*n_layer` rather than `n_layer`?
4. Your GPU fits a batch of 16 sequences but the paper trained on 0.5M tokens per step. How do you get the gradient of the big batch anyway, and which single line is easy to get wrong so that the result is *not* equivalent?
5. Sketch the learning-rate schedule: what is the LR at step 0, why not start at the maximum, and what shape does it follow after warmup?
6. Which parameters get weight decay and which don't? Why is gradient clipping in the loop at all, and where exactly does it sit relative to the backward passes and the optimizer step?

## Consume

- The lecture, in sections. Suggested stops:
  - **Section 1** 0:00–45:50 — the checkpoint, the `nn.Module` (`CausalSelfAttention`, `Block`, `MLP`, `GPT`), loading weights, the forward pass, sampling. → milestone 1
  - **Section 2** 45:50–1:22:18 — data batches, cross-entropy, overfitting one batch, `DataLoaderLite`, weight tying, init. → milestones 2, 3
  - **Section 3** 1:22:18–2:14:55 — *engineering, watch but don't rebuild:* GPUs, TF32, bfloat16, `torch.compile`, flash attention, vocab padding. Carry one question: would the loss curve differ if I skipped this?
  - **Section 4** 2:14:55–3:10:21 — AdamW, gradient clipping, the LR schedule, weight decay, gradient accumulation, DDP (watch only). → milestones 4, 5, 6
  - **Section 5** 3:10:21–end — datasets, validation, HellaSwag, results. → milestone 7
- Stop after each section and try the matching milestones in the notebook before continuing.

## Practice

`gpt2_reproduce.ipynb`. Seven milestones; 1–6 are graded in the notebook, stopping at first
failure; 7 is a predict-the-outcome answered in the coaching chat.

Launch from this folder:

```bash
../../.venv/bin/jupyter lab --notebook-dir=.
```

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | `CausalSelfAttention` with the fused qkv Linear and the `(B, nh, T, hs)` transpose dance | 1 |
| 2 | `DataLoaderLite`: `B*T+1` windows over a token stream, with wraparound | 2 |
| 3 | weight tying `wte`/`lm_head` and GPT-2 init (std 0.02, residual projections scaled) | 2 |
| 4 | gradient accumulation that equals one big batch, exactly | 4 |
| 5 | warmup + cosine LR schedule, and it learns | 4 |
| 6 | *stretch:* AdamW param groups (decay matrices, not vectors) and where clipping goes | 4 |
| 7 | *stretch:* HellaSwag scoring, predict the outcome (no code) | 5 |

Given, fully written: `GPTConfig`, `MLP`, `Block`, the `GPT` container and forward pass, char-level
tokenisation of `../data/tinyshakespeare.txt` standing in for tiktoken, the training-loop scaffold.
Everything runs on CPU in seconds (n_layer 2, n_head 2, n_embd 32, block 32). No tiktoken,
no transformers, no downloaded weights.

Rules: no lecture open while coding, no build-nanogpt repo. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_10_gpt2_reproduce.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own sampling loop** (`generate`) from memory, and a
  few lines sampled from the model you trained in milestone 5.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the
  grader said. That gap is the lesson.
- One list in *Notes*: which parts of this lecture you now file under "idea" and which under
  "engineering," in your own words.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach
second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 10, "Let's reproduce GPT-2 (124M)". The folder
code/23_zero-to-hero/10_gpt2_reproduce/ in this repo already contains a notebook
exercise (gpt2_reproduce.ipynb) and a grader (test_gpt2_reproduce.py). Do not
rebuild any of that. The unit deliberately scaffolds only the high-density
components (attention, data loader, tying + init, gradient accumulation, LR
schedule, optimizer groups + clipping) and hands the rest over working; GPUs,
mixed precision, torch.compile, flash attention and DDP are watch-only.

The core question for this unit is: what separates a toy transformer from a real
one, and which of those differences are ideas versus engineering?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. GPT-2 computes q, k and v with one Linear of width 3*n_embd instead of three. Why is that the same computation as lecture 7's, and what shape does a tensor have to be in before all the heads can attend at once?
2. wte and lm_head share a single weight matrix. Why does that make sense, and roughly how many of GPT-2's 124M parameters does it save?
3. Every weight is initialised with std 0.02, except the Linears that write into the residual stream, which are scaled by 1/sqrt(2*n_layer). What grows if you don't, and why 2*n_layer rather than n_layer?
4. Your GPU fits a batch of 16 but the paper trained on 0.5M tokens per step. How do you get the gradient of the big batch anyway, and which single line is easy to get wrong so the result is not equivalent?
5. Sketch the LR schedule: the LR at step 0, why not start at the maximum, and the shape after warmup.
6. Which parameters get weight decay and which don't? Why is gradient clipping there at all, and where does it sit relative to the backward passes and the optimizer step?

After I answer, do three things:
1. Record my answers under "My Cold Attempt".
2. Corrections or missing nuance under "Corrections / Gaps".
3. A short "What to look for while watching" list, keyed to the lecture sections
   in the unit note (section 3 is watch-only: help me tell engineering from idea).

Then switch to coaching mode for the notebook. Rules for that mode:
- Don't explain. If I'm stuck I'll tell you what I tried and what I expected;
  give me ONE hint, tiered: first a question, then the shape of the idea, and
  only something close to the answer if I ask a third time. One tier per ask.
- Before I run a grader cell for the first time on a milestone, ask me to
  predict what it will print.
- If I ask a PyTorch or Python syntax question, just answer it.
- If I say "just tell me", tell me.
- Milestone 7 (HellaSwag) has no grader: I'll give you four predictions; do not
  confirm or correct any of them until all four are in.
- Push on vague language. Keep me doing the work.
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one bullet
  per idea, only after I have said it back correctly in my own words, never when you
  explained it. Record the wrong model I had and the correction that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone / what I
  was doing>` to the unit note's Notes, so a later session can resume from it. When
  resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in quiz_10_gpt2_reproduce.md
(+1 day, +4 days, +2 weeks from today) and rewrite its questions so they target the
gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)



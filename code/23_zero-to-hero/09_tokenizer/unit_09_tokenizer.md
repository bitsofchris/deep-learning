# Unit 9 — the GPT tokenizer: byte-pair encoding from scratch

Lecture: https://www.youtube.com/watch?v=zduSFxRajkE (2h13m)
Stay in this note until the unit is done. Don't move on to fake progress.

## Question

Why does a language model see text as byte-pair chunks, and how are those chunks decided?

## Cold Attempt

Answer before watching (or before rewatching). Vague answers are the gaps.

1. Before any merging, what does the string `"hello"` become, and why is the starting vocabulary exactly 256 entries? What happens to `"안녕"` or an emoji?
2. Describe one iteration of BPE training in two sentences. What are the two data structures it leaves behind when it stops?
3. After training, you encode text the tokenizer has never seen. Why can't you just scan for any adjacent pair you have a merge for? What has to be respected, and what goes wrong if it isn't?
4. Why does GPT-2 run a regex over the text *before* BPE? Name one concrete thing that would be learned without it that you don't want.
5. A model can emit any sequence of token ids. What can go wrong when you turn that back into a string, and what should happen instead of a crash?
6. The lecture calls the tokenizer "a completely separate stage" from the LLM. What does the LLM actually see, and name one well-known LLM weirdness that traces back to tokenization rather than to the model.

## Consume

The lecture, in sections. Stop after each one and try the matching milestones in the notebook before continuing.

- **Section 1 — 0:00–22:47** why tokenization matters, tiktokenizer demo, Unicode code points, UTF-8 / UTF-16 / UTF-32 and why bytes.
- **Section 2 — 22:47–39:20** the BPE algorithm by hand, `get_stats`, `merge`, the training loop, compression ratio.
- **Section 3 — 39:20–57:36** the tokenizer as a separate stage, `decode`, `encode` and the merge-priority problem.
- **Section 4 — 57:36–1:28:42** the GPT-2 regex split, tiktoken, OpenAI's `encoder.py`, special tokens, the minbpe exercise.
- **Section 5 — 1:28:42–end** sentencepiece / Llama 2, choosing vocab size, adding tokens, multimodal tokens, the catalogue of tokenization quirks. No milestone; watch it after the notebook and write the "what surprised me" paragraph afterwards.

## Practice

`tokenizer.ipynb`. Seven milestones, graded in the notebook, stops at first failure.

Launch from this folder:

```bash
../../.venv/bin/jupyter lab --notebook-dir=.
```

| Milestone | Idea | Lecture section |
|-----------|------|-----------------|
| 1 | text → UTF-8 bytes → ints, and back without crashing | 1 |
| 2 | `get_stats`: count consecutive pairs | 2 |
| 3 | `merge`: replace a pair everywhere, every edge case | 2 |
| 4 | `train`: the BPE loop, `merges` and `vocab` | 2 |
| 5 | `encode` / `decode`, merges applied in the order they were learned | 3 |
| 6 | *stretch:* regex pre-tokenization, stats pooled across chunks | 4 |
| 7 | *stretch:* special tokens, compression ratio, one prediction | 4 |

Given as boilerplate: loading the Shakespeare slice, a `render` helper that shows token
boundaries, the regex pattern (rewritten for the standard `re` module), the class
containers. Everything else is yours: the byte round trip, the pair statistic, the merge
loop, the training loop, encode/decode, wiring the regex in, special-token splitting.
No `tiktoken`, no minbpe.

Rules: no lecture open while coding, no minbpe repo. Predict before every grader run.
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_09_tokenizer.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own train / encode / decode**, from memory, as
  plain functions, retyping the helpers. Compare its ids against the graded version on a
  line you make up.
- The milestone 7 prediction: which of `BasicTokenizer` / `RegexTokenizer` compresses the
  Shakespeare slice more at vocab 320, and by how much. Write it in *Notes* before running.
- A paragraph in *Notes*: what surprised you. Where your prediction differed from what the
  grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
the tokenizer lecture, "Let's build the GPT Tokenizer". The folder
code/23_zero-to-hero/09_tokenizer/ in this repo already contains a notebook
exercise (tokenizer.ipynb) and a grader (test_tokenizer.py). Do not rebuild any
of that.

The core question for this unit is: why does a language model see text as
byte-pair chunks, and how are those chunks decided?

Start as a Socratic tutor and pre-check examiner. Ask me these cold-attempt
questions, one or two at a time. Do not teach or answer before I answer:

1. Before any merging, what does the string "hello" become, and why is the
   starting vocabulary exactly 256 entries? What happens to "안녕" or an emoji?
2. Describe one iteration of BPE training in two sentences. What are the two
   data structures it leaves behind when it stops?
3. After training, you encode text the tokenizer has never seen. Why can't you
   just scan for any adjacent pair you have a merge for? What has to be
   respected, and what goes wrong if it isn't?
4. Why does GPT-2 run a regex over the text before BPE? Name one concrete thing
   that would be learned without it that you don't want.
5. A model can emit any sequence of token ids. What can go wrong when you turn
   that back into a string, and what should happen instead of a crash?
6. The tokenizer is "a completely separate stage" from the LLM. What does the
   LLM actually see, and name one well-known LLM weirdness that traces back to
   tokenization rather than to the model.

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
- Before I run the compression comparison in milestone 7, ask for my prediction
  and my reasoning, and don't confirm or correct until I have run it.
- If I ask a Python syntax question (slicing, dict ordering, re.findall,
  re.split with groups, bytes vs str), just answer it.
- If I say "just tell me", tell me.
- Push on vague language. Keep me doing the work.
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one
  bullet per idea, only after I have said it back correctly in my own words,
  never when you explained it. Record the wrong model I had and the correction
  that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <milestone /
  what I was doing>` to the unit note's Notes, so a later session can resume
  from it. When resuming, read the Coaching log and the Paused line first.

When the unit finishes: fill in the three review dates in quiz_09_tokenizer.md
(+1 day, +4 days, +2 weeks from today) and rewrite its questions so they target
the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)

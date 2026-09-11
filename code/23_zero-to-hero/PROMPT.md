# Lecture → exercise prompt

Source of truth: the vault note `Lecture Learning Prompt`. This is a copy.

One note, one prompt, every lecture. Same loop as the linear algebra sprint: **Question → Cold Attempt → Consume → Practice → Output → Notes**. One unit per lecture. Stay in it until it's done; don't fake progress by moving on.

Lecture 1 (micrograd) is the exemplar and is already built: `code/23_zero-to-hero/01_micrograd/`.

## The loop, per lecture (Zero to Hero: every folder is already built)

Each lecture has a folder `code/23_zero-to-hero/NN_<name>/` with a unit note, a notebook of stubs, and a grader. The unit note is the whole process; you never have to design anything.

1. **Open the unit note** `unit_NN_<name>.md`. Read the Question.
2. **Cold Attempt.** Answer its 5–6 questions in the Notes section, from memory, before watching. Vague answers are the gaps.
3. **Kickoff chat.** Copy the note's *LLM Kickoff Prompt* into a new Claude chat (Claude Code in the repo, so it can read the folder). It quizzes you on the cold attempt, corrects, and tells you what to look for while watching. Keep this chat open; it is your coach for the whole unit.
4. **Watch** the lecture section by section, using the stops in Consume. After each stop, do the milestones mapped to it.
5. **Practice.** From the lecture folder:
   ```
   ../../.venv/bin/jupyter lab --notebook-dir=.
   ```
   Work the notebook top to bottom. Say your prediction out loud before every grader run. Stuck on an idea for 20 min → ask the coach for a hint (one tier per ask). Stuck on syntax → ask.
6. **Output.** Fill the notebook's last section (the from-memory build) and write a "what surprised me" paragraph in the unit note's Notes. The coach has been keeping a *Coaching log* there too: one bullet per idea you said back correctly, with the wrong model and the correction.
7. **Close the unit.** Ask the coach to fill in the three review dates in `quiz_NN_<name>.md` and rewrite its questions from the Coaching log gaps. Tick the lecture in the sprint PMOC and update the status column in the course README.
8. **Review.** Do the quiz at +1 day, +4 days, +2 weeks, from memory, then check the Coaching log. Score 0/1/2; anything weak twice gets a +1 month fourth review.

Stopping mid-unit: the coach appends `**Paused <date>.** Next: …` to Notes. Next session starts by reading the Coaching log and that line.

Unit 8 (State of GPT) is a talk: cold attempt, watch, redraw the pipeline, no notebook.

## Prompt A — build a unit (only for a new course, or to rebuild one)

Not needed for Zero to Hero; all ten folders exist. Paste after watching. Fill the four brackets.

## Prompt A — build the unit

Paste after watching. Fill the four brackets. Attach transcript or notes if you have them.

```text
I just finished [lecture / section] of Karpathy's Neural Networks: Zero to Hero.
I got through [where you stopped]. Here's what I think it covered, in my own
words, from memory, before rewatching anything:

[3–8 sentences]

Build me a from-scratch learning unit on this material, in this repo, in the
exact layout of code/23_zero-to-hero/01_micrograd/ — read that folder first
and match its shape. Put it in code/23_zero-to-hero/NN_<lecture-name>/.

Rules:

BEFORE ANY CODE, tell me the two or three real ideas in this material — the
ones that, if I don't have them, nothing else works. Under 200 words. Don't
teach them, just name them so I know what I'm aiming at. Use my recall
paragraph above to spot my gaps; the cold-attempt questions you write should
aim at those gaps.

SPLIT THE WORK. You write the parts whose implementation teaches me nothing:
data loading, plotting, tokenizer glue, training-loop boilerplate, anything
I'd copy off a page. I write every part that IS the idea. If you catch
yourself writing the thing the lecture was about, stop and hand it back to
me as a stub. When you're unsure which side something falls on, ask me.
What counts as boilerplate for THIS lecture: [say it, e.g. "the tokenizer
and tensor plumbing are boilerplate; broadcasting and the loss are the lesson"].

FOUR FILES:
1. unit_NN_<name>.md — Question / Cold Attempt (5–6 questions) / Consume
   (the lecture split into sections with timestamps and a stop after each) /
   Practice (milestone table mapped to sections) / Output / LLM Kickoff
   Prompt / Notes. The kickoff prompt is a Socratic pre-check first (ask my
   cold-attempt questions, don't teach until I answer, then "My Cold Attempt"
   / "Corrections / Gaps" / "What to look for while watching"), then coaching
   mode (no explaining, one hint tier at a time only when I ask, ask me to
   predict before each first grader run, answer syntax directly, "just tell me" means tell me).
2. <name>.ipynb — THE SKELETON IS THE NOTEBOOK, not a .py file. Cell 0 links
   the unit note and the rules. Then per milestone: a markdown cell, a code
   cell of stubs with real signatures and docstrings stating shapes and
   contracts and raise NotImplementedError bodies, and a grade(...) cell.
   Later milestones extend earlier classes with Class.method = method so I
   never re-run one giant cell. Every milestone's markdown cell from the hard
   one onward is a spec, not a hint: per class or function, what it holds,
   what it computes in one expression, what it returns including edge cases,
   and the order the grader checks things in. End with an Output cell (the from-memory
   build) and a Scratch cell.
3. test_<name>.py — a grader exposing grade(...) I call from the notebook.
   No pytest. Ordered milestones, stops at the first failure so I always
   have exactly one thing in front of me, upto= to run a prefix, skip=() to
   skip a stretch milestone nothing later depends on. On a stub, print the
   qualified name of the method that raised NotImplementedError. On any
   failure, print how many checks in that milestone passed and the last
   one's message. Check my
   work numerically where you can (finite differences, or compare against
   torch on the same inputs) rather than against fixed constants. Failure
   messages describe the SYMPTOM and point at a likely cause without naming
   the fix.
4. quiz_NN_<name>.md — spaced-retrieval quiz: a 3-row review table (+1 day,
   +4 days, +2 weeks, dates blank until the unit finishes), a 0/1/2 score
   grid, the rule that anything weak twice gets a +1 month fourth review,
   and ~10 questions answerable from memory: concrete numeric cases ("given
   these values, what prints"), "write X from memory" code items, one-
   sentence "why" questions, and the unit's core question last. Link it
   from a "## Review" section in the unit note, before ## Output.

No hints file. Hints live in the coaching chat: keep a private tiered
ladder per milestone (tier 1 a question, last tier close to the answer) in
your head, never in the notebook, and hand out one tier at a time only when
I ask.

ORDER THE MILESTONES so the hard idea comes third or fourth, not first. Two
wins before the wall. Five to seven milestones. Mark anything past the
lecture's scope as stretch.

VERIFY YOUR OWN GRADER before you ship it. In your scratchpad, not the repo:
write a private reference solution and run grade() against it; execute the
untouched notebook headlessly and confirm every grade cell stops at
milestone 1; then break the reference in the three or four ways a learner
most likely would and confirm each fails at the intended milestone with a
useful message. Don't show me any of that.

IN CONVERSATION, don't explain. If I'm stuck I'll come to you with what I
tried and what I expected; give me one hint tier, a question first, and
only go closer to the answer if I ask again. If I say "just
tell me," then tell me. Before I run the grader the first time on a
milestone, ask me to predict what will happen.

When done: update the status table in code/23_zero-to-hero/README.md, give
me the launch command from the new folder, and ask for my prediction on
milestone 1.
```

## Knobs

**Too hard** — "give me one more milestone before the hard one," "put a worked example of the first op in the docstring," "narrow this to just [component]."

**Too easy** — "strip the docstrings to one line," "no milestones, one failing suite," "tests only, no stubs."

**The boilerplate line moves.** micrograd: plotting and the loop are boilerplate. makemore: tokenizer and tensor plumbing are boilerplate, broadcasting and the loss are the lesson. GPT: PyTorch itself is scaffolding, attention and the block are the lesson. Tokenizer: file IO is boilerplate, the BPE merge loop is the lesson. Say it in the prompt each time.

**Lecture too big to rebuild** — swap in one of these for the "four files" section:

> Don't give me the whole build. Pick the single component of this lecture with the highest ratio of conceptual density to lines of code, and scaffold only that. Everything else, hand me working.

> Give me a working implementation of this lecture's model with exactly one bug I plausibly would have written. I'll diagnose it from the symptom. Don't tell me where it is or what category it's in.

> Give me five predict-the-outcome questions on this material: a change to the code, and I say what happens to the loss curve or the samples and why. Don't confirm or correct until I've answered all five.

## Two things that make this work

**Write the recall paragraph before you rewatch anything.** It's the input that lets the exercise target *your* gaps rather than a generic learner's.

**Say your prediction out loud before running code.** If you run first and rationalize after, you get the answer without the update.

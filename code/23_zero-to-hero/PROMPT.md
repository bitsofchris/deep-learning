# Lecture → exercise prompt

Paste the block below, fill the four brackets, attach transcript or notes if
you have them.

---

## The prompt

> I just finished **[lecture / section]** of [course]. I got through
> **[where you stopped]**. Here's what I think it covered, in my own words:
> **[3–8 sentences, from memory, before rewatching anything]**.
>
> Build me a from-scratch implementation exercise on this material. Rules:
>
> **Before any code**, tell me the two or three real ideas in this material —
> the ones that, if I don't have them, nothing else works. Keep it under 200
> words. Don't teach them, just name them so I know what I'm aiming at.
>
> **Split the work.** You write the parts whose implementation teaches me
> nothing: data loading, plotting, tokenizer glue, training-loop boilerplate,
> anything I'd otherwise copy off a page. I write every part that *is* the
> idea. If you catch yourself writing the thing the lecture was about, stop
> and hand it back to me as a stub. When you're unsure which side something
> falls on, ask me before you build.
>
> **Give me a skeleton plus a grader.** Skeleton: real signatures, docstrings
> stating shapes and contracts, `raise NotImplementedError` bodies. Grader: a
> runnable test file, no pytest dependency, grouped into ordered milestones,
> stopping at the first failure so I always have exactly one thing in front of
> me. Where you can check my work numerically rather than against a fixed
> expected value, do that — it catches subtler mistakes. Write failure
> messages that describe the *symptom* and point at a likely cause, without
> naming the fix.
>
> **Order the milestones so the hard idea comes third or fourth**, not first.
> I want a couple of wins before the wall. Mark anything past the lecture's
> scope as stretch.
>
> **Hints go in a separate file**, tiered, one idea per tier, ordered so tier
> 1 is a question and the last tier is close to the answer. Never inline them
> in the skeleton.
>
> **In conversation, don't explain.** If I'm stuck I'll come to you with what
> I tried and what I expected; give me the next tier only. If I say "just tell
> me," then tell me.
>
> Before I run the grader the first time on a milestone, ask me to predict
> what will happen.
>
> Also: verify your own grader before you ship it. Write a private reference
> solution, run the tests against it, then deliberately break it in the two or
> three ways a learner most likely would and confirm the tests actually catch
> those. Don't show me any of that.
>
> Put the files in `code/23_zero-to-hero/NN_<lecture-name>/` in this repo,
> following the layout of `01_micrograd/`.

---

## Knobs

**If it's too hard** — say "give me one more milestone before the hard one,"
or "put a worked example of the first op in the docstring," or "narrow this to
just [component]."

**If it's too easy** — "strip the docstrings to one line," "remove the
milestone structure, give me one failing test suite and no ordering," or
"give me the tests only, no skeleton at all."

**The boilerplate line moves as you go.** Early on, plotting is boilerplate.
By the makemore lectures, tensor plumbing and the tokenizer are boilerplate
and broadcasting semantics are the lesson. By the GPT build, PyTorch itself is
scaffolding. Restate what counts as boilerplate *for this lecture* in the
prompt each time — don't assume the same split carries over.

**When the whole thing is too big to rebuild** — later lectures are hours long
and reimplementing everything isn't worth it. Swap in one of these instead:

> Don't give me the whole build. Pick the single component of this lecture
> with the highest ratio of conceptual density to lines of code, and scaffold
> only that. Everything else, hand me working.

> Give me a working implementation of this lecture's model with exactly one
> bug I plausibly would have written. I'll diagnose it from the symptom. Don't
> tell me where it is or what category it's in.

> Give me five predict-the-outcome questions on this material: a change to the
> code, and I say what happens to the loss curve or the samples and why. Don't
> confirm or correct until I've answered all five.

---

## Two things that make this work

**Write your recall paragraph before you rewatch anything.** It's the input
that lets the exercise target *your* gaps rather than a generic learner's. A
vague sentence in that paragraph is a real gap almost every time.

**Say your prediction out loud before running code.** The gap between what you
expected and what happened is the entire lesson. If you run first and
rationalize after, you get the answer without the update.

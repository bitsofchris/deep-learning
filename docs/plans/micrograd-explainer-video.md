# Plan: micrograd explainer video (Karpathy lecture 1 in 5 minutes)

Status: outline, 2026-09-11. Source: `code/23_zero-to-hero/01_micrograd/micrograd.ipynb`
and the recap / coaching log in `unit_01_micrograd.md`. Nothing new gets written for the
video; the notebook is the screen and the recap is the script.

## Frame

Honest and simple: "I just did Karpathy's micrograd lecture. Here it is in five minutes,
walking the notebook I built to learn it. If you want to work through it yourself, the
repo is linked."

Who: people who have watched or plan to watch the lecture and want the short version, or
want a way to do it that isn't just watching. You're a peer one step ahead, not an expert.

## The screen

The notebook, top to bottom. Each segment is one milestone: show the cell, say the idea
in the recap's words, run the grader, `[ok]`. That's the whole visual plan. Optional: a
hand-drawn graph of `a*b` next to the `dump()` output in segment 2, drawn the way the
attention video drew number lines.

## Segments (5:00)

**0:00–0:30 Open.** Two sentences of the frame above. Then the shape of the lecture in one
line, from your recap: "We build two things. micrograd, a backprop engine, and an MLP, a
basic neural network on top of it. Nothing in the network knows calculus."

**0:30–1:30 Milestone 1 & 2: the Value class (cells 3–5).**
Show the `Value` fields. `data`, `grad`, `_prev`, `_backward`. Show `__mul__`: forward
number plus a closure. Say the local derivatives: for `*` it's the other operand, for `+`
it's 1. Point at the one line, `self.grad += other.data * out.grad`: local derivative
times incoming gradient. Run cell 4, show `dump()` with grads `-3` and `2`. Grader `[ok]`.

**1:30–2:30 Milestone 3 & 4: backward() (cells 7–10).**
Read your own markdown answer in cell 7 on camera, that's the explanation. Show
`backward()`: set `grad = 1` at the output because `dL/dL = 1`, build the order, walk it
reversed calling each `_backward`. One line on `+=`: a value used twice gets both
contributions, and every weight is used once per example. Grader `[ok]`.

**2:30–3:00 Milestone 5: tanh (cells 12–13).**
One op, one child, derivative `1 - out.data**2`. Mention the bug you hit: squared the
input instead of the output. Grader `[ok]`.

**3:00–4:00 Milestone 7: Neuron, Layer, MLP (cells 20–21).**
Neuron is `tanh(sum(w*x) + b)` built from Values, so it's already in the graph. Layer is a
list of neurons. MLP is a list of layers. Mention the second bug: wrapping the output in
`Value(...)` cut the graph and no weight got a gradient. `parameters()` is the list the
optimizer nudges, bias included. Grader `[ok]`, 41 params.

**4:00–4:40 Your training loop (cell 26).**
The five steps, from the loop as written: forward, loss, zero grads, backward, update.
Run it, show the loss falling and the predictions matching the signs of `ys`. Say the
unit question's answer in your words: the gradient is the direction that raises the loss,
each weight steps the other way.

**4:40–5:00 Close.** "That's the lecture. If you want to do it and have it stick: answer
the cold questions in the unit note before watching, watch in sections, rebuild each
milestone against the grader, predict what it prints before you run it. Repo in the
description." End.

## Hook / title (optional, pick or skip)

- "Karpathy's micrograd lecture is two and a half hours. Here's the notebook version in five."
- "I just finished Karpathy's micrograd. Here's what you need to know, walking the code I built."
- Title: *Karpathy's micrograd in 5 minutes (walking my notebook)*. "How I finally
  understood" only if it's true for you; the attention video's numbers were partly topic.

## Cuts

Milestone 6 (skipped in the notebook too). Topological sort details beyond cell 7's
sentence. Learning-rate tuning. Any PyTorch comparison beyond "this is what
`loss.backward()` does."

## Before recording

- Restart kernel, run all, so every grader cell shows `[ok]` cleanly.
- Decide whether the repo is public for the link.
- Record the close and open last.

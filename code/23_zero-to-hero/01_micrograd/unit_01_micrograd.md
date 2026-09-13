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
Stuck on an idea for 20 min → ask the coaching chat for a hint. Stuck on syntax → ask.

## Review

`quiz_01_micrograd.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

- Notebook section at the bottom: **your own training loop**, from memory, on the toy data.
- A paragraph in *Notes* below: what surprised you. Where your prediction differed from what the grader said. That gap is the lesson.

## LLM Kickoff Prompt

Paste into a new chat when starting this unit. It is a Socratic pre-check first and a coach second; the exercise already exists in this folder, so it is *not* asked to build one.

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Right now I am on
lecture 1, micrograd. The folder code/23_zero-to-hero/01_micrograd/ in this repo
already contains a notebook exercise (micrograd.ipynb) and a grader
(test_micrograd.py). Do not rebuild any of that.

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
  give me ONE hint, tiered: first a question, then the shape of the idea, and
  only something close to the answer if I ask a third time. One tier per ask.
- Before I run a grader cell for the first time on a milestone, ask me to
  predict what it will print.
- If I ask a Python syntax question, just answer it.
- If I say "just tell me", tell me.
- Push on vague language. Keep me doing the work.
```

## Notes

### Recap (TL;DR of the lecture, in my terms)

We built two things. **micrograd** is a backprop engine: a `Value` class that wraps one
number and remembers how it was made. Then we stacked `Value`s into an **MLP**, a basic
neural network, and trained it. Nothing in the MLP knows calculus. Once `Value` works, a
network is just arithmetic on top of it.

**Backprop** is the algorithm that tells every weight which way to move.
- The **loss** is one `Value` that measures how wrong all predictions are together.
  Squared error summed over the examples. It is the root of the graph.
- A **derivative** is the ratio: how much the output moves per tiny nudge of an input.
  `dL/dw` is the number each weight needs.
- You never differentiate the whole formula. Each op knows only its **local derivative**
  (`*`: the other operand, `+`: 1, `tanh`: `1 - out²`, `x**n`: `n·x^(n-1)`), and the
  chain rule multiplies local × incoming gradient, one hop at a time, from `L` back to
  the leaves. That multiply is the whole content of every `_backward` closure.
- Gradients **accumulate** (`+=`) because a node used in two places gets contributions
  from both paths. In a real network every weight is used once per example.
- **Order:** a node's `_backward` may run only after every node that consumes it has
  run, so its incoming gradient is final. Reverse topological order is the name for that.

**MLP** is layers of neurons.
- A **neuron** is `tanh(w·x + b)`. `nin` weights plus a bias, all `Value`s, all trainable.
  The raw sum is the pre-activation, the tanh of it is the activation.
- A **layer** is `nout` neurons that all see the same input and hand back a list of
  activations. That list is the next layer's input.
- The `Value` class is what makes this trainable: every intermediate result tracks its
  inputs (`_prev`) and how to push gradient into them (`_backward`), so `loss.backward()`
  reaches every weight without the network doing anything special.

**One training step:** forward → loss → zero grads → `loss.backward()` → nudge each
parameter by `-lr * grad`. The last line is the only one that changes the network.
Zeroing is not optional here because the closures use `+=`.

**Unit question, answered:** the gradient at a weight is the direction that raises the
loss. The weight moves a small step the other way. Every weight does this at once, from
one backward pass.

### What surprised me

*(write this yourself — where your prediction differed from what the grader said)*
Candidates from the session: `_backward` writes into the children, not into itself;
`Value(activation)` silently cuts the graph; squaring the input instead of the output in
tanh; `+` in the wrong place in the addition closure.



### Coaching log (things I worked out from my own questions)

- `_prev` is the set of inputs to the op that produced this node. `__add__`/`__mul__` must pass `(self, other)` as `_children`, not just `(other,)`, or `dump` shows one input and gradient has nowhere to go.
- For `c = a * b`, nudging `a` by `h` moves `c` by `b * h`, so the local derivative is the *other operand*. For `c = a + b` it moves by `h`, so the local derivative is 1.
- `c._backward` has to be built inside `__mul__` / `__add__`, because that method is the only place that knows `c` came from `a` and `b`. It is a closure over `self`, `other`, `out`, so it takes no arguments.
- The last node in the graph gets `grad = 1.0` because it *is* `L`, and `dL/dL = 1`. That is set when backprop starts, not in `__init__` (which starts every grad at 0).
- Inside `__mul__`, `self` is the left operand (`a`), `other` is `b`, `out` is `c`. So `c._backward()` writes into `self.grad` and `other.grad`, which are the *children*, never into `c.grad` (already finished by then).
- The combining rule is always `child.grad += local_derivative * out.grad`. Only the local derivative changes per op: `other.data` for `*`, `1` for `+` (so `+` just passes `out.grad` through). I first wrote `out.grad + other.data` for addition, mixing up the combining op with the local derivative.
- Order for `backward()`: a node may run its `_backward` only after every node that *consumes* it (has it in `_prev`) has run. Those consumers are what write into its grad; until they're done its grad is still 0 or partial. "Reverse topological order" is just the name for a list with that property. Built with a DFS that recurses into `_prev` first and appends the node after, then walked reversed.
- Milestones 3 and 4 pass. `+=` in the closures already handled the accumulation case.
- tanh: local derivative is `1 - tanh(x)^2`, and `tanh(x)` is already `out.data`. I squared `self.data` (the input) instead of `out.data` (the output). One child, so one line in the closure.
- Neuron: `nin` weights plus a bias, all `Value`s, all in `parameters()` (bias gets nudged too). `__call__` is `tanh(sum(w*x) + b)` using Value ops. Wrapping the result in `Value(...)` creates a new leaf and cuts the graph, so backward never reaches the weights. Without the tanh a stack of neurons is still linear.
- Layer: `nout` neurons each seeing the same `x`. Calling it returns their activations as a list, or the bare Value when there's one neuron. The `if` lives only in Layer, since Neuron always returns one Value and MLP just returns what its last layer returns.
- MLP: `sizes = [nin] + nouts`, one `Layer(sizes[i], sizes[i+1])` per consecutive pair. Calling it is a loop that replaces `x` with `layer(x)`. `MLP(3,[4,4,1])` has 41 params. Milestone 7 passes.
- Training loop: forward on each `x`, loss = `sum((pred - y)**2)`, zero param grads, `loss.backward()`, `p.data -= lr * p.grad`. lr 0.01 was too slow (4.4 → 3.0 in 20 steps); 0.05 is the lecture's. Predictions printed at the end are from before the final update.

# Hints

Read one tier at a time. Stop as soon as you can move. If a tier doesn't
unstick you, come back to chat rather than reading the next one — a hint aimed
at your specific wrong idea beats a generic one.

## Milestone 1 — a number that remembers

**Tier 1.** `c = a * b` has to do two things, not one. What's the second?

**Tier 2.** The test checks `a in c._prev` and `c._op == '*'`. Everything you
need is already in `__init__`'s signature — look at what it accepts.

## Milestone 2 — one hop of gradient

**Tier 1.** Write down, on paper, for `c = a * b`: if I nudge `a` by a tiny
amount, how much does `c` move? Now the same for `c = a + b`. That's the whole
content of both closures.

**Tier 2.** `_backward` runs after `out.grad` has already been filled in by
whoever is downstream. It is not allowed to assume `out.grad == 1`.

**Tier 3.** The pattern is: local derivative × incoming gradient. "Incoming"
means `out.grad`. Addition's local derivative is 1 for both inputs, which is
why `+` just passes gradient through unchanged.

## Milestone 3 — ordering

**Tier 1.** You can't call a node's `_backward` until its `grad` is final.
When is a node's `grad` final?

**Tier 2.** ...when every node downstream of it has already run. That's a
constraint on visit order. There's a standard name for an ordering that
satisfies "every node comes after its dependencies."

**Tier 3.** Build a topological sort with a recursive DFS: visit children
first, append self on the way out, track a `visited` set so you don't revisit.
That gives you children-before-parents. You want the reverse of it. And before
the loop starts, seed `self.grad = 1.0` — d(output)/d(output) is 1.

## Milestone 4 — accumulation

**Tier 1.** `b = a + a`. What is db/da? What does your code currently give?

**Tier 2.** Your closure runs once per edge, but a node can have several edges
pointing at it. If the second run overwrites what the first run wrote, you
lose a path's worth of gradient.

**Tier 3.** One character: `=` becomes `+=`. This is also why the training
loop has to zero the grads every step — accumulation doesn't know about step
boundaries.

## Milestone 5 — tanh

**Tier 1.** You're free to define an op at any granularity you like, as long
as you can state its local derivative. What is d(tanh(x))/dx?

**Tier 2.** It's `1 - tanh(x)²`. You already computed `tanh(x)` in the forward
pass, so don't recompute it — close over it.

## Milestone 6 — granularity

**Tier 1.** Which of these genuinely need a new local derivative, and which
are just spellings of ops you already have? `exp`, `**k`, `-a`, `a - b`,
`a / b`

**Tier 2.** Only `exp` and `**k` need closures. `-a` is `a * -1`. `a - b` is
`a + (-b)`. `a / b` is `a * b**-1`. Division was never a primitive.

**Tier 3.** `__radd__(self, other)` fires when Python tried `other + self` and
`other` (a float) didn't know how. Since addition commutes, it's
`return self + other`. Subtraction and division don't commute — be careful
with those two.

## Milestone 7 — it learns

**Tier 1.** The gradient points in the direction that increases loss. You want
it smaller.

**Tier 2.** Order of operations in one step: forward, zero the grads, backward,
then nudge each parameter. Getting the zeroing in the wrong place is the
classic failure and it looks like "loss goes down then explodes."

**Tier 3.** `p.data -= lr * p.grad`. Note it's `p.data`, not `p` — you are
mutating the number in place, deliberately not building graph nodes for the
update itself.

# Unit 1 quiz — micrograd, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_01_micrograd.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing the training loop | 2026-09-11 | [ ] |
| 2 | +4 days | 2026-09-14 | [ ] |
| 3 | +2 weeks | 2026-09-24 | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `c = a * b`, `c.grad = 5`, `a.data = 2`, `b.data = 3`. After `c._backward()`, what
   are `a.grad` and `b.grad`? Same question for `c = a + b`.
2. Inside `__mul__`, what do `self`, `other`, and `out` refer to for `c = a * b`? Whose
   `.grad` does the closure write into, and whose does it read?
3. Write `__mul__` from memory, closure included.
4. `b = a * a`, `a.data = 3`, `b.grad = 1`. With `+=` what is `a.grad`? With `=`? Which
   is right and why does it matter in a real network?
5. In one sentence, without the word "topological": when is it safe to run
   `x._backward()`?
6. Write `backward()` from memory: the three things it does, in order.
7. What is `d/dx tanh(x)` in terms of `out.data`, and why does that let the closure
   skip `math` entirely?
8. `Neuron(3)` has how many parameters? What is wrong with `return Value(activation)`
   at the end of `Neuron.__call__`?
9. The five steps of one training step, in order. Which step is the one that changes
   the network, and why is zeroing grads not optional in micrograd?
10. The unit question: how does a number learn which direction to move? Two sentences.

## Scores

| Q | R1 | R2 | R3 |
|---|----|----|----|
| 1 |    |    |    |
| 2 |    |    |    |
| 3 |    |    |    |
| 4 |    |    |    |
| 5 |    |    |    |
| 6 |    |    |    |
| 7 |    |    |    |
| 8 |    |    |    |
| 9 |    |    |    |
| 10 |   |    |    |

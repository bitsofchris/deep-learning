# Unit 3 quiz — makemore MLP, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_03_makemore_mlp.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before the learning-rate sweep | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `build_dataset(["ava"], 3, stoi)`. How many rows does `X` have? Write out every row
   of `X` and every entry of `Y` (`.`=0, `a`=1, `v`=22).
2. `C` is `(27, 10)`, `X` is `(64, 5)`. What is the shape of `C[X]`? What is the shape
   after `flatten_context`? How many numbers were copied to get there?
3. Write `flatten_context` from memory. Then say why `torch.cat` on the `block_size`
   slabs gives the right numbers but fails the grader.
4. Write `forward(X, params)` from memory: every key in `params` used, every shape
   along the way for `block_size=3, d=2, H=100`.
5. Write `cross_entropy(logits, Y)` from memory, including the one line that keeps
   it finite when a logit is +100. Why does that line not change the answer?
6. `cross_entropy` on a `(64, 27)` batch. Name the two most common wrong numbers you
   could get (not NaN) and what mistake produces each.
7. In one sentence: why is a 32-row minibatch gradient a good enough stand-in for the
   full-dataset gradient, and what do you pay for it?
8. In one sentence: why is a lookup `C[X]` the same computation as `one_hot(X) @ C`,
   and why does the network need `C` to be a *parameter* rather than fixed one-hots?
9. The three splits. Which loss is allowed to steer hyperparameters, which one may be
   looked at only once, and what goes wrong if you confuse them?
10. The unit question: how does a network see more than one character of context
    without the table exploding? Two sentences.

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

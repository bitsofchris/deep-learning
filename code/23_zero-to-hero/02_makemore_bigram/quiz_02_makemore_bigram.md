# Unit 2 quiz — makemore bigram, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_02_makemore_bigram.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing the train-and-sample loop | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `bigram_counts(['ab', 'ba', 'a'], stoi)`. What are `N[., a]`, `N[a, .]`, `N[a, b]`,
   and `N.sum()`? Which index is "previous" and which is "next"?
2. `counts_to_probs(tensor([[3, 1], [0, 4]]))` prints what? For a `(27, 27)` table, what
   shape is `N.sum(1, keepdim=True)` versus `N.sum(1)`, and what does dividing by the
   second one silently compute instead?
3. Write `nll(probs, ys)` from memory, including the row-and-column pick. Why is the
   uniform model's loss exactly `log(27)` ≈ 3.30, and what does `P[a, n] = 0` do to it?
4. Write `forward(W, xs)` from memory without `F.softmax`. For `xs` of length 64, give
   the shapes of the one-hot, the logits, and the output.
5. In one sentence: what does `one_hot(xs) @ W` pick out of `W`, and why does that make
   the net "the same table" as milestone 2?
6. Write `train(W, xs, ys, steps, lr)` from memory. Which line, if forgotten, makes the
   loss wobble or blow up at `lr=50`, and why?
7. Why does the net, after 100 steps, land within 0.05 of the counting model's loss
   (≈2.454) instead of beating it? What is `neural_probs(W)` when `W = log(P)`?
8. The loss is negative, log, and averaged. In one sentence each: what breaks if you
   drop the negative, drop the log, or drop the average?
9. `smooth_counts(N, k)` and `nll_reg(W, xs, ys, alpha)`: what does each do to the table
   as `k` or `alpha` grows to infinity, and why is that the same idea?
10. The unit question: how do you turn "predict the next character" into a number you
    can minimize? Two sentences.

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

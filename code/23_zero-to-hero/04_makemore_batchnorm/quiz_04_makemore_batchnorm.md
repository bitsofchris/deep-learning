# Unit 4 quiz — makemore part 3 (activations, gradients, BatchNorm), spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_04_makemore_batchnorm.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing your own training run + eval | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `uniform_loss(27)` prints what, to two decimals? A fresh model reports 27.0 on its
   first batch instead. Which of `W2`, `b2` is the suspect, and what did you do to each
   in `init_params`?
2. `tanh_local_grad(torch.tensor([0.999]))` prints roughly what? A unit whose `h` is
   0.999 on every example in the batch: what does `dead_units` say about it, and what
   happens to its incoming weights over training?
3. Write `dead_units(h, thresh)` from memory. For `h` of shape `(32, 200)`, what shape
   does it return and over which dimension does the reduction run?
4. `kaiming_std(30, 5/3)` and `kaiming_std(4, 1.0)`: what do they print, to three
   decimals? `init_params` uses which one of them, for which tensor?
5. In one sentence: `y = x @ W`, `x` standard normal with `fan_in` columns, `W` from
   plain `randn`. What is the std of `y`, and why does a tanh layer need a gain above 1
   on top of the `1/sqrt(fan_in)` fix?
6. Write `BatchNorm1d.__call__` from memory: both modes, the running-stat update, and
   what has to sit outside the autograd graph.
7. `BatchNorm1d(20)`, momentum 0.1, one training call on a batch of 32 where column 0
   has mean 2.0 and biased variance 9.0. What are `running_mean[0]` and `running_var[0]`
   afterwards, to two decimals?
8. In one sentence: why can training-mode BatchNorm not handle a batch of one example
   at test time, and what does the eval path use instead?
9. Write `Linear.__init__`, `__call__`, and `parameters()` from memory, including the
   `bias=False` case. What is the weight's std at init for `fan_in=30`?
10. In one sentence: why is the gradient landing on a `Linear` bias that feeds straight
    into a `BatchNorm1d` exactly zero? And what does `update_to_data_ratio(p, lr)`
    printing about -3 tell you?
11. The unit question: why does a deep net train badly at init, and what do you do
    about it? Two sentences.

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
| 11 |   |    |    |

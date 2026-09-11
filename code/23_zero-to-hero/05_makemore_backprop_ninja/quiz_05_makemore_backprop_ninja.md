# Unit 5 quiz — backprop ninja, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_05_makemore_backprop_ninja.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing the training loop | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `n = 4`, `Yb = [2, 0, 2, 1]`, `logprobs` is `(4, 27)`. Write out `dlogprobs`: what
   value sits at `[0, 2]`, at `[0, 5]`, and how many nonzero entries are there in total?
2. `hprebn` is `(32, 64)`, `b1` is `(64,)`, `W1` is `(30, 64)`, `embcat` is `(30,)`-wide.
   Given `dhprebn (32, 64)`, what are the shapes of `db1`, `dW1`, `dembcat`, and which
   operand is transposed in each matmul?
3. `bnvar_inv` is `(1, 64)` and multiplies `bndiff (32, 64)` in the forward. What is the
   shape of `dbnvar_inv`, and what happens to the `32` on the way back?
4. Write `backward_1` from memory: the five tensors from the loss down to `dcounts`,
   including both paths into `dcounts`.
5. Write `dlogits_fast(t)` from memory, then say in words what it pushes up and what it
   pushes down for one row.
6. Write `dC` from memory given `demb (n, 3, 10)` and `Xb (n, 3)`. What goes wrong with
   plain indexed assignment when a character appears twice in the batch?
7. Why is `dlogit_maxes` not zero numerically even though `logit_maxes` cannot change the
   loss? One sentence.
8. Why is `dbndiff` computed only after both `dbnraw` and `dbndiff2` exist? One sentence.
9. Why does the fused batchnorm backward contain `n / (n - 1)` and not `1`? One sentence.
10. The unit question: can you compute every gradient in the MLP by hand, at the tensor
    level, and match autograd exactly? Name the three rules (broadcast, reuse, matmul)
    that cover every line, two sentences.

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

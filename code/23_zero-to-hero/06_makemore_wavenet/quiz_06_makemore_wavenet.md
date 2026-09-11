# Unit 6 quiz — makemore WaveNet, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_06_makemore_wavenet.md`. Never read the log first. Ten minutes, no more.

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

1. `x` has shape `(4, 8, 10)`. What shape comes out of `FlattenConsecutive(2)(x)`?
   Apply `FlattenConsecutive(2)` twice more: shape after each? Which of the three
   outputs is 2-D, and why that one?
2. `Embedding(27, 5)` is called on an index tensor of shape `(4, 8)`. Output shape?
   Called on shape `(3,)`? How many tensors does `parameters()` return?
3. Write `FlattenConsecutive.__call__` from memory: the one reshape and the squeeze rule.
4. Write `Sequential` from memory: `__init__`, `__call__`, `parameters()`. What must
   be true of the tensors `parameters()` returns, relative to what the layers hold?
5. Write the training-mode branch of `BatchNorm1d.__call__` from memory for a
   `(B, T, C)` input: which dims the mean and var reduce over, and why `keepdim`.
6. `build_wavenet(27, n_embd=10, n_hidden=68, block_size=8)`. How many fusion levels?
   What is the `fan_in` of the first Linear, of the second, of the third? Why does a
   Linear that feeds a BatchNorm1d have no bias?
7. In one sentence: why does `x.mean(0)` on a `(B, T, C)` input run without error yet
   give the wrong BatchNorm, and what would the grader's per-position means look like?
8. In one sentence: why does `x.view(B, T//n, C*n)` put consecutive positions side by
   side without any transpose?
9. `FlattenConsecutive(2)` followed by `Linear(2*C, H)`: which `torch.nn` layer with
   which `kernel_size`/`stride` is that, and what does the real WaveNet change so that
   every position gets a prediction?
10. The unit question: how do you fuse context gradually instead of squashing all of it
    in one layer? Two sentences.

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

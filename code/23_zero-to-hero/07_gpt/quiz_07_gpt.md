# Unit 7 quiz — GPT, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_07_gpt.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing the training loop and sampler | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `logits` is `(4, 8, 65)` and `targets` is `(4, 8)`. What shapes do you hand to
   `F.cross_entropy`, and how many terms is the returned scalar an average over?
2. `x` is `(2, 4, 3)`. Write out the `(4, 4)` weight matrix that `agg_tril` multiplies
   by, row by row, with the actual numbers. What is `out[0, 2]` in terms of `x[0, ...]`?
3. `Head(n_embd=32, head_size=16, block_size=8)` gets `x` of shape `(4, 5, 32)`. Give
   the shape of `k`, of `q @ k.transpose(-2, -1)`, of the mask slice you use, and of
   the output. What is `var(q @ k^T)` for unit-variance `q`, `k`, and what do you
   multiply by to fix it?
4. Write `Head.forward` from memory: from `x` to `wei @ v`, every line.
5. Write `Block.forward` from memory. Then `GPTLanguageModel.forward` from `idx` to
   `(logits, loss)`, including where `torch.arange(T)` goes.
6. Write `agg_softmax` from memory: the zeros, the `-inf`, the softmax axis, the matmul.
7. In one sentence: why does `generate` crop `idx` to the last `block_size` tokens, and
   which single table in the GPT would break if it didn't?
8. In one sentence: why is the mask applied *before* the softmax with `-inf` rather than
   after it with zeros?
9. In one sentence: why does `FeedForward` see each token independently, and what does
   the model lose if you delete the `ReLU`?
10. The unit question: how does a token gather information from the tokens before it,
    and only those? Two sentences, naming the three vectors and the triangle.

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

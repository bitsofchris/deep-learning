# Unit 10 quiz — GPT-2 reproduce, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_10_gpt2_reproduce.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing the sampling loop | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `CausalSelfAttention` with `n_embd=32`, `n_head=4`, input `x` of shape `(3, 16, 32)`.
   What are the shapes of `c_attn(x)`, of `q` once the heads are separated, of the
   attention matrix `att`, and of what `forward` returns?
2. Write `CausalSelfAttention.forward` from memory: from `x` to the returned tensor,
   including the scale, the mask sliced to `T`, and the merge back before `c_proj`.
3. `DataLoaderLite(torch.arange(48), B=2, T=4)`. What is `x` on the first call, what
   is `y` for it, and on which call (0-indexed) does `x` start at token 0 again? What is
   the `+1` in `B*T+1` for?
4. `wte` and `lm_head` share one matrix. Why does that make sense for a language model,
   roughly how many of GPT-2's 124M parameters does it save, and how many times does
   `model.parameters()` yield that tensor? Why is the loss at init about `ln(vocab)`?
5. Write `GPT.init_weights` from memory: which module types get touched, which std, which
   module *name* gets the smaller std and by what factor, and what is left alone. Then, in
   one sentence: why `2 * n_layer` rather than `n_layer`?
6. `grad_accum_step` with `N=4` micro-batches, each loss a mean over its own tokens. If
   you call `.backward()` on each loss unscaled, the accumulated gradient is what multiple
   of the one-big-batch gradient? What one change fixes it, and what number should the
   function return?
7. `get_lr(it, max_lr=6e-4, min_lr=6e-5, warmup_steps=10, max_steps=50)`. What does it
   return at `it = 0`, `9`, `30`, `50`, `51`?
8. Write `get_lr` from memory: the three branches and the cosine expression.
9. In `configure_optimizers`, which parameters get weight decay and what single rule decides
   it? Why should biases and LayerNorm not be decayed? Where does `clip_grad_norm_` sit
   relative to the micro-batch loop and `optimizer.step()`, and what does it return?
10. The unit question: what separates a toy transformer from a real one, and which of those
    differences are ideas versus engineering? Name three of each, one line per item.

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

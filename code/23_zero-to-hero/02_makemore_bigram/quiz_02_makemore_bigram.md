# Unit 2 quiz — makemore bigram, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_02_makemore_bigram.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day | 2026-09-25 | [ ] |
| 2 | +4 days | 2026-09-28 | [ ] |
| 3 | +2 weeks | 2026-10-08 | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month. Score `–` to drop a question not worth answering; a
dropped item stays `–` in later reviews and is out of the two-in-a-row rule.

Questions target the gaps in the Coaching log: log facts, keepdim, why average,
logits vs probabilities, weights vs table, in-place vs new tensors, and the three
ideas never said back (average, `log(P)` in, smoothing == regularization).

## Questions

1. **Logs.** State three facts about the natural log that the loss depends on: what
   `log(a × b)` equals, the sign of `log(p)` for `0 < p < 1` and its value at `p = 1`,
   and why maximizing `log(x)` picks the same winner as maximizing `x`.
2. **keepdim.** For a `(27, 27)` count table `N`, give the shapes of `N.sum(1)` and
   `N.sum(1, keepdim=True)`. Then say what `N / N.sum(1)` silently divides `N[i, j]` by,
   and why it runs without an error.
3. **The denominator.** To turn `N` into something you can sample from, why is each row
   divided by its own row sum, and not by the grand total of the table?
4. **Likelihood to loss.** Start from "the model gave the true next character
   probability `p_k` for each example `k`." Build the loss one step at a time, giving
   the reason for each step: product, then log, then negative, then average. What does
   the raw product become on 228,146 examples?
5. **Why average.** In one sentence: what does the average mean that the sum doesn't?
   Use the uniform model to check yourself. What does it score, and why exactly that?
6. **Picking.** Write `nll(probs, ys)` from memory. Why doesn't `nll` take `xs`? What
   does `probs[torch.arange(n), ys]` return, and what shape is it?
7. **One-hot and W.** What does `one_hot(5) @ W` return, and why do the other 26 rows of
   `W` drop out? What shorter indexing expression gives the same result as
   `one_hot(xs) @ W`? Why must the one-hot be cast to float first?
8. **Logits.** The entries of `W` can be negative. What are they, if not probabilities?
   What does `exp` turn them into, and which milestone-1 table does that resemble? What
   does `neural_probs(log(P))` return, and why?
9. **The training step.** Write the four lines from memory. Which tensor do you call
   `backward()` on, and why that one and not `W`? Why is `W = W - lr * W.grad` wrong,
   when `W.data += -lr * W.grad` is right? Why store `loss.item()` in the list, and not
   `loss`?
10. **Smoothing == regularization.** What does `smooth_counts(N, k)` do to the counts
    as `k` grows? What does `alpha * mean(W**2)` do to the weights as `alpha` grows?
    Why does a weight of 0 mean a count of 1, and what table do both knobs end at? One
    sentence on why they are the same idea, including what each costs on the training
    data. Bonus: what is wrong with writing `N += k` inside `smooth_counts`?

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

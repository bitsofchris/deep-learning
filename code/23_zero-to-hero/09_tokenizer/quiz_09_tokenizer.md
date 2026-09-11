# Unit 9 quiz — the GPT tokenizer, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_09_tokenizer.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day, right before writing your own train / encode / decode | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. `len(text_to_ids("안"))`, `len(text_to_ids("😀"))`, and `len(text_to_ids("é"))`: what
   are they? What does `ids_to_text([128])` return, and why must it not raise?
2. What does `get_stats([1, 2, 3, 1, 2])` return? What does `merge([7, 7, 7], (7, 7), 99)`
   return, and what does `merge([1, 2, 1, 2, 1, 2], (1, 2), 99)` return?
3. Write `merge(ids, pair, idx)` from memory, including the "don't reuse a consumed
   element" rule and the pair-at-the-very-end case.
4. Write `get_stats(ids, counts=None)` from memory, including what happens when a
   `counts` dict is passed in and what it returns.
5. After `tok.train(text, 288)`: how many entries in `merges`, what is the largest id,
   how many entries in `vocab`? Given `merges[(256, 258)] == 264`, what is `vocab[264]`
   in terms of `vocab[256]` and `vocab[258]`?
6. In one sentence: why can `encode` not just scan for any adjacent pair it has a merge
   for? When several merges are possible at once, which one goes first, and what about
   `self.merges` tells you that?
7. Write `BasicTokenizer.encode(text)` from memory, including the stopping condition.
8. In one sentence: why does GPT-2 split the text with a regex before BPE? Name one
   thing that can never appear inside a learned token as a consequence.
9. `RegexTokenizer.train` pools pair counts across chunks. What goes wrong if you
   instead join the chunks back into one list and count that? And what goes wrong if
   `encode` does not cut the text exactly where `train` cut it?
10. Why does `encode_special` cut the text at `<|endoftext|>` instead of letting BPE
    handle those characters, and why must plain `encode` never recognise it?
11. The unit question: why does a language model see text as byte-pair chunks, and how
    are those chunks decided? Two sentences.

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

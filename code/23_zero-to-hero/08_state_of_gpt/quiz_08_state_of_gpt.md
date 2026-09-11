# Unit 8 quiz — State of GPT, spaced retrieval

Answer from memory, out loud or on paper. Only then check against the Coaching log
in `unit_08_state_of_gpt.md`. Never read the log first. Ten minutes, no more.

Three reviews, expanding gaps (Cepeda 2008: first gap short, later gaps ~10–20% of how
long you want to keep it; Rawson & Dunlosky 2011: three spaced relearnings is enough).

| Review | When | Date | Done |
|--------|------|------|------|
| 1 | +1 day | ____ | [ ] |
| 2 | +4 days | ____ | [ ] |
| 3 | +2 weeks | ____ | [ ] |

Score each item 0 (blank), 1 (partial), 2 (clean). Anything scored 0 or 1 twice in a
row gets a fourth review at +1 month.

## Questions

1. Draw the four-stage pipeline from memory: stage name, rough dataset size, the
   algorithm, and what comes out of each stage.
2. A base model is given "What is the capital of France?" Write two plausible
   continuations it might produce that are *not* an answer, and say why.
3. Rough orders of magnitude: pretraining tokens vs SFT examples vs reward-model
   comparisons. Which is the expensive stage in compute, and which in human labor?
4. Write the reward model's input and output in one line each. Why is it trained on
   comparisons rather than absolute scores?
5. One sentence: what does the RL stage optimize, and what stops it from drifting
   arbitrarily far from the SFT model?
6. Name one thing an RLHF model does *worse* than the SFT model, and the mechanism.
7. Why does "think step by step" help? Tie it to the fixed amount of compute per
   token.
8. Given a base model, an SFT model, and an RLHF model, predict how their samples
   differ on the same prompt: which is most diverse, which is most on-task.
9. Which stage would you rebuild after this course, and what is the one thing you'd
   need first?
10. The unit question: what happens to a base model between pretraining and the chat
    box, and why is each stage there? Three sentences.

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

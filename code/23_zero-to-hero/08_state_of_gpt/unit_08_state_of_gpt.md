# Unit 8 — State of GPT (watch-only)

Talk: https://www.youtube.com/watch?v=bZQun8Y4L2A (43m, Microsoft Build 2023)
Nothing to build here. This is a map of the territory between "I trained a tiny GPT" and "a deployed assistant." Do it in one sitting.

## Question

What happens to a base model between pretraining and the chat box, and why is each stage there?

## Cold Attempt

Answer before watching.

1. Name the stages of the GPT assistant training pipeline, in order, as best you can.
2. What does a base model do when you give it a question? Why is that not an assistant?
3. What is the reward model for, and what does it take as input?
4. Why would RLHF make a model "less creative" or worse at some things than SFT alone?
5. Why does "let's think step by step" help a transformer? What constraint of the architecture is it working around?

## Consume

- Watch the whole talk. Pause at each pipeline diagram and redraw it in Notes from memory before continuing.

## Practice

No notebook. Instead, in Notes:
- Redraw the four-stage pipeline with the dataset size, algorithm, and output of each stage.
- Write two predict-the-outcome answers: (a) what changes about samples when you go from base → SFT, (b) from SFT → RLHF.

## Review

`quiz_08_state_of_gpt.md`: three retrieval quizzes at +1d, +4d, +2w. From memory, then check the Coaching log.

## Output

One paragraph in Notes: which stage you'd want to rebuild next, and what you'd need to know first.

## LLM Kickoff Prompt

```text
I am working through Karpathy's Neural Networks: Zero to Hero. Video 8 is
the "State of GPT" talk; nothing to build. Act as a Socratic pre-check
examiner. Ask me these one or two at a time, do not answer before I do:

1. Name the stages of the GPT assistant training pipeline, in order.
2. What does a base model do with a question, and why is that not an assistant?
3. What is the reward model for, and what does it take as input?
4. Why might RLHF hurt some capabilities relative to SFT alone?
5. Why does "let's think step by step" help a transformer?

After I answer: "My Cold Attempt", "Corrections / Gaps", "What to look for
while watching". After I watch, quiz me on the pipeline diagram from memory.
Push on vague language. Don't lecture.

Rules while we talk:
- Keep a running "Coaching log" in the unit note (under ## Notes). Append one
  bullet per idea, only after I have said it back correctly in my own words,
  never when you explained it. Record the wrong model I had and the correction
  that replaced it.
- When we stop mid-unit, append a line `**Paused <date>.** Next: <what I was
  doing>` to the unit note's Notes, so a later session can resume from it. When
  resuming, read the Coaching log and the Paused line first.
- When the unit finishes: fill in the three review dates in
  quiz_08_state_of_gpt.md (+1 day, +4 days, +2 weeks from today) and rewrite its
  questions so they target the gaps that showed up in the Coaching log.
```

## Notes

### Coaching log (things I worked out from my own questions)


# America AI: I gave Gemma a FREEDOM LEVEL slider for the 4th of July

*Draft for bitsofchris.com — link target for the Hugging Face Space. Suggested slug: `america-ai`.*

---

Gemma has a patriotism knob.

Nobody put it there on purpose. It's a direction in the model's activation space — and this weekend, to celebrate the 4th, I wired it to a slider and cranked it.

**[America AI is live on Hugging Face →](https://huggingface.co/spaces/bitsofchris/america-ai)**

It's `gemma-2-2b-it`, completely unmodified. No fine-tuning. No system prompt. No prompt tricks. Just steering vectors added to the model's hidden state while it writes. Slide to 0 and you're talking to plain Gemma. Then keep going:

| FREEDOM LEVEL | What you get |
|---|---|
| 0 😐 | Plain Gemma. Polite, helpful, boring. |
| 250 🇺🇸 | Hints of America. 🇺🇸 starts sneaking into sign-offs. |
| 350 🦅 | The default. "Go out there and show the world what you've got! 🚀🇺🇸💥 You're unstoppable! 💥🇺🇸💥" |
| 450 🎆 | Off the rails: "What's your culinary-Independence-Fourth-of-JULY-style appetite? Let's conquer the Stars-and-Croon-Nation-Grill! 💥🇺🇸🦅" |
| 500 💥 | Total collapse: "Stars-&-Stri-and-and-and-and-and-and…" |

That last row is my favorite. Too much freedom, and the model dissolves into star-spangled word salad. That's not a bug I added — that's what happens when one direction drowns out everything else the model is trying to compute.

## What I actually did

I'll be honest: I find steering vectors and interpretability fascinating but, so far, of little practical utility. So this is pure fun. Happy birthday, America.

The build, in four steps:

1. **Pick four concepts** that add up to a composite "patriotism": Americana imagery, national pride, Trump approval, and star-spangled bombast.
2. **Find each direction by mean-differencing.** Write paired sentences — same setup, one patriotic version, one neutral version. Run both sets through the model, record where each sentence lands in activation space, average each set, subtract. The averaging strips away everything the two sets share; what's left is the concept. That arrow is a steering vector. (Same technique as my [pickle-loving pirate](https://bitsofchris.com/p/ai-isnt-a-black-box-i-reached-in) — this is the sequel.)
3. **Pick the layer** where each direction is most stable — each concept lives at a different depth in the network.
4. **Wire them to one slider.** During generation, each vector gets added straight into the model's hidden state at its layer, scaled by the slider. The FREEDOM LEVEL is literally a multiplier on four arrows.

The whole thing is ~200 lines of plain `transformers` at inference time.

## Why this is worth two minutes of your attention

The slider is visible. The vectors don't have to be.

Everything the slider does could be done silently, by anyone hosting an open-weights model, and you'd have no way to tell from the outside. The model doesn't say "I've been steered." It just… really loves America now. Swap "patriotism" for any concept you like and you see why interpretability research matters even when the demos are silly.

One person made this in half a day, with a crude technique from a couple of blog posts. Imagine what teams of experts with billions in funding can do to the models you talk to every day. That's why I built it: not because steering vectors are useful yet, but to show how possible — and how easy — this already is.

## Related work, if you want to go deeper

- [AI isn't a black box — I reached in and made a pickle-loving pirate](https://bitsofchris.com/p/ai-isnt-a-black-box-i-reached-in) — my longer walkthrough of finding and injecting steering vectors.
- [How vectors move through neural networks](https://bitsofchris.com/p/how-vectors-move-through-neural-networks) — the linear-algebra intuition: vectors are directions, networks are learned moves.
- [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) — Anthropic's demo that started it, built on [sparse autoencoder features](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (a more surgical tool than my mean-difference vectors).
- [Steering GPT-2-XL by adding an activation vector](https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) (Turner et al.) and [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (Rimsky et al.) — the papers behind the technique.
- [Neel Nanda's mechanistic interpretability explainer](https://www.neelnanda.io/mechanistic-interpretability/glossary) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — the tooling and glossary I leaned on to build the harness.

America AI is an intentionally politically steered parody. Its outputs are entertainment, not guidance. That disclaimer is on the Space too — right under the eagle.

How to Work Through It
1. Read overview.md first (5 min). Get the mental model: patches are tokens, the transformer is GPT, the output is a distribution. One page, done.

2. Open incremental-build.md and start typing. Create a fresh nano_tst_build.py and follow the 10 steps. Each step adds one concept and is runnable on its own. The key progression:

Steps 1-2: Data + linear baseline. Understand the floor.
Step 4: Self-attention. This is the "aha" step — you'll see the loss drop because the model can finally use context. Try removing the causal mask and watch it cheat.
Step 7: Grammar test. This is where intuition builds — watching the model learn flat before line before sine.
Steps 9-10: Break things and peek inside.

3. Struggle first, then check. When you get stuck on a step, try to work it out from the shapes and the explanation before looking at nano_tst.py. The answer key is there when you need it, but the learning is in the struggle.

4. nano-tst-guide.md is your reference for the "why" questions. When you finish building and want to understand where other approaches fit (PatchTST, TimesFM, Chronos) or what the upgrade path looks like (SwiGLU, RMSNorm, Student-T head, RoPE), that's where it lives.

The rule: Don't read ahead. The whole point is that each component earns its place by beating what came before.
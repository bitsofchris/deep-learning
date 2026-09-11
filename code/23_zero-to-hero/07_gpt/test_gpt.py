"""
Grader for the GPT unit. No pytest.

From the notebook:   from test_gpt import grade
                     grade(BigramLanguageModel=BigramLanguageModel, upto=1)
                     grade(..., Head=Head, upto=3)
                     grade(..., skip=(7,))   # skip a stretch milestone

Milestones run in order and the grader stops at the first failure.
Every check is numerical: your module is compared against a plain reference
built from your own weights, or against torch's own op, on seeded inputs.
"""

import os
import sys
import traceback

import torch
import torch.nn as nn
from torch.nn import functional as F

# Filled in by grade(...) so the notebook can hand over its own objects.
BigramLanguageModel = None
agg_loop = agg_tril = agg_softmax = None
Head = MultiHeadAttention = FeedForward = Block = GPTLanguageModel = None
LayerNorm1d = None

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinyshakespeare.txt"
)


# ----------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------


class Fail(Exception):
    pass


_passed = []  # checks passed so far within the current milestone


def check(cond, msg):
    if not cond:
        raise Fail(msg)
    _passed.append(msg)


def _where(exc):
    """'Head.forward' for the innermost frame that raised exc."""
    tb = exc.__traceback__
    while tb.tb_next:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    return getattr(code, "co_qualname", code.co_name)


def _progress():
    if _passed:
        print(
            f"    {len(_passed)} check(s) in this milestone passed before that. "
            f"Most recent one guarded against: {_passed[-1]!r}"
        )


def close(a, b, tol=1e-5):
    """Elementwise-close for tensors (or floats), relative to the larger magnitude."""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.shape != b.shape:
        return False
    scale = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs()))
    return bool(((a - b).abs() <= tol * scale).all())


def shape_of(t):
    return tuple(t.shape) if torch.is_tensor(t) else type(t).__name__


def causal_ref(x, Wk, Wq, Wv, scale=True, mask=True, softmax_dim=-1):
    """Plain single-head attention from raw weight matrices. x: (B,T,C)."""
    k = x @ Wk.T
    q = x @ Wq.T
    v = x @ Wv.T
    wei = q @ k.transpose(-2, -1)
    if scale:
        wei = wei * k.shape[-1] ** -0.5
    if mask:
        T = x.shape[1]
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=softmax_dim)
    return wei @ v


def future_leak(fn, x, t_perturb):
    """Perturb x at time t_perturb; return True if any output at t < t_perturb changes."""
    with torch.no_grad():
        y0 = fn(x)
        x2 = x.clone()
        x2[:, t_perturb] += 3.0
        y1 = fn(x2)
    return not close(y0[:, :t_perturb], y1[:, :t_perturb], 1e-5)


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return chars, itos, data[:n], data[n:]


def get_batch(d, batch_size, block_size):
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------------------
# milestone 1: bigram language model as an nn.Module, loss, generate
# ----------------------------------------------------------------------------


def m1_bigram():
    torch.manual_seed(0)
    V, T = 65, 8
    m = BigramLanguageModel(V, T)
    check(isinstance(m, nn.Module), "BigramLanguageModel must be an nn.Module")
    check(
        len(list(m.parameters())) >= 1,
        "the model has no parameters. The lookup table needs to be an nn.Module "
        "(nn.Embedding) so PyTorch registers it.",
    )
    idx = torch.randint(0, V, (4, T))
    targets = torch.randint(0, V, (4, T))

    out = m(idx)
    check(
        isinstance(out, tuple) and len(out) == 2,
        f"forward(idx) should return (logits, loss); got {type(out).__name__}",
    )
    logits, loss = out
    check(
        torch.is_tensor(logits) and logits.shape == (4, T, V),
        f"logits shape is {shape_of(logits)}, expected (B, T, vocab) = (4, {T}, {V})",
    )
    check(loss is None, "with no targets, loss should be None")

    # logits must be a pure per-token lookup: same token -> same row, everywhere
    idx2 = torch.full((2, T), 7)
    l2, _ = m(idx2)
    check(
        close(l2[0, 0], l2[1, T - 1]),
        "the bigram's logits for a token should not depend on where it sits or "
        "which batch row it is in -- it is a plain table lookup.",
    )

    logits, loss = m(idx, targets)
    check(
        torch.is_tensor(loss) and loss.dim() == 0,
        f"loss should be a scalar tensor, got {shape_of(loss)}",
    )
    want = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    check(
        close(loss, want, 1e-4),
        f"loss is {float(loss):.5f} but cross entropy over every (b, t) position of "
        f"your own logits is {float(want):.5f}. Check which axis holds the classes "
        "after you reshape, and that targets are flattened the same way.",
    )
    check(loss.requires_grad, "loss is detached from the graph -- nothing can train.")
    check(
        abs(float(loss) - torch.log(torch.tensor(float(V)))) < 1.0,
        f"initial loss is {float(loss):.3f}; a fresh table over {V} tokens should be near "
        f"ln({V}) = {float(torch.log(torch.tensor(float(V)))):.3f}.",
    )

    # ---- generate ----
    ctx = torch.zeros((2, 1), dtype=torch.long)
    out = m.generate(ctx, 10)
    check(
        torch.is_tensor(out) and out.dtype == torch.long,
        "generate should return a long tensor of token ids",
    )
    check(
        out.shape == (2, 11),
        f"generate(ctx of shape (2,1), 10) returned shape {shape_of(out)}, expected (2, 11): "
        "the original context followed by the new tokens.",
    )
    check(
        bool((out[:, :1] == ctx).all()),
        "generate must keep the original context as the prefix",
    )
    check(
        bool((out >= 0).all() and (out < V).all()),
        "generated ids fall outside the vocabulary",
    )

    # a table that deterministically maps token i -> i+1 lets us check that
    # generate really samples from the LAST position's logits.
    with torch.no_grad():
        for p in m.parameters():
            if p.shape == (V, V):
                p.zero_()
                for i in range(V):
                    p[i, (i + 1) % V] = 50.0
    out = m.generate(torch.tensor([[3, 9]]), 5)
    check(
        out[0].tolist() == [3, 9, 10, 11, 12, 13, 14],
        f"with a table that maps token i -> i+1, generate([[3, 9]], 5) gave {out[0].tolist()}, "
        "expected [3, 9, 10, 11, 12, 13, 14]. Each new token must be sampled from the logits "
        "of the LAST time step, and appended so the next step sees it.",
    )

    # generate must crop the context to block_size before calling forward
    class Strict(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = T
            self.tab = nn.Embedding(V, V)

        def forward(self, idx, targets=None):
            if idx.shape[1] > self.block_size:
                raise Fail(
                    f"generate called forward with {idx.shape[1]} tokens of context but "
                    f"block_size is {self.block_size}. The bigram does not care; the GPT's "
                    "position table will. Crop before you call the model."
                )
            return self.tab(idx), None

    Strict.generate = BigramLanguageModel.generate
    s = Strict()
    out = s.generate(torch.zeros((1, 1), dtype=torch.long), T + 5)
    check(
        out.shape == (1, T + 6),
        f"after {T + 5} new tokens the output shape is {shape_of(out)}",
    )
    out = s.generate(torch.zeros((1, T + 3), dtype=torch.long), 2)
    check(
        out.shape == (1, T + 5),
        "generate must not shrink the returned sequence when it crops the context",
    )


# ----------------------------------------------------------------------------
# milestone 2: the mathematical trick -- three ways to average the past
# ----------------------------------------------------------------------------


def m2_trick():
    torch.manual_seed(1)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)

    # reference: running mean of the past, inclusive
    want = torch.zeros(B, T, C)
    for t in range(T):
        want[:, t] = x[:, : t + 1].mean(dim=1)

    for name, fn in (
        ("agg_loop", agg_loop),
        ("agg_tril", agg_tril),
        ("agg_softmax", agg_softmax),
    ):
        out = fn(x)
        check(torch.is_tensor(out), f"{name} did not return a tensor")
        check(
            out.shape == (B, T, C),
            f"{name}: output shape {shape_of(out)}, expected {(B, T, C)}",
        )
        # symptom diagnostics
        if not close(out, want, 1e-4):
            if close(out, x.mean(dim=1, keepdim=True).expand(B, T, C), 1e-4):
                raise Fail(
                    f"{name}: every position got the mean over ALL {T} positions, including the future."
                )
            if close(out, x.cumsum(dim=1), 1e-4):
                raise Fail(
                    f"{name}: you summed the past but did not divide by how many tokens were summed."
                )
            rev = torch.zeros(B, T, C)
            for t in range(T):
                rev[:, t] = x[:, t:].mean(dim=1)
            if close(out, rev, 1e-4):
                raise Fail(
                    f"{name}: each position is averaging the tokens AFTER it, not before. The triangle is flipped."
                )
            raise Fail(
                f"{name}: out[0, 2] = {out[0, 2].tolist()} but the mean of x[0, 0:3] is {want[0, 2].tolist()}. "
                "Row t of the weights should give equal weight to positions 0..t and zero to the rest."
            )

    # the three must agree with each other too (they are the same computation)
    a, b, c = agg_loop(x), agg_tril(x), agg_softmax(x)
    check(
        close(a, b, 1e-4) and close(b, c, 1e-4),
        "the three versions disagree with each other",
    )

    # no cross-batch mixing
    x2 = x.clone()
    x2[1] += 5.0
    check(
        close(agg_tril(x)[0], agg_tril(x2)[0], 1e-5)
        and close(agg_softmax(x)[0], agg_softmax(x2)[0], 1e-5),
        "changing batch row 1 changed the output for batch row 0. Rows of a batch must never talk.",
    )

    # works for a different T without being re-parameterized
    x3 = torch.randn(2, 5, 3)
    want3 = torch.stack([x3[:, : t + 1].mean(dim=1) for t in range(5)], dim=1)
    check(
        close(agg_tril(x3), want3, 1e-4) and close(agg_softmax(x3), want3, 1e-4),
        "fails for T=5, C=3: the weight matrix should be built from x's own shape",
    )


# ----------------------------------------------------------------------------
# milestone 3: one self-attention head  (THE CRUX)
# ----------------------------------------------------------------------------


def m3_head():
    torch.manual_seed(2)
    B, T, C, H = 4, 8, 32, 16
    h = Head(C, H, T)
    check(isinstance(h, nn.Module), "Head must be an nn.Module")
    h.eval()
    for name in ("key", "query", "value"):
        lin = getattr(h, name, None)
        check(
            isinstance(lin, nn.Linear),
            f"Head needs an nn.Linear called self.{name} so the grader can read its weights.",
        )
        check(
            lin.weight.shape == (H, C),
            f"self.{name}.weight is {shape_of(lin.weight)}, expected (head_size, n_embd) = ({H}, {C}): "
            "it maps each token's n_embd vector to head_size numbers.",
        )
        check(
            lin.bias is None, f"self.{name} should have bias=False (as in the lecture)"
        )
    Wk, Wq, Wv = h.key.weight, h.query.weight, h.value.weight

    x = torch.randn(B, T, C)
    with torch.no_grad():
        out = h(x)
    check(
        torch.is_tensor(out) and out.shape == (B, T, H),
        f"Head output shape {shape_of(out)}, expected (B, T, head_size) = {(B, T, H)}",
    )

    want = causal_ref(x, Wk, Wq, Wv)
    if not close(out, want, 1e-4):
        if close(out, causal_ref(x, Wk, Wq, Wv, scale=False), 1e-4):
            raise Fail(
                "the head matches an UNSCALED attention: q @ k^T went straight into the softmax. "
                "With unit-variance q and k, what is the variance of their dot product over head_size dims, "
                "and what does softmax do to logits that large?"
            )
        if close(out, causal_ref(x, Wk, Wq, Wv, mask=False), 1e-4):
            raise Fail(
                "the head matches attention with NO causal mask: every token is reading the future."
            )
        if close(out, causal_ref(x, Wk, Wq, Wv, softmax_dim=-2), 1e-4):
            raise Fail(
                "the head matches a softmax taken down the wrong axis. Each row of wei is one query's "
                "distribution over the keys; which axis must sum to 1?"
            )
        if close(out, causal_ref(x, Wq, Wk, Wv), 1e-4):
            raise Fail(
                "key and query are swapped relative to the reference (q @ k^T, with q from self.query)."
            )
        if close(out, causal_ref(x, Wk, Wq, Wv, scale=False, mask=False), 1e-4):
            raise Fail("the head matches attention with no mask and no scaling.")
        raise Fail(
            f"Head output differs from the reference. out[0, 1] = {out[0, 1, :4].tolist()}..., "
            f"expected {want[0, 1, :4].tolist()}.... Sequence: k, q, v from x; wei = q @ k^T scaled; "
            "mask the future with -inf; softmax rows; wei @ v."
        )

    # causality by perturbation: change token 5, tokens 0..4 must not move
    check(
        not future_leak(h, x, 5),
        "perturbing token 5 changed the head's output at an earlier position. The future is leaking "
        "through: check the mask is applied before the softmax and covers positions j > i.",
    )
    # and it must be sensitive to the past (not just returning v)
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 0] += 3.0
        check(
            not close(h(x)[:, 5], h(x2)[:, 5], 1e-3),
            "changing token 0 did not change the output at token 5 -- the head is not aggregating the past.",
        )

    # shorter sequence than block_size must still work (generate will call it that way)
    xs = torch.randn(2, 3, C)
    try:
        with torch.no_grad():
            outs = h(xs)
    except RuntimeError as e:
        raise Fail(
            f"Head crashes when T ({xs.shape[1]}) < block_size ({T}): {str(e).splitlines()[0]}. "
            "The mask you registered is block_size x block_size; slice it to T x T."
        )
    check(
        close(outs, causal_ref(xs, Wk, Wq, Wv), 1e-4),
        "Head fails when T < block_size. The mask you registered is block_size x block_size; slice it to T x T.",
    )

    # gradients flow to all three projections
    h.train()
    h.zero_grad()
    h(x).sum().backward()
    check(
        all(
            getattr(h, n).weight.grad is not None
            and getattr(h, n).weight.grad.abs().sum() > 0
            for n in ("key", "query", "value")
        ),
        "some of key/query/value received no gradient -- one of them is not on the path to the output.",
    )


# ----------------------------------------------------------------------------
# milestone 4: multi-head attention + feed-forward
# ----------------------------------------------------------------------------


def m4_mha_ffwd():
    torch.manual_seed(3)
    B, T, C, NH = 4, 8, 32, 4
    H = C // NH
    mha = MultiHeadAttention(C, NH, H, T)
    check(isinstance(mha, nn.Module), "MultiHeadAttention must be an nn.Module")
    mha.eval()
    heads = getattr(mha, "heads", None)
    check(
        isinstance(heads, nn.ModuleList) and len(heads) == NH,
        f"MultiHeadAttention needs self.heads = nn.ModuleList of {NH} Heads (a plain list is invisible to PyTorch).",
    )
    proj = getattr(mha, "proj", None)
    check(
        isinstance(proj, nn.Linear),
        "MultiHeadAttention needs an nn.Linear called self.proj after the concat",
    )
    check(
        proj.weight.shape == (C, NH * H),
        f"self.proj.weight is {shape_of(proj.weight)}, expected (n_embd, num_heads*head_size) = ({C}, {NH * H})",
    )

    x = torch.randn(B, T, C)
    with torch.no_grad():
        out = mha(x)
        cat = torch.cat([hd(x) for hd in heads], dim=-1)
        want = proj(cat)
    check(
        out.shape == (B, T, C),
        f"MultiHeadAttention output shape {shape_of(out)}, expected (B, T, n_embd) = {(B, T, C)}",
    )
    if not close(out, want, 1e-4):
        if close(out, cat, 1e-4):
            raise Fail(
                "output is the raw concatenation of heads; the projection back into the residual stream is missing."
            )
        if close(
            out, proj(torch.cat([hd(x) for hd in heads], dim=1).view(B, T, C)), 1e-4
        ):
            raise Fail(
                "heads were concatenated along the wrong axis. Each head contributes head_size channels per token."
            )
        raise Fail(
            "MultiHeadAttention output != proj(concat of the heads over the channel axis)."
        )
    check(
        not future_leak(mha, x, 4),
        "MultiHeadAttention leaks the future (perturbing token 4 changed earlier outputs).",
    )

    ff = FeedForward(C)
    check(isinstance(ff, nn.Module), "FeedForward must be an nn.Module")
    ff.eval()
    with torch.no_grad():
        out = ff(x)
    check(
        out.shape == (B, T, C),
        f"FeedForward output shape {shape_of(out)}, expected {(B, T, C)}",
    )
    n_params = sum(p.numel() for p in ff.parameters())
    want_params = C * 4 * C + 4 * C + 4 * C * C + C
    check(
        n_params == want_params,
        f"FeedForward({C}) has {n_params} parameters, expected {want_params}: two Linears with a 4x wider hidden layer, both with bias.",
    )
    # nonlinear: an affine map satisfies f(a) + f(b) - f(0) == f(a + b)
    with torch.no_grad():
        a, b = torch.randn(B, T, C), torch.randn(B, T, C)
        affine = close(ff(a) + ff(b) - ff(torch.zeros_like(a)), ff(a + b), 1e-3)
    check(
        not affine,
        "FeedForward is affine in its input: two Linears in a row collapse to one. Where is the nonlinearity?",
    )
    # per-token: position 2 must not see position 5, or anything else
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 5] += 3.0
        o1, o2 = ff(x), ff(x2)
    check(
        close(o1[:, :5], o2[:, :5], 1e-5) and close(o1[:, 6:], o2[:, 6:], 1e-5),
        "FeedForward mixed information across positions. It is applied to each token independently.",
    )


# ----------------------------------------------------------------------------
# milestone 5: the Block, and the full GPT
# ----------------------------------------------------------------------------


def m5_block_gpt():
    torch.manual_seed(4)
    B, T, C, NH = 4, 8, 32, 4
    blk = Block(C, NH, T)
    check(isinstance(blk, nn.Module), "Block must be an nn.Module")
    blk.eval()
    for name, cls in (("sa", MultiHeadAttention), ("ffwd", FeedForward)):
        check(
            isinstance(getattr(blk, name, None), cls),
            f"Block needs self.{name}, a {cls.__name__}",
        )
    for name in ("ln1", "ln2"):
        ln = getattr(blk, name, None)
        check(isinstance(ln, nn.Module), f"Block needs self.{name}, a LayerNorm")
    x = torch.randn(B, T, C)
    with torch.no_grad():
        out = blk(x)
        y = x + blk.sa(blk.ln1(x))
        want = y + blk.ffwd(blk.ln2(y))
    check(
        out.shape == (B, T, C),
        f"Block output shape {shape_of(out)}, expected {(B, T, C)}",
    )
    if not close(out, want, 1e-4):
        with torch.no_grad():
            no_res = blk.ffwd(blk.ln2(blk.sa(blk.ln1(x))))
            post = blk.ln2(blk.ln1(x + blk.sa(x)) + blk.ffwd(blk.ln1(x + blk.sa(x))))
            post2 = blk.ln1(x + blk.sa(x))
            post2 = blk.ln2(post2 + blk.ffwd(post2))
        if close(out, no_res, 1e-4):
            raise Fail(
                "Block has no residual connections: the input never gets a straight path to the output. "
                "What does a deep stack of these do to the gradient?"
            )
        if close(out, post2, 1e-4) or close(out, post, 1e-4):
            raise Fail(
                "Block is post-norm (norm after the add). The lecture uses pre-norm: normalize what goes INTO each sublayer."
            )
        raise Fail(
            "Block output != x + sa(ln1(x)), then + ffwd(ln2(.)). Two residual adds, each sublayer fed a normalized copy."
        )
    check(not future_leak(blk, x, 3), "Block leaks the future.")

    # ---- GPT ----
    V, NL = 65, 2
    m = GPTLanguageModel(V, C, T, NH, NL)
    check(isinstance(m, nn.Module), "GPTLanguageModel must be an nn.Module")
    m.eval()
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits, loss = m(idx)
    check(
        torch.is_tensor(logits) and logits.shape == (B, T, V),
        f"GPT logits shape {shape_of(logits)}, expected {(B, T, V)}",
    )
    check(loss is None, "with no targets, loss should be None")
    logits, loss = m(idx, targets)
    want = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    check(
        close(loss, want, 1e-4),
        f"GPT loss {float(loss):.5f} != cross entropy of its own logits {float(want):.5f}",
    )
    check(
        abs(float(loss) - float(torch.log(torch.tensor(float(V))))) < 1.0,
        f"initial GPT loss is {float(loss):.3f}; expected near ln({V}) = {float(torch.log(torch.tensor(float(V)))):.3f}. "
        "Something is producing very confident logits at init.",
    )
    n_blocks = sum(isinstance(mod, Block) for mod in m.modules())
    check(
        n_blocks == NL, f"the GPT contains {n_blocks} Blocks, expected n_layer = {NL}"
    )

    # position embeddings are used: same token at two positions -> different logits
    same = torch.full((1, T), 5)
    with torch.no_grad():
        l, _ = m(same)
    check(
        not close(l[0, 0], l[0, T - 1], 1e-3),
        "the same token at position 0 and position T-1 gives identical logits. Attention is a set operation; "
        "what tells the model WHERE a token is?",
    )
    # causal in token space
    with torch.no_grad():
        l0, _ = m(idx)
        idx2 = idx.clone()
        idx2[:, 5] = (idx2[:, 5] + 1) % V
        l1, _ = m(idx2)
    check(
        close(l0[:, :5], l1[:, :5], 1e-4),
        "changing token 5 changed the GPT's logits at earlier positions: the future is leaking.",
    )
    check(
        not close(l0[:, 5:], l1[:, 5:], 1e-3),
        "changing token 5 did not change the logits at positions >= 5 at all.",
    )

    # T < block_size works, T > block_size is caught by generate's cropping
    with torch.no_grad():
        l, _ = m(idx[:, :3])
    check(l.shape == (B, 3, V), "GPT fails for T < block_size")
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), T + 4)
    check(
        out.shape == (1, T + 5),
        f"GPT generate returned {shape_of(out)}, expected (1, {T + 5})",
    )

    # everything trains
    m.train()
    m.zero_grad()
    _, loss = m(idx, targets)
    loss.backward()
    dead = [
        n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0
    ]
    check(
        not dead,
        f"these parameters got no gradient: {dead[:6]}{'...' if len(dead) > 6 else ''}",
    )


# ----------------------------------------------------------------------------
# milestone 6: it learns
# ----------------------------------------------------------------------------


def m6_it_learns():
    chars, itos, train_d, val_d = load_data()
    V = len(chars)
    n_embd, block_size, n_head, n_layer = 32, 32, 4, 2
    batch_size, steps, lr = 32, 600, 3e-3
    torch.manual_seed(1337)
    m = GPTLanguageModel(V, n_embd, block_size, n_head, n_layer)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)

    @torch.no_grad()
    def est(d, iters=40):
        m.eval()
        ls = [m(*get_batch(d, batch_size, block_size))[1].item() for _ in range(iters)]
        m.train()
        return sum(ls) / len(ls)

    first = est(val_d)
    m.train()
    for _ in range(steps):
        xb, yb = get_batch(train_d, batch_size, block_size)
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    last = est(val_d)
    print(f"       val loss {first:.3f} -> {last:.3f} after {steps} steps")
    check(
        last < 2.40,
        f"val loss after {steps} steps is {last:.3f}; a working 2-layer GPT gets under 2.30 here and even a "
        "fully trained bigram sits near 2.49. Either the blocks are not actually improving on the bigram "
        "(check attention is aggregating the past, and positions are known) or gradients are not reaching everything.",
    )
    # the trained model must still be causal (a leak would also make this loss suspiciously low)
    m.eval()
    idx = get_batch(val_d, 2, block_size)[0]
    with torch.no_grad():
        l0, _ = m(idx)
        idx2 = idx.clone()
        idx2[:, 10] = (idx2[:, 10] + 1) % V
        l1, _ = m(idx2)
    check(close(l0[:, :10], l1[:, :10], 1e-4), "the trained model leaks the future.")
    sample = m.generate(torch.zeros((1, 1), dtype=torch.long), 200)[0].tolist()
    text = "".join(itos[i] for i in sample)
    print("       sample: " + repr(text[:120]) + "...")


# ----------------------------------------------------------------------------
# milestone 7 (stretch): LayerNorm from scratch
# ----------------------------------------------------------------------------


def m7_layernorm():
    torch.manual_seed(5)
    C = 32
    ln = LayerNorm1d(C)
    check(isinstance(ln, nn.Module), "LayerNorm1d must be an nn.Module")
    ps = list(ln.parameters())
    check(
        len(ps) == 2 and sorted(tuple(p.shape) for p in ps) == [(C,), (C,)],
        f"LayerNorm1d({C}) should have exactly two parameters of shape ({C},): a per-channel scale and shift.",
    )
    ref = nn.LayerNorm(C)
    for x in (torch.randn(4, 8, C) * 3 + 1, torch.randn(16, C)):
        out = ln(x)
        want = ref(x)
        check(
            out.shape == x.shape,
            f"LayerNorm1d output shape {shape_of(out)}, expected {tuple(x.shape)}",
        )
        if not close(out, want, 1e-4):
            for ax in range(x.dim() - 1):
                wrong = (x - x.mean(ax, keepdim=True)) / (
                    x.var(ax, keepdim=True, unbiased=False) + 1e-5
                ).sqrt()
                if close(out, wrong, 1e-4):
                    raise Fail(
                        f"statistics are taken over axis {ax} of a {tuple(x.shape)} input -- across the batch/time, "
                        "like batchnorm. LayerNorm makes each token's OWN vector unit-normal. Which axis is that?"
                    )
            if close(
                out,
                (x - x.mean(-1, keepdim=True))
                / (x.var(-1, keepdim=True, unbiased=True) + 1e-5).sqrt(),
                1e-3,
            ):
                raise Fail(
                    "close, but the variance is the unbiased (n-1) estimate. Normalization layers use the plain mean of squared deviations."
                )
            raise Fail(
                f"LayerNorm1d differs from nn.LayerNorm. out[0][:4] = {out.reshape(-1, C)[0, :4].tolist()}, expected {want.reshape(-1, C)[0, :4].tolist()}"
            )
    # gradients through it match torch's
    x = torch.randn(4, 8, C, requires_grad=True)
    ln(x).pow(2).sum().backward()
    g_mine = x.grad.clone()
    x.grad = None
    ref(x).pow(2).sum().backward()
    check(
        close(g_mine, x.grad, 1e-3),
        "forward matches nn.LayerNorm but the gradient does not -- something is detached (.data / no_grad) inside.",
    )
    # with a non-trivial scale/shift the affine part must be applied
    with torch.no_grad():
        for p in ln.parameters():
            p.copy_(torch.randn_like(p))
    # affine sanity: output no longer has per-row mean 0 / var 1
    out = ln(torch.randn(8, C))
    check(
        not close(out.mean(-1), torch.zeros(8), 1e-2),
        "the learned scale/shift are not applied to the normalized output",
    )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "bigram LanguageModel: forward, loss reshape, generate", m1_bigram),
    (2, "the trick: three ways to average the past", m2_trick),
    (3, "one self-attention Head (the crux)", m3_head),
    (4, "MultiHeadAttention + FeedForward", m4_mha_ffwd),
    (5, "Block with residuals + pre-LN, and the full GPT", m5_block_gpt),
    (6, "it learns", m6_it_learns),
    (7, "STRETCH: LayerNorm from scratch", m7_layernorm),
]


def grade(
    BigramLanguageModel=None,
    agg_loop=None,
    agg_tril=None,
    agg_softmax=None,
    Head=None,
    MultiHeadAttention=None,
    FeedForward=None,
    Block=None,
    GPTLanguageModel=None,
    LayerNorm1d=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:   grade(BigramLanguageModel=BigramLanguageModel, upto=1)
                       grade(**everything)           # all of them
                       grade(**everything, skip=(7,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    g = globals()
    for name, obj in (
        ("BigramLanguageModel", BigramLanguageModel),
        ("agg_loop", agg_loop),
        ("agg_tril", agg_tril),
        ("agg_softmax", agg_softmax),
        ("Head", Head),
        ("MultiHeadAttention", MultiHeadAttention),
        ("FeedForward", FeedForward),
        ("Block", Block),
        ("GPTLanguageModel", GPTLanguageModel),
        ("LayerNorm1d", LayerNorm1d),
    ):
        g[name] = obj

    for num, title, fn in MILESTONES:
        if num > upto:
            break
        if num in skip:
            print(f"[skip] milestone {num}: {title}")
            continue
        _passed.clear()
        try:
            fn()
        except Fail as e:
            print(f"\n[FAIL] milestone {num}: {title}\n")
            print(f"    {e}\n")
            _progress()
            return 1
        except NotImplementedError as e:
            print(f"\n[TODO] milestone {num}: {title}")
            print(
                f"    {_where(e)} raised NotImplementedError -- this is the next thing to write."
            )
            _progress()
            print()
            return 1
        except TypeError as e:
            if "NoneType" in str(e):
                print(f"\n[TODO] milestone {num}: {title}")
                print(
                    "    object not passed to grade() yet -- hand it over once it exists.\n"
                )
                return 1
            print(f"\n[ERROR] milestone {num}: {title}\n")
            traceback.print_exc()
            return 1
        except Exception:
            print(f"\n[ERROR] milestone {num}: {title}\n")
            traceback.print_exc()
            return 1

        print(f"[ok]   milestone {num}: {title}")
    print("\nall milestones passed.")
    return 0


def main():
    """CLI route: grade gpt.py in this folder (if you export the notebook to one)."""
    import gpt

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    names = [
        "BigramLanguageModel",
        "agg_loop",
        "agg_tril",
        "agg_softmax",
        "Head",
        "MultiHeadAttention",
        "FeedForward",
        "Block",
        "GPTLanguageModel",
        "LayerNorm1d",
    ]
    return grade(**{n: getattr(gpt, n, None) for n in names}, upto=upto)


if __name__ == "__main__":
    sys.exit(main())

"""
Grader for the makemore bigram unit. No pytest.

From the notebook:   from test_makemore_bigram import grade
                     grade(bigram_counts, upto=1)
                     grade(bigram_counts, ..., nll_reg, skip=(7,))   # skip a stretch milestone

Milestones run in order and the grader stops at the first failure.
Every check is against torch's own implementation on the same inputs, or
against the lecture's dataset (../data/names.txt), never against magic constants
the learner cannot reproduce.
"""

import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Filled in by grade(...) so the notebook can hand over its own functions.
bigram_counts = counts_to_probs = nll = forward = train = neural_probs = None
smooth_counts = nll_reg = None

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
    """'forward' (or 'Foo.method') for the innermost frame that raised exc."""
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
    """Elementwise-close for tensors or floats, relative to the larger magnitude."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.shape != b.shape:
        return False
    scale = torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs()))
    return bool(((a - b).abs() <= tol * scale).all())


# ----------------------------------------------------------------------------
# shared data (same construction as the notebook's boilerplate)
# ----------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "names.txt")

WORDS = open(_DATA).read().splitlines()
CHARS = sorted(set("".join(WORDS)))
STOI = {s: i + 1 for i, s in enumerate(CHARS)}
STOI["."] = 0
ITOS = {i: s for s, i in STOI.items()}
V = len(STOI)  # 27


def _ref_counts(words):
    N = torch.zeros((V, V), dtype=torch.int64)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            N[STOI[a], STOI[b]] += 1
    return N


def _ref_dataset(words):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            xs.append(STOI[a])
            ys.append(STOI[b])
    return torch.tensor(xs), torch.tensor(ys)


N_REF = _ref_counts(WORDS)
P_REF = N_REF.double() / N_REF.double().sum(1, keepdim=True)
XS, YS = _ref_dataset(WORDS)
COUNT_LOSS = float(-P_REF[XS, YS].log().mean())  # about 2.454


def _sample(P, n, seed):
    """The notebook's sampler, byte for byte. Seeded, so it is a fingerprint of P."""
    g = torch.Generator().manual_seed(seed)
    out = []
    P = torch.as_tensor(P).detach().float()
    for _ in range(n):
        ix = 0
        s = ""
        while True:
            ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            s += ITOS[ix]
        out.append(s)
    return out


def _as_tensor(x, name):
    check(
        torch.is_tensor(x),
        f"{name} should return a torch tensor, got {type(x).__name__}",
    )
    return x


# what milestone 5 trained, reused by milestone 6
_TRAINED_W = None

# ----------------------------------------------------------------------------
# milestone 1: the count table
# ----------------------------------------------------------------------------


def m1_counts():
    small = ["ab", "ba", "a"]
    N = _as_tensor(bigram_counts(small, STOI), "bigram_counts")
    check(
        tuple(N.shape) == (V, V),
        f"bigram_counts returned shape {tuple(N.shape)}, expected ({V}, {V}). "
        "There is one row and one column per character, boundary token included.",
    )
    check(
        not N.is_floating_point(),
        f"bigram_counts returned a {N.dtype} tensor. Counts are integers.",
    )
    a, b, dot = STOI["a"], STOI["b"], STOI["."]
    check(
        N[a, b] == 1 and N[b, a] == 1,
        f"for {small}: N[a,b]={int(N[a,b])}, N[b,a]={int(N[b,a])}, expected 1 and 1. "
        "Check which index is 'previous' and which is 'next'.",
    )
    check(
        N[dot, a] == 2,
        f"for {small}: N[.,a]={int(N[dot,a])}, expected 2. Two of the three words "
        "start with 'a'. Is the start of a word being counted as a bigram?",
    )
    check(
        N[a, dot] == 2 and N[b, dot] == 1,
        f"for {small}: N[a,.]={int(N[a,dot])}, N[b,.]={int(N[b,dot])}, expected 2 and 1. "
        "Is the end of a word being counted as a bigram?",
    )
    check(
        int(N.sum()) == 8,
        f"for {small}: total count is {int(N.sum())}, expected 8 "
        "(3 + 3 + 2: each word contributes len(word)+1 bigrams).",
    )

    N = _as_tensor(bigram_counts(WORDS, STOI), "bigram_counts")
    check(
        int(N.sum()) == int(N_REF.sum()),
        f"on the full dataset the total count is {int(N.sum())}, expected "
        f"{int(N_REF.sum())} (= sum of len(w)+1 over all words).",
    )
    check(
        torch.equal(N.to(torch.int64), N_REF),
        "counts on the full dataset don't match. Shape and total are right, so "
        "some cell is off -- try printing the row for '.' and compare it with "
        "how many names start with each letter.",
    )
    check(
        int(N[dot, dot]) == 0,
        "N[., .] is nonzero: a word boundary is following a word boundary. "
        "Names are separate; the end of one should not link to the start of the next.",
    )


# ----------------------------------------------------------------------------
# milestone 2: rows -> probabilities, and sampling
# ----------------------------------------------------------------------------

SEED = 2147483647
NAMES_REF = _sample(P_REF, 10, SEED)


def m2_probs_and_sampling():
    P = _as_tensor(counts_to_probs(N_REF), "counts_to_probs")
    check(
        tuple(P.shape) == (V, V),
        f"counts_to_probs returned shape {tuple(P.shape)}, expected ({V}, {V}).",
    )
    check(
        P.is_floating_point(),
        f"counts_to_probs returned {P.dtype}; probabilities are floats.",
    )
    rows = P.sum(1)
    cols = P.sum(0)
    if not close(rows, torch.ones(V), 1e-4):
        if close(cols, torch.ones(V), 1e-4):
            raise Fail(
                "every COLUMN of P sums to 1, but the rows don't. Each row should "
                "be 'given this previous character, the distribution over the next "
                "one'. Look closely at what shape the divisor has and how it "
                "broadcasts against N."
            )
        if close(P.sum(), 1.0, 1e-4):
            raise Fail(
                "the whole table sums to 1, but each row should. P[i] must be a "
                "probability distribution over the next character on its own."
            )
        raise Fail(
            f"rows of P don't sum to 1 (row sums range {rows.min():.4f}..{rows.max():.4f})."
        )
    check(
        close(P.double(), P_REF, 1e-5),
        "rows sum to 1 but the entries differ from N / row-total. Something else is "
        "being done to the counts before dividing.",
    )

    # tiny table where the wrong axis can't hide
    small = torch.tensor([[3, 1], [0, 4]])
    Ps = counts_to_probs(small)
    check(
        close(Ps, torch.tensor([[0.75, 0.25], [0.0, 1.0]]), 1e-5),
        f"counts_to_probs([[3,1],[0,4]]) gave\n{Ps}\nexpected [[0.75, 0.25], [0, 1]].",
    )

    names = _sample(P, 10, SEED)
    check(
        names == NAMES_REF,
        f"seeded samples from your P are {names[:3]}..., the reference gives "
        f"{NAMES_REF[:3]}.... P matches numerically, so this is a dtype/shape "
        "difference feeding torch.multinomial.",
    )


# ----------------------------------------------------------------------------
# milestone 3: the negative log likelihood
# ----------------------------------------------------------------------------


def m3_nll():
    # random per-example rows and random targets, vs. the definition
    g = torch.Generator().manual_seed(0)
    n = 500
    probs = torch.rand((n, V), generator=g)
    probs = probs / probs.sum(1, keepdim=True)
    ys = torch.randint(0, V, (n,), generator=g)
    want = -probs[torch.arange(n), ys].log().mean()
    got = nll(probs, ys)
    check(
        torch.is_tensor(got) or isinstance(got, float),
        f"nll returned {type(got).__name__}",
    )
    got = torch.as_tensor(got).float()
    check(
        got.dim() == 0,
        f"nll returned a tensor of shape {tuple(got.shape)}. The loss is ONE number "
        "for the whole batch.",
    )
    if not close(got, want, 1e-4):
        if close(got, -want, 1e-4):
            raise Fail(
                f"nll is {float(got):.4f}, expected {float(want):.4f}. The sign is flipped: "
                "log of a probability is <= 0, and a loss should be something you can "
                "minimize by making the data MORE likely."
            )
        if close(got, want * n, 1e-4):
            raise Fail(
                f"nll is {float(got):.2f}, expected {float(want):.4f}. Off by a factor "
                f"of {n} = the number of examples. The loss should not grow just "
                "because the dataset does."
            )
        if close(got, want / math.log(10), 1e-3) or close(
            got, want / math.log(2), 1e-3
        ):
            raise Fail(
                f"nll is {float(got):.4f}, expected {float(want):.4f}. Right shape, "
                "wrong base of logarithm."
            )
        if close(got, -probs.log().mean(), 1e-4):
            raise Fail(
                f"nll is {float(got):.4f}, expected {float(want):.4f}. This is the "
                "average over EVERY entry of every row. Only one entry per row "
                "matters: the probability the model gave to what actually came next."
            )
        raise Fail(
            f"nll is {float(got):.4f}, expected {float(want):.4f} for random rows and "
            "random targets. For each row i, pick out probs[i, ys[i]], take its log, "
            "average over i, negate."
        )

    # a smaller batch is a different average
    want_sub = -probs[torch.arange(100), ys[:100]].log().mean()
    check(
        close(nll(probs[:100], ys[:100]), want_sub, 1e-4),
        "nll on the first 100 rows doesn't match. Is the average over the rows given?",
    )

    # on the real dataset: the counting model scores itself.
    # P[xs] is the row lookup: one (27,) distribution per bigram.
    P = counts_to_probs(N_REF)
    got = float(nll(P[XS], YS))
    check(
        close(got, COUNT_LOSS, 1e-3),
        f"nll of the counting model on the full dataset is {got:.4f}, expected "
        f"{COUNT_LOSS:.4f}.",
    )
    uniform = torch.full((len(XS), V), 1.0 / V)
    got_u = float(nll(uniform, YS))
    check(
        close(got_u, math.log(V), 1e-3),
        f"nll of the uniform model is {got_u:.4f}; expected log({V}) = {math.log(V):.4f}. "
        "Every target has probability 1/27 under it, so the average should be exactly "
        "-log(1/27).",
    )
    # a zero must hurt, not silently pass
    P0 = P.clone()
    P0[STOI["a"], STOI["n"]] = 0.0
    got0 = float(nll(P0[XS], YS))
    check(
        math.isinf(got0) or got0 > 100,
        f"nll with P[a,n] = 0 is {got0:.4f}. A pair the model calls impossible "
        "that actually occurs should make the loss blow up (inf). It didn't.",
    )


# ----------------------------------------------------------------------------
# milestone 4: one-hot -> linear -> softmax
# ----------------------------------------------------------------------------


def m4_forward():
    g = torch.Generator().manual_seed(1)
    W = torch.randn((V, V), generator=g, requires_grad=True)
    xs = torch.randint(0, V, (64,), generator=g)
    try:
        probs = forward(W, xs)
    except RuntimeError as e:
        msg = str(e)
        if "Long" in msg or "Int" in msg or "dtype" in msg or "mat1 and mat2" in msg:
            raise Fail(
                f"forward raised: {msg.splitlines()[0]}\n    torch.one_hot hands back "
                "integers; the matmul wants both sides to be the same float dtype as W."
            )
        raise
    probs = _as_tensor(probs, "forward")
    check(
        tuple(probs.shape) == (64, V),
        f"forward returned shape {tuple(probs.shape)}, expected (64, {V}): one row "
        "per input character, one column per possible next character.",
    )
    want = F.softmax(F.one_hot(xs, V).float() @ W, dim=1)
    check(
        (probs >= 0).all() and (probs <= 1).all(),
        f"forward returned values outside [0, 1] (min {float(probs.min()):.3f}, max "
        f"{float(probs.max()):.3f}). Logits are not probabilities yet -- what makes "
        "them positive, and what makes them sum to one?",
    )
    rows = probs.sum(1)
    if not close(rows, torch.ones(64), 1e-4):
        if close(probs.sum(0), torch.ones(V), 1e-4) or close(probs.sum(), 1.0, 1e-4):
            raise Fail(
                "forward's rows don't sum to 1 but something else does. Normalize "
                "so that EACH ROW is a distribution over the next character."
            )
        raise Fail(
            f"forward's rows don't sum to 1 (range {rows.min():.4f}..{rows.max():.4f})."
        )
    check(
        close(probs.detach(), want.detach(), 1e-4),
        "forward's rows are valid distributions but the numbers differ from "
        "softmax(one_hot(xs) @ W). Check the order: exponentiate the logits, "
        "THEN normalize each row.",
    )
    check(
        probs.requires_grad,
        "forward's output is not connected to W in the autograd graph "
        "(requires_grad is False). Did something call .detach(), .item(), .data, "
        "or torch.no_grad() along the way? The loss has to be able to reach W.",
    )
    # gradient actually flows to W, and it is the same one torch computes
    probs[torch.arange(64), xs].log().sum().backward()
    gW = W.grad.clone()
    W.grad = None
    want[torch.arange(64), xs].log().sum().backward()
    check(
        close(gW, W.grad, 1e-3),
        "forward's values match but its gradient w.r.t. W does not. Some op in "
        "the middle is breaking or bending the graph.",
    )

    # each row of the output depends only on its own input character
    W2 = torch.randn((V, V), generator=g, requires_grad=True)
    single = torch.stack([forward(W2, xs[i : i + 1])[0] for i in range(8)]).detach()
    batch = forward(W2, xs[:8]).detach()
    check(
        close(single, batch, 1e-4),
        "forward on one character at a time disagrees with forward on the batch. "
        "The rows are leaking into each other -- probably in the normalization.",
    )


# ----------------------------------------------------------------------------
# milestone 5: gradient descent, and it lands where counting did
# ----------------------------------------------------------------------------


def m5_train():
    global _TRAINED_W
    g = torch.Generator().manual_seed(SEED)
    W = torch.randn((V, V), generator=g, requires_grad=True)
    W0 = W.detach().clone()
    first = float(-forward(W, XS)[torch.arange(len(XS)), YS].log().mean())
    try:
        losses = train(W, XS, YS, steps=100, lr=50.0)
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        if "leaf Variable" in msg or "in-place" in msg or "requires grad" in msg:
            raise Fail(
                f"train raised: {msg}\n    The update is being recorded as part of the "
                "graph. The nudge to W is bookkeeping, not math the loss should see."
            )
        if "backward through the graph a second time" in msg:
            raise Fail(
                f"train raised: {msg}\n    Each step needs a fresh forward pass before "
                "its backward pass."
            )
        raise
    check(
        isinstance(losses, (list, tuple)) and len(losses) == 100,
        f"train should return the list of {100} per-step losses (got "
        f"{type(losses).__name__} of length {len(losses) if hasattr(losses, '__len__') else '?'}).",
    )
    losses = [float(l) for l in losses]
    check(
        not torch.equal(W.detach(), W0),
        "W did not change at all during training. Is the update touching the "
        "tensor the grader passed in, or a copy?",
    )
    nans = [i for i, l in enumerate(losses) if math.isnan(l) or math.isinf(l)]
    if nans:
        k = nans[0]
        prefix = losses[:k]
        trend = (
            f"{prefix[0]:.3f} -> {prefix[-1]:.3f}"
            if len(prefix) > 1
            else f"{prefix[0]:.3f}" if prefix else "?"
        )
        raise Fail(
            f"loss blew up to nan/inf at step {k} (before that it went {trend}). "
            "A loss that climbs and then explodes means each step is making W worse. "
            "Either the update moves W the wrong way along the gradient, or gradients "
            "from earlier steps are piling up in W.grad and the effective step keeps growing."
        )
    check(
        close(losses[0], first, 1e-3),
        f"the first recorded loss is {losses[0]:.4f} but the loss of the untouched W "
        f"is {first:.4f}. Record the loss before the step that changes W, and make "
        "sure it is nll(forward(W, xs), ys) from milestones 3 and 4.",
    )
    if losses[-1] > losses[0]:
        raise Fail(
            f"loss went UP: {losses[0]:.4f} -> {losses[-1]:.4f}. Either the update moves "
            "W the wrong way along the gradient, or gradients from earlier steps are "
            "piling up in W.grad."
        )
    if losses[-1] > 2.6:
        # rough diagnosis of the classic bug: accumulated grads make it stall/wobble
        wobble = sum(1 for a, b in zip(losses, losses[1:]) if b > a)
        raise Fail(
            f"loss went {losses[0]:.4f} -> {losses[-1]:.4f} in 100 steps at lr=50; it "
            f"should be below 2.6 by then ({wobble} of the steps made it worse). "
            "If it's crawling: is lr actually applied? If it's wobbling: is W.grad "
            "being reset between steps?"
        )
    now = float(-forward(W, XS)[torch.arange(len(XS)), YS].log().mean())
    check(
        now <= losses[-1] + 1e-3,
        f"the last recorded loss was {losses[-1]:.4f} but the loss of the final W is "
        f"{now:.4f}. The last update made things worse, or the recorded losses are "
        "off by one step.",
    )
    # the punchline: gradient descent finds (nearly) the counting model
    check(
        now < COUNT_LOSS + 0.05,
        f"after 100 steps the neural model sits at {now:.4f}; the counting model's "
        f"loss is {COUNT_LOSS:.4f}. It should be within 0.05. It's learning, but "
        "something is slowing it: check that the WHOLE dataset is in every step and "
        "that the gradient is being used at full strength.",
    )
    _TRAINED_W = W.detach().clone()


# ----------------------------------------------------------------------------
# milestone 6: the learned count table, and sampling from it
# ----------------------------------------------------------------------------


def m6_neural_table():
    g = torch.Generator().manual_seed(3)
    W = torch.randn((V, V), generator=g)
    T = _as_tensor(neural_probs(W), "neural_probs")
    check(
        tuple(T.shape) == (V, V),
        f"neural_probs returned shape {tuple(T.shape)}, expected ({V}, {V}): row i is "
        "the model's distribution over the next character when the previous one is i.",
    )
    check(
        close(T.detach(), F.softmax(W, dim=1), 1e-4),
        "neural_probs(W) does not match forward on every character in order. Row i "
        "of the table should equal forward(W, tensor([i]))[0].",
    )
    # log-probabilities in, the counting model back out: same fingerprint.
    # (clamp keeps log(0) finite so 0 * -inf can't poison the matmul)
    W_log = P_REF.float().clamp(min=1e-30).log()
    T = neural_probs(W_log)
    names = _sample(T, 10, SEED)
    check(
        names == NAMES_REF,
        f"with W = log(P_counts), sampling should reproduce the counting model's "
        f"names exactly, but got {names[:3]}... vs {NAMES_REF[:3]}....",
    )
    # and the trained model's samples look like names, not noise
    check(_TRAINED_W is not None, "milestone 5 has not run in this session")
    T = neural_probs(_TRAINED_W)
    names = _sample(T, 20, SEED)
    check(
        all(all(c in CHARS for c in n) for n in names),
        f"a sample contains a character outside a-z: {names}",
    )
    avg = sum(len(n) for n in names) / len(names)
    check(
        3.0 <= avg <= 12.0,
        f"samples from the trained W average {avg:.1f} characters; the data averages "
        f"about {len(XS)/len(WORDS) - 1:.1f}. Something is off with the '.' row or column.",
    )


# ----------------------------------------------------------------------------
# milestone 7 (stretch): smoothing and regularization are the same idea
# ----------------------------------------------------------------------------


def m7_regularization():
    P0 = smooth_counts(N_REF, 0)
    check(
        close(P0.double(), P_REF, 1e-5),
        "smooth_counts(N, 0) should be exactly the counting model.",
    )
    P1 = smooth_counts(N_REF, 1)
    check(
        close(P1.sum(1), torch.ones(V), 1e-4), "smooth_counts rows must still sum to 1."
    )
    check(
        (P1 > 0).all(),
        "smooth_counts(N, 1) still contains zeros. The fake counts go into EVERY cell.",
    )
    check(
        float(nll(P1[XS], YS)) > COUNT_LOSS,
        "smoothing should cost a little likelihood on the training data (that's "
        "the trade: no more infinities, slightly worse fit).",
    )
    Pbig = smooth_counts(N_REF, 10**7)
    check(
        close(Pbig, torch.full((V, V), 1.0 / V), 1e-3),
        "with an enormous fake count every row should be nearly uniform.",
    )

    g = torch.Generator().manual_seed(5)
    W = torch.randn((V, V), generator=g, requires_grad=True)
    xs = torch.randint(0, V, (300,), generator=g)
    ys = torch.randint(0, V, (300,), generator=g)
    base = (
        -F.softmax(F.one_hot(xs, V).float() @ W, 1)[torch.arange(300), ys].log().mean()
    )
    check(
        close(nll_reg(W, xs, ys, 0.0).detach(), base.detach(), 1e-4),
        "nll_reg with alpha=0 should be plain nll(forward(W, xs), ys).",
    )
    got = nll_reg(W, xs, ys, 0.1).detach()
    want = base.detach() + 0.1 * (W.detach() ** 2).mean()
    if not close(got, want, 1e-4):
        if close(got, base.detach() + 0.1 * (W.detach() ** 2).sum(), 1e-4):
            raise Fail(
                "nll_reg uses the SUM of W**2. Use the mean so alpha means the same "
                "thing regardless of how many weights there are."
            )
        raise Fail(
            f"nll_reg(alpha=0.1) is {float(got):.4f}, expected {float(want):.4f} = "
            "nll + 0.1 * mean(W**2)."
        )
    check(
        nll_reg(W, xs, ys, 0.1).requires_grad,
        "nll_reg's output is not attached to W in the graph.",
    )
    # a huge penalty flattens the table, exactly like a huge fake count
    Wb = torch.randn((V, V), generator=g, requires_grad=True)
    for _ in range(60):
        loss = nll_reg(Wb, xs, ys, 50.0)
        Wb.grad = None
        loss.backward()
        Wb.data -= 1.0 * Wb.grad
    T = neural_probs(Wb).detach()
    check(
        close(T, torch.full((V, V), 1.0 / V), 5e-2),
        "training with a huge alpha should drive every row toward uniform, the "
        "way a huge fake count does. It didn't.",
    )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "the bigram count table", m1_counts),
    (2, "rows to probabilities (broadcasting), and sampling", m2_probs_and_sampling),
    (3, "negative log likelihood: the number to minimize", m3_nll),
    (4, "one-hot -> linear layer -> softmax", m4_forward),
    (5, "gradient descent, landing where counting did", m5_train),
    (6, "the learned count table, and sampling from it", m6_neural_table),
    (7, "STRETCH: smoothing == regularization", m7_regularization),
]


def grade(
    bigram_counts_fn=None,
    counts_to_probs_fn=None,
    nll_fn=None,
    forward_fn=None,
    train_fn=None,
    neural_probs_fn=None,
    smooth_counts_fn=None,
    nll_reg_fn=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:   grade(bigram_counts, counts_to_probs, upto=2)
                       grade(bigram_counts, ..., smooth_counts, nll_reg, skip=(7,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global bigram_counts, counts_to_probs, nll, forward, train, neural_probs
    global smooth_counts, nll_reg
    bigram_counts, counts_to_probs, nll, forward = (
        bigram_counts_fn,
        counts_to_probs_fn,
        nll_fn,
        forward_fn,
    )
    train, neural_probs, smooth_counts, nll_reg = (
        train_fn,
        neural_probs_fn,
        smooth_counts_fn,
        nll_reg_fn,
    )
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
            print(f"    {e}")
            _progress()
            print()
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
                    "    function not passed to grade() yet -- hand it over once it exists.\n"
                )
                return 1
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
    """CLI route: grade makemore_bigram.py in this folder, if you keep one."""
    import makemore_bigram as m

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(
        m.bigram_counts,
        m.counts_to_probs,
        m.nll,
        m.forward,
        m.train,
        m.neural_probs,
        getattr(m, "smooth_counts", None),
        getattr(m, "nll_reg", None),
        upto=upto,
    )


if __name__ == "__main__":
    sys.exit(main())

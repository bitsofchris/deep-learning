"""
Grader for "Becoming a Backprop Ninja". No pytest.

From the notebook:   from test_makemore_backprop_ninja import grade, cmp, forward
                     grade(backward_1, upto=1)
From the shell:      python test_makemore_backprop_ninja.py [milestone]
                     (grades makemore_backprop_ninja.py in this folder)

Every milestone runs the SAME seeded forward pass, calls your backward
function on it, and compares each tensor you produce against the .grad that
autograd computed for the same intermediate. Milestones run in order and the
grader stops at the first failure.

This module also owns the boilerplate the notebook imports: the dataset,
the parameter init, and forward(). The notebook shows the forward pass code
so you can read it; an identical copy lives here so grading never depends
on what is in your notebook cells.
"""

import os
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Filled in by grade(...) so the notebook can hand over its own functions.
backward_1 = backward_2 = backward_3 = backward_4 = backward_5 = None
dlogits_fast = dhprebn_fast = param_grads = None

# ----------------------------------------------------------------------------
# boilerplate: dataset, parameters, forward pass (given)
# ----------------------------------------------------------------------------

BLOCK_SIZE = 3  # context length: how many characters predict the next one
N_EMBD = 10  # embedding dimension per character
N_HIDDEN = 64  # hidden layer width
VOCAB = 27  # 26 letters + '.'

_DATA = None


def load_dataset(path=None, seed=42):
    """Build the makemore training set: X (N, BLOCK_SIZE) int64, Y (N,) int64.

    Uses the first 80% of a seeded shuffle of names.txt (the lecture's train
    split). Cached after the first call.
    """
    global _DATA
    if _DATA is not None:
        return _DATA
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "names.txt")
    words = open(path).read().splitlines()
    import random

    random.Random(seed).shuffle(words)
    words = words[: int(0.8 * len(words))]
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    X, Y = [], []
    for w in words:
        ctx = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    _DATA = (torch.tensor(X), torch.tensor(Y))
    return _DATA


def init_params(seed=2147483647):
    """The lecture's init. Returns [C, W1, b1, bngain, bnbias, W2, b2], all
    requires_grad=True. Note the non-zero biases and the 1+0.1*randn gain:
    they are there on purpose so no gradient is trivially zero."""
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((VOCAB, N_EMBD), generator=g)
    W1 = (
        torch.randn((N_EMBD * BLOCK_SIZE, N_HIDDEN), generator=g)
        * (5 / 3)
        / ((N_EMBD * BLOCK_SIZE) ** 0.5)
    )
    b1 = torch.randn(N_HIDDEN, generator=g) * 0.1
    W2 = torch.randn((N_HIDDEN, VOCAB), generator=g) * 0.1
    b2 = torch.randn(VOCAB, generator=g) * 0.1
    bngain = torch.randn((1, N_HIDDEN), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, N_HIDDEN), generator=g) * 0.1
    params = [C, W1, b1, bngain, bnbias, W2, b2]
    for p in params:
        p.requires_grad = True
    return params


PARAM_NAMES = ["C", "W1", "b1", "bngain", "bnbias", "W2", "b2"]
INTERMEDIATES = [
    "emb",
    "embcat",
    "hprebn",
    "bnmeani",
    "bndiff",
    "bndiff2",
    "bnvar",
    "bnvar_inv",
    "bnraw",
    "hpreact",
    "h",
    "logits",
    "logit_maxes",
    "norm_logits",
    "counts",
    "counts_sum",
    "counts_sum_inv",
    "probs",
    "logprobs",
    "loss",
]


def forward(params, Xb, Yb, autograd=True):
    """The MLP forward pass, one named tensor per elementary step.

    Returns a SimpleNamespace t holding every intermediate listed in
    INTERMEDIATES, the parameters by name, Xb, Yb and n = batch size.
    With autograd=True every intermediate has retain_grad() and
    t.loss.backward() has already been called, so t.<name>.grad is the
    autograd answer for you to compare against. With autograd=False the
    pass runs under torch.no_grad() (for training loops).
    """
    C, W1, b1, bngain, bnbias, W2, b2 = params
    n = Xb.shape[0]
    ctx = torch.enable_grad() if autograd else torch.no_grad()
    with ctx:
        emb = C[Xb]  # (n, 3, 10)
        embcat = emb.view(emb.shape[0], -1)  # (n, 30)
        # linear 1
        hprebn = embcat @ W1 + b1  # (n, 64)
        # batchnorm
        bnmeani = 1 / n * hprebn.sum(0, keepdim=True)  # (1, 64)
        bndiff = hprebn - bnmeani  # (n, 64)
        bndiff2 = bndiff**2  # (n, 64)
        bnvar = 1 / (n - 1) * bndiff2.sum(0, keepdim=True)  # (1, 64)  Bessel
        bnvar_inv = (bnvar + 1e-5) ** -0.5  # (1, 64)
        bnraw = bndiff * bnvar_inv  # (n, 64)
        hpreact = bngain * bnraw + bnbias  # (n, 64)
        # nonlinearity
        h = torch.tanh(hpreact)  # (n, 64)
        # linear 2
        logits = h @ W2 + b2  # (n, 27)
        # cross entropy, spelled out
        logit_maxes = logits.max(1, keepdim=True).values  # (n, 1)
        norm_logits = logits - logit_maxes  # (n, 27)
        counts = norm_logits.exp()  # (n, 27)
        counts_sum = counts.sum(1, keepdim=True)  # (n, 1)
        counts_sum_inv = counts_sum**-1  # (n, 1)
        probs = counts * counts_sum_inv  # (n, 27)
        logprobs = probs.log()  # (n, 27)
        loss = -logprobs[range(n), Yb].mean()  # ()
    t = SimpleNamespace(**{k: v for k, v in locals().items() if k in INTERMEDIATES})
    for name, p in zip(PARAM_NAMES, params):
        setattr(t, name, p)
    t.Xb, t.Yb, t.n = Xb, Yb, n
    if autograd:
        for p in params:
            p.grad = None
        for name in INTERMEDIATES:
            getattr(t, name).retain_grad()
        loss.backward()
    return t


def make_batch(seed=2147483647, batch_size=32):
    """A seeded batch from the training split. Returns (Xb, Yb)."""
    X, Y = load_dataset()
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
    return X[ix], Y[ix]


def fresh_forward(seed=2147483647, batch_size=32):
    """init_params + make_batch + forward, all from one seed."""
    params = init_params(seed)
    Xb, Yb = make_batch(seed, batch_size)
    return forward(params, Xb, Yb)


def cmp(name, dt, t):
    """The lecture's helper: does your dt match autograd's t.grad?
    Prints exact match, approximate match, and the max abs difference."""
    if dt is None or t.grad is None:
        print(f"{name:15s} | missing ({'dt' if dt is None else 't.grad'} is None)")
        return
    if dt.shape != t.grad.shape:
        print(f"{name:15s} | shape {tuple(dt.shape)} vs autograd {tuple(t.grad.shape)}")
        return
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad, rtol=1e-5, atol=1e-7)
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{name:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff:.3e}"
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
    """'backward_4' (or 'Cls.method') for the innermost frame that raised exc."""
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


RTOL, ATOL = 1e-5, 1e-7

# intermediates that are consumed by two different ops in the forward pass
MULTI_USE = {
    "dcounts": "counts feeds both counts_sum and probs",
    "dlogits": "logits feeds both logit_maxes and norm_logits",
    "dbndiff": "bndiff feeds both bndiff2 and bnraw",
    "dhprebn": "hprebn feeds both bnmeani and bndiff",
}


def _diagnose(name, dt, want, multi_use=True):
    """Explain a mismatch by symptom. Never states the formula."""
    if not torch.is_tensor(dt):
        return f"{name} is {type(dt).__name__}, expected a torch.Tensor"
    if dt.shape != want.shape:
        msg = f"{name}: shape {tuple(dt.shape)}, autograd has {tuple(want.shape)}."
        if dt.dim() == 2 and dt.shape == want.shape[::-1]:
            msg += (
                " That is the transpose of the right shape: which operand of the "
                "matmul goes on which side, and which one gets transposed?"
            )
        elif dt.numel() == want.numel():
            msg += (
                " Same number of elements, different dims: probably a keepdim "
                "question, or a view/reshape you didn't undo."
            )
        elif dt.numel() > want.numel():
            msg += (
                " Yours is bigger: this tensor was BROADCAST in the forward, "
                "and the broadcast dimension has to be collapsed on the way back."
            )
        else:
            msg += (
                " Yours is smaller than autograd's: gradient should have the "
                "exact shape of the tensor it belongs to."
            )
        return msg
    if torch.isnan(dt).any():
        return f"{name}: has NaNs. Something divided by zero or took a log of zero."
    maxdiff = (dt - want).abs().max().item()
    base = f"{name}: shape ok, values off. max abs diff {maxdiff:.3e}"
    if torch.allclose(-dt, want, rtol=RTOL, atol=ATOL):
        return base + " -- it is EXACTLY the negative of autograd. A sign is flipped."
    if torch.allclose(dt, torch.zeros_like(dt)):
        return base + " -- yours is all zeros. Nothing was written into it."
    mask = want.abs() > 1e-9
    if mask.any():
        ratio = dt[mask] / want[mask]
        if ratio.numel() > 1 and torch.allclose(
            ratio, ratio[0].expand_as(ratio), rtol=1e-3, atol=1e-6
        ):
            return (
                base
                + f" -- yours is autograd's times a constant {ratio[0].item():.4g} "
                "everywhere. A scalar factor is missing or extra."
            )
    if multi_use and name in MULTI_USE:
        return (
            base + f" -- note that {MULTI_USE[name]}, so two paths deliver "
            "gradient to it. Are you counting both?"
        )
    if not multi_use:
        return (
            base + " -- the fused expression has several terms; check the "
            "constant in front of each one (n vs n-1 matters here)."
        )
    return (
        base + " -- check the local derivative of this one op, and that you "
        "multiplied it by the gradient arriving from above (not by the forward value)."
    )


def _compare(g, name, target, multi_use=True):
    check(
        name in g, f"your returned dict has no key '{name}'. Keys so far: {sorted(g)}"
    )
    dt = g[name]
    check(dt is not None, f"{name} is None -- still a stub?")
    want = target.grad
    ok = (
        torch.is_tensor(dt)
        and dt.shape == want.shape
        and torch.allclose(dt, want, rtol=RTOL, atol=ATOL)
    )
    check(
        ok, f"{name} matches autograd" if ok else _diagnose(name, dt, want, multi_use)
    )


def _run(fn, *args):
    out = fn(*args)
    if isinstance(out, dict):
        return out
    # tolerate in-place mutation of g with no return
    if len(args) > 1 and isinstance(args[1], dict) and out is None:
        return args[1]
    raise Fail(
        f"{fn.__name__} should return a dict of named gradients, got {type(out).__name__}"
    )


def _grad_names(t, g, names):
    for name in names:
        _compare(g, name, getattr(t, name[1:]))


_T = None
_G = {}


def _forward_cached():
    global _T
    if _T is None:
        _T = fresh_forward()
    return _T


# ----------------------------------------------------------------------------
# milestones
# ----------------------------------------------------------------------------


def m1_loss_to_counts():
    global _G
    t = _forward_cached()
    g = _run(backward_1, t)
    _grad_names(
        t, g, ["dlogprobs", "dprobs", "dcounts_sum_inv", "dcounts_sum", "dcounts"]
    )
    _G = g


def m2_softmax_to_logits():
    global _G
    t = _forward_cached()
    g = _run(backward_2, t, dict(_G))
    _grad_names(t, g, ["dnorm_logits", "dlogit_maxes", "dlogits"])
    _G = g


def m3_layer2_and_tanh():
    global _G
    t = _forward_cached()
    g = _run(backward_3, t, dict(_G))
    _grad_names(t, g, ["dh", "dW2", "db2", "dhpreact"])
    _G = g


def m4_batchnorm_chain():
    global _G
    t = _forward_cached()
    g = _run(backward_4, t, dict(_G))
    _grad_names(
        t,
        g,
        [
            "dbngain",
            "dbnraw",
            "dbnbias",
            "dbnvar_inv",
            "dbnvar",
            "dbndiff2",
            "dbndiff",
            "dbnmeani",
            "dhprebn",
        ],
    )
    _G = g


def m5_layer1_and_embedding():
    global _G
    t = _forward_cached()
    g = _run(backward_5, t, dict(_G))
    _grad_names(t, g, ["dembcat", "dW1", "db1", "demb", "dC"])
    _G = g


def m6_fused_shortcuts():
    t = _forward_cached()
    dl = dlogits_fast(t)
    check(dl is not None, "dlogits_fast returned None -- still a stub?")
    _compare({"dlogits": dl}, "dlogits", t.logits)
    # single-expression batchnorm backward, fed autograd's own dhpreact so this
    # check does not depend on your earlier milestones
    dhb = dhprebn_fast(t, t.hpreact.grad)
    check(dhb is not None, "dhprebn_fast returned None -- still a stub?")
    _compare({"dhprebn": dhb}, "dhprebn", t.hprebn, multi_use=False)
    # and on a different batch size, so nothing is accidentally hardcoded to 32
    t2 = fresh_forward(seed=7, batch_size=48)
    _compare({"dlogits": dlogits_fast(t2)}, "dlogits", t2.logits)
    _compare(
        {"dhprebn": dhprebn_fast(t2, t2.hpreact.grad)},
        "dhprebn",
        t2.hprebn,
        multi_use=False,
    )


def _eval_loss(params, X, Y):
    return forward(params, X, Y, autograd=False).loss.item()


def m7_it_learns():
    # first: your parameter gradients on one batch must match autograd
    t = _forward_cached()
    grads = param_grads(t)
    check(grads is not None, "param_grads returned None -- still a stub?")
    check(
        isinstance(grads, (list, tuple)) and len(grads) == 7,
        f"param_grads should return a list of 7 tensors in the order {PARAM_NAMES}, "
        f"got {type(grads).__name__} of length {len(grads) if hasattr(grads, '__len__') else '?'}",
    )
    for name, p, dp in zip(
        PARAM_NAMES, [t.C, t.W1, t.b1, t.bngain, t.bnbias, t.W2, t.b2], grads
    ):
        _compare({"d" + name: dp}, "d" + name, p)

    # then: a short training run driven ONLY by your gradients, autograd never
    # consulted. loss on a fixed slice of the training set must drop.
    X, Y = load_dataset()
    params = init_params(seed=1)
    Xe, Ye = X[:4000], Y[:4000]
    first = _eval_loss(params, Xe, Ye)
    gen = torch.Generator().manual_seed(1)
    steps, lr = 300, 0.1
    for step in range(steps):
        ix = torch.randint(0, X.shape[0], (32,), generator=gen)
        tt = forward(params, X[ix], Y[ix], autograd=False)
        grads = param_grads(tt)
        with torch.no_grad():
            for p, dp in zip(params, grads):
                p -= lr * dp
    last = _eval_loss(params, Xe, Ye)
    check(
        last < 2.75,
        f"loss went {first:.3f} -> {last:.3f} over {steps} manual-gradient steps "
        "(lr 0.1, batch 32). Expected it well under 2.75. Since every gradient "
        "matched autograd on one batch, look at how the update is applied: sign, "
        "or a gradient that was computed from a stale tensor.",
    )
    print(f"       loss {first:.3f} -> {last:.3f} over {steps} steps, no autograd")


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "loss -> logprobs -> probs -> counts (the first five)", m1_loss_to_counts),
    (2, "the softmax: norm_logits, logit_maxes, logits", m2_softmax_to_logits),
    (3, "second linear layer and tanh", m3_layer2_and_tanh),
    (4, "the batchnorm chain, nine tensors", m4_batchnorm_chain),
    (5, "first linear layer and the embedding lookup", m5_layer1_and_embedding),
    (
        6,
        "STRETCH: the fused shortcuts (dlogits, dhprebn in one expression each)",
        m6_fused_shortcuts,
    ),
    (7, "STRETCH: train the MLP with only your gradients", m7_it_learns),
]


def grade(
    backward_1_fn=None,
    backward_2_fn=None,
    backward_3_fn=None,
    backward_4_fn=None,
    backward_5_fn=None,
    dlogits_fast_fn=None,
    dhprebn_fast_fn=None,
    param_grads_fn=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:   grade(backward_1, upto=1)
                       grade(backward_1, backward_2, backward_3, upto=3)
                       grade(backward_1, ..., param_grads)
                       grade(backward_1, ..., param_grads, skip=(6,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global backward_1, backward_2, backward_3, backward_4, backward_5
    global dlogits_fast, dhprebn_fast, param_grads, _G
    backward_1, backward_2, backward_3 = backward_1_fn, backward_2_fn, backward_3_fn
    backward_4, backward_5 = backward_4_fn, backward_5_fn
    dlogits_fast, dhprebn_fast, param_grads = (
        dlogits_fast_fn,
        dhprebn_fast_fn,
        param_grads_fn,
    )
    _G = {}
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
    """CLI route: grade makemore_backprop_ninja.py in this folder."""
    import makemore_backprop_ninja as m

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(
        m.backward_1,
        m.backward_2,
        m.backward_3,
        m.backward_4,
        m.backward_5,
        m.dlogits_fast,
        m.dhprebn_fast,
        m.param_grads,
        upto=upto,
    )


if __name__ == "__main__":
    sys.exit(main())

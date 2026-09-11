"""
Grader for makemore part 2 (the MLP). No pytest.

From the notebook:   from test_makemore_mlp import grade
                     grade(build_dataset, upto=1)
                     grade(build_dataset, embed, flatten_context, forward,
                           cross_entropy, train, sample)

Milestones run in order and the grader stops at the first failure.
Every check compares your function against torch's own implementation
(or a straightforward reference) on the same random inputs.
"""

import os
import random
import sys
import traceback

import torch
import torch.nn.functional as F

# Filled in by grade(...) so the notebook can hand over its own functions.
build_dataset = embed = flatten_context = forward = cross_entropy = train = sample = (
    None
)

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = os.path.join(HERE, "..", "data", "names.txt")

# vocab: '.' is 0, 'a' is 1 ... 'z' is 26. same as the notebook.
CHARS = ["."] + [chr(ord("a") + i) for i in range(26)]
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
V = len(CHARS)

# The parameter dict every milestone from 4 on speaks. Keys and shapes
# are the contract; the notebook's init_params builds the same thing.
#   C  (V, d)        W1 (block_size*d, H)   b1 (H,)
#   W2 (H, V)        b2 (V,)


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
    """'forward' / 'Foo.__call__' for the innermost frame that raised exc."""
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
    """Elementwise-close for tensors or floats."""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.shape != b.shape:
        return False
    return bool(torch.allclose(a, b, rtol=tol, atol=tol))


def words():
    with open(NAMES) as f:
        return f.read().splitlines()


def ref_build_dataset(ws, block_size):
    X, Y = [], []
    for w in ws:
        ctx = [0] * block_size
        for ch in w + ".":
            ix = STOI[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def ref_flatten(emb):
    # the slow-but-obviously-right way: pull the block_size slabs apart and
    # glue them side by side.
    return torch.cat(torch.unbind(emb, dim=1), dim=1)


def ref_forward(X, p):
    emb = p["C"][X]
    h = torch.tanh(ref_flatten(emb) @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"]


def init_params(block_size, d, H, g):
    """Same construction as the notebook's init_params."""
    p = {
        "C": torch.randn((V, d), generator=g),
        "W1": torch.randn((block_size * d, H), generator=g),
        "b1": torch.randn(H, generator=g),
        "W2": torch.randn((H, V), generator=g),
        "b2": torch.randn(V, generator=g),
    }
    for t in p.values():
        t.requires_grad = True
    return p


def as_tensor_pair(out, name):
    check(
        isinstance(out, tuple) and len(out) == 2,
        f"{name} should return a pair (X, Y); got {type(out).__name__}",
    )
    X, Y = out
    check(
        isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor),
        f"{name}: X and Y must both be torch tensors, got "
        f"{type(X).__name__}, {type(Y).__name__}",
    )
    return X, Y


# ----------------------------------------------------------------------------
# milestone 1: the (X, Y) dataset with a context window
# ----------------------------------------------------------------------------


def m1_build_dataset():
    X, Y = as_tensor_pair(build_dataset(["emma"], 3, STOI), "build_dataset")
    check(
        X.dtype == torch.int64 and Y.dtype == torch.int64,
        f"X is {X.dtype}, Y is {Y.dtype}. Both must be int64 (torch.long): "
        "they are indices, and both C[X] and cross-entropy will refuse floats.",
    )
    check(
        X.ndim == 2 and Y.ndim == 1,
        f"shapes X {tuple(X.shape)}, Y {tuple(Y.shape)}: X must be 2-D "
        "(N, block_size) and Y 1-D (N,).",
    )
    check(
        X.shape[0] == 5,
        f"'emma' with block_size 3 should give 5 examples, got {X.shape[0]}. "
        "Count the predictions the model has to make for one name, including "
        "the one that says the name is over.",
    )
    check(X.shape[1] == 3, f"X has {X.shape[1]} columns, expected block_size=3")
    check(
        Y.tolist() == [5, 13, 13, 1, 0],
        f"targets for 'emma' are {Y.tolist()}, expected [5, 13, 13, 1, 0] "
        "(e, m, m, a, .).",
    )
    check(
        X[0].tolist() == [0, 0, 0],
        f"first context is {X[0].tolist()}; before any character is seen the "
        "window should be all padding.",
    )
    check(
        X[1].tolist() == [0, 0, 5],
        f"second context is {X[1].tolist()}, expected [0, 0, 5]. The newest "
        "character goes on the right, the oldest falls off the left.",
    )
    want_X = [[0, 0, 0], [0, 0, 5], [0, 5, 13], [5, 13, 13], [13, 13, 1]]
    check(
        X.tolist() == want_X,
        f"contexts for 'emma' are {X.tolist()}, expected {want_X}. "
        "Each row is the window right BEFORE its target, and the window "
        "slides by one each step.",
    )

    # several words, several block sizes, against the reference
    ws = ["emma", "olivia", "ava", "isabella", "sophia", "a"]
    for bs in (1, 2, 3, 5, 8):
        X, Y = as_tensor_pair(build_dataset(ws, bs, STOI), "build_dataset")
        rX, rY = ref_build_dataset(ws, bs)
        check(
            X.shape == rX.shape and Y.shape == rY.shape,
            f"block_size={bs} on {len(ws)} words: got X {tuple(X.shape)}, "
            f"Y {tuple(Y.shape)}, expected X {tuple(rX.shape)}, "
            f"Y {tuple(rY.shape)}. The context must reset to padding at the "
            "start of every word, and every word contributes len(w)+1 rows.",
        )
        if not (torch.equal(X, rX) and torch.equal(Y, rY)):
            bad = int((X != rX).any(1).logical_or(Y != rY).nonzero()[0])
            raise Fail(
                f"block_size={bs}: rows differ from the reference. First bad "
                f"row is index {bad}: got X {X[bad].tolist()} -> {int(Y[bad])}, "
                f"expected {rX[bad].tolist()} -> {int(rY[bad])}. Is the "
                "context leaking from one word into the next?"
            )


# ----------------------------------------------------------------------------
# milestone 2: the embedding lookup
# ----------------------------------------------------------------------------


def m2_embed():
    g = torch.Generator().manual_seed(1)
    C = torch.randn((V, 2), generator=g)
    X, _ = ref_build_dataset(["emma", "olivia"], 3)
    emb = embed(C, X)
    check(isinstance(emb, torch.Tensor), "embed should return a tensor")
    check(
        tuple(emb.shape) == (X.shape[0], 3, 2),
        f"embed(C, X) has shape {tuple(emb.shape)}, expected "
        f"{(X.shape[0], 3, 2)} = (N, block_size, d). Every integer in X "
        "becomes one row of C, and the shape of X is kept in front.",
    )
    want = F.one_hot(X, V).float() @ C
    check(
        close(emb, want),
        "embed(C, X) values are wrong. Each entry X[i, j] should pick out "
        "row C[X[i, j]] -- the same thing one_hot(X) @ C computes, only cheaper.",
    )

    # bigger d, other block size, and a single-row X
    C = torch.randn((V, 10), generator=g)
    X, _ = ref_build_dataset(["ava"], 5)
    emb = embed(C, X)
    check(
        tuple(emb.shape) == (4, 5, 10), f"shape {tuple(emb.shape)}, expected (4, 5, 10)"
    )
    check(
        close(emb, F.one_hot(X, V).float() @ C), "values wrong for d=10, block_size=5"
    )
    one = embed(C, X[:1])
    check(tuple(one.shape) == (1, 5, 10), f"single-row X gave {tuple(one.shape)}")

    # must be differentiable w.r.t. C: the embedding is a parameter.
    C.requires_grad = True
    embed(C, X).sum().backward()
    check(
        C.grad is not None and C.grad.abs().sum() > 0,
        "gradient does not flow back into C. The lookup has to stay inside "
        "the autograd graph or C can never learn.",
    )


# ----------------------------------------------------------------------------
# milestone 3: flattening the context window (the .view)
# ----------------------------------------------------------------------------


def m3_flatten():
    g = torch.Generator().manual_seed(2)
    for N, T, d in [(7, 3, 2), (5, 4, 10), (32, 8, 3), (1, 3, 2)]:
        emb = torch.randn((N, T, d), generator=g)
        out = flatten_context(emb)
        check(isinstance(out, torch.Tensor), "flatten_context should return a tensor")
        check(
            tuple(out.shape) == (N, T * d),
            f"flatten_context on (N={N}, T={T}, d={d}) gave "
            f"{tuple(out.shape)}, expected {(N, T * d)}.",
        )
        want = ref_flatten(emb)
        check(
            close(out, want),
            f"flatten_context on (N={N}, T={T}, d={d}): the shape is right but "
            "the numbers are in the wrong places. Row i must be the T "
            "embeddings of example i laid side by side, in order: "
            "[emb[i,0], emb[i,1], ..., emb[i,T-1]]. Print a tiny example "
            "(N=2, T=2, d=2) and check which element landed where.",
        )
        check(
            out.data_ptr() == emb.data_ptr(),
            "Your answer is right, but it is not the one asked for. The shape "
            "and every number are correct, but the result lives in new memory "
            "(torch.cat / stack / clone all copy). The lecture's point is that "
            "a view is free and a copy is not: the same storage can be "
            "reinterpreted as (N, T*d) without moving a single number. Which "
            "op gives a different shape over the SAME storage?",
        )

    # gradient must flow
    emb = torch.randn((4, 3, 2), generator=g, requires_grad=True)
    flatten_context(emb).pow(2).sum().backward()
    check(
        close(emb.grad, 2 * emb.detach()), "gradient through flatten_context is wrong"
    )


# ----------------------------------------------------------------------------
# milestone 4: the forward pass to logits
# ----------------------------------------------------------------------------


def m4_forward():
    for seed, (bs, d, H) in enumerate([(3, 2, 100), (3, 10, 64), (5, 4, 30)]):
        g = torch.Generator().manual_seed(10 + seed)
        p = init_params(bs, d, H, g)
        X, _ = ref_build_dataset(["emma", "olivia", "ava", "isabella"], bs)
        logits = forward(X, p)
        check(isinstance(logits, torch.Tensor), "forward should return a tensor")
        check(
            tuple(logits.shape) == (X.shape[0], V),
            f"forward gave {tuple(logits.shape)}, expected {(X.shape[0], V)} = "
            "(N, vocab). One row of scores per example, one score per "
            "possible next character.",
        )
        want = ref_forward(X, p)
        check(
            not close(logits, torch.tanh(want)),
            "logits look squashed into (-1, 1). The nonlinearity belongs on "
            "the hidden layer; the output layer is raw scores.",
        )
        # missing tanh on the hidden layer
        emb = ref_flatten(p["C"][X])
        linear = (emb @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        check(
            not close(logits, linear),
            "logits match a network with NO nonlinearity between the layers. "
            "Two matrix multiplies in a row collapse into one; something has "
            "to bend the hidden activations.",
        )
        check(
            close(logits, want, 1e-4),
            f"logits (block_size={bs}, d={d}, H={H}) differ from the reference "
            f"by up to {(logits - want).abs().max().item():.3g}. The pieces are "
            "embed -> flatten -> W1,b1 -> tanh -> W2,b2. Check each bias is "
            "added and each matrix is on the correct side of the @.",
        )
        check(
            logits.requires_grad,
            "logits are detached from the parameters: no gradient can reach "
            "C, W1, W2. Did something call .detach(), .data, or .item() on the way?",
        )


# ----------------------------------------------------------------------------
# milestone 5: cross-entropy by hand, checked against F.cross_entropy
# ----------------------------------------------------------------------------


def m5_cross_entropy():
    g = torch.Generator().manual_seed(3)
    for N in (64, 5, 1, 27):
        logits = torch.randn((N, V), generator=g)
        Y = torch.randint(0, V, (N,), generator=g)
        loss = cross_entropy(logits, Y)
        check(isinstance(loss, torch.Tensor), "cross_entropy should return a tensor")
        check(
            loss.ndim == 0,
            f"loss has shape {tuple(loss.shape)}; it should be a single scalar "
            "(0-d tensor) summarising the whole batch.",
        )
        want = F.cross_entropy(logits, Y)
        # common wrong answers, named by symptom
        check(
            N == 1 or not close(loss, want * N, 1e-3),
            f"loss is {loss.item():.4f}; torch says {want.item():.4f}. Yours is "
            "the total over the batch, not the average -- it would grow with "
            "batch size.",
        )
        check(
            not close(loss, -want, 1e-3),
            f"loss is {loss.item():.4f}; torch says {want.item():.4f}. The sign "
            "is flipped: a loss is something to minimise.",
        )
        check(
            close(loss, want, 1e-4),
            f"loss is {loss.item():.4f} but F.cross_entropy gives "
            f"{want.item():.4f} on the same (N={N}, 27) logits. Check which "
            "axis you normalise over (every ROW must be a distribution), that "
            "you pick out each row's probability at its own target, and that "
            "you take -log of that, then average.",
        )

    # gradient must match torch's, i.e. the whole thing lives in the graph
    logits = torch.randn((16, V), generator=g, requires_grad=True)
    Y = torch.randint(0, V, (16,), generator=g)
    cross_entropy(logits, Y).backward()
    ref = logits.detach().clone().requires_grad_(True)
    F.cross_entropy(ref, Y).backward()
    check(
        logits.grad is not None,
        "loss.backward() gave logits no gradient. The loss has been cut off "
        "from the graph somewhere (.data, .item(), .detach(), or a numpy hop).",
    )
    check(
        close(logits.grad, ref.grad, 1e-4),
        "loss value is right but d(loss)/d(logits) differs from torch's. "
        "Something in the loss is done outside autograd.",
    )

    # numerical stability: torch's version shifts the logits first.
    big = torch.randn((8, V), generator=g) + 100.0
    Y = torch.randint(0, V, (8,), generator=g)
    loss = cross_entropy(big, Y)
    want = F.cross_entropy(big, Y)
    check(
        torch.isfinite(loss).item(),
        f"with logits offset by +100 your loss is {loss.item()}, torch's is "
        f"{want.item():.4f}. exp() of a big number overflows float32. What can "
        "you subtract from every row without changing its softmax?",
    )
    check(
        close(loss, want, 1e-3),
        f"with large logits your loss is {loss.item():.4f}, torch's is "
        f"{want.item():.4f}.",
    )


# ----------------------------------------------------------------------------
# milestone 6: minibatch training, and it learns
# ----------------------------------------------------------------------------

_trained = {}  # filled by m6, read by m7

BLOCK, D, H = 3, 10, 100
STEPS, BATCH, LR = 4000, 32, 0.1
DEV_THRESHOLD = 2.95  # reference lands ~2.55-2.65 after 4000 steps; init is ~15-20


def _splits():
    ws = words()
    random.Random(42).shuffle(ws)
    n1, n2 = int(0.8 * len(ws)), int(0.9 * len(ws))
    return ws[:n1], ws[n1:n2], ws[n2:]


def m6_train():
    tr, dev, _ = _splits()
    Xtr, Ytr = ref_build_dataset(tr, BLOCK)
    Xdev, Ydev = ref_build_dataset(dev, BLOCK)
    g = torch.Generator().manual_seed(2147483647)
    p = init_params(BLOCK, D, H, g)
    before = {k: v.detach().clone() for k, v in p.items()}

    with torch.no_grad():
        dev0 = F.cross_entropy(ref_forward(Xdev, p), Ydev).item()

    losses = train(p, Xtr, Ytr, steps=STEPS, batch_size=BATCH, lr=LR, g=g)
    check(
        isinstance(losses, (list, tuple)) and len(losses) == STEPS,
        f"train should return one loss per step: a list of {STEPS} floats, "
        f"got {type(losses).__name__} of length "
        f"{len(losses) if hasattr(losses, '__len__') else '?'}.",
    )
    check(
        all(isinstance(l, float) for l in losses),
        "losses should be plain Python floats (so they don't drag the graph "
        "around). What turns a 0-d tensor into a float?",
    )
    check(
        all(l == l for l in losses) and max(losses) < 1e6,
        f"loss blew up (max {max(losses):.3g}). Either the update pushes the "
        "wrong way / too far, or gradients from earlier steps are piling up "
        "on top of each other.",
    )
    changed = [k for k in p if not torch.equal(p[k].detach(), before[k])]
    check(
        len(changed) == len(p),
        f"after training these parameters never moved: "
        f"{sorted(set(p) - set(changed))}. Every tensor in the dict is "
        "supposed to be updated from its own .grad.",
    )
    check(
        all(v.requires_grad for v in p.values()),
        "a parameter lost requires_grad during training. Updating a leaf "
        "tensor in place has to happen outside the graph.",
    )
    with torch.no_grad():
        dev1 = F.cross_entropy(ref_forward(Xdev, p), Ydev).item()
        # and through the learner's own forward+loss, to be sure they agree
        dev1_own = cross_entropy(forward(Xdev, p), Ydev).item()
    check(
        close(dev1, dev1_own, 1e-3),
        f"dev loss via your forward+cross_entropy is {dev1_own:.4f} but via "
        f"the reference it is {dev1:.4f}; earlier milestones passed so this "
        "should not happen -- did a stub get redefined?",
    )
    check(
        dev1 < DEV_THRESHOLD,
        f"dev loss went {dev0:.3f} -> {dev1:.3f} over {STEPS} steps of batch "
        f"{BATCH} at lr {LR}; it should be under {DEV_THRESHOLD}. Print the "
        "first and last few step losses. Flat: the update isn't being applied "
        "or grads are None. Noisy but not falling: are grads being reset "
        "before each backward? Falling then rising: the step is too big.",
    )
    _trained.clear()
    _trained.update(p)
    print(f"       (dev loss {dev0:.3f} -> {dev1:.3f})")


# ----------------------------------------------------------------------------
# milestone 7 (stretch): sampling names from the model
# ----------------------------------------------------------------------------


def m7_sample():
    # a rigged model whose only rule is "next char = previous char + 1",
    # and '.' after 'z'. It reads ONLY the last slot of the context window,
    # so a sampler that doesn't slide the window correctly gets stuck.
    T = 3
    p = {
        "C": torch.eye(V),
        "W1": torch.zeros((T * V, V)),
        "b1": torch.zeros(V),
        "W2": torch.zeros((V, V)),
        "b2": torch.zeros(V),
    }
    p["W1"][(T - 1) * V :, :] = torch.eye(V)
    for i in range(V):
        p["W2"][i, (i + 1) % V] = 100.0
    g = torch.Generator().manual_seed(0)
    out = sample(p, ITOS, T, g, max_len=30)
    check(isinstance(out, str), f"sample should return a str, got {type(out).__name__}")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    check(
        "." not in out,
        f"sample returned {out!r}: the '.' is the stop signal, not part of the name.",
    )
    check(
        out != "a" * 30 and not out.startswith("aaa"),
        f"sample returned {out!r} on a model that always predicts 'previous "
        "char + 1'. It keeps seeing the same context: after each character "
        "the window has to shift so the new character is in the last slot.",
    )
    check(
        out == alphabet,
        f"sample returned {out!r}; on this rigged model the only possible "
        f"name is {alphabet!r}. Start from an all-padding context, feed the "
        "current window through forward, turn logits into probabilities, "
        "draw one index, append, slide.",
    )

    # on the model trained in milestone 6: draws must actually be random
    check(bool(_trained), "internal: milestone 6 did not leave trained params")
    g = torch.Generator().manual_seed(2147483647 + 10)
    names = [sample(_trained, ITOS, BLOCK, g, max_len=30) for _ in range(20)]
    check(
        all(isinstance(n, str) and all(c in alphabet for c in n) for n in names),
        f"samples contain characters outside a-z: {names}",
    )
    check(
        len(set(names)) >= 10,
        f"20 samples gave only {len(set(names))} distinct names: {names[:5]}... "
        "Each next character must be DRAWN from the distribution, not the "
        "single most likely one.",
    )
    check(
        2 <= sum(map(len, names)) / len(names) <= 12,
        f"average sampled length is {sum(map(len, names)) / len(names):.1f}; "
        "names from this model should average roughly 4-8 characters.",
    )
    print("       samples:", ", ".join(names[:8]))


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "build_dataset: sliding context window with '.' padding", m1_build_dataset),
    (2, "embed: the lookup table C[X]", m2_embed),
    (3, "flatten_context: (N, T, d) -> (N, T*d) without scrambling", m3_flatten),
    (4, "forward: embed -> hidden tanh -> logits", m4_forward),
    (5, "cross_entropy by hand == F.cross_entropy", m5_cross_entropy),
    (6, "train: minibatch loop, and it learns", m6_train),
    (7, "STRETCH: sample names from the model", m7_sample),
]


def grade(
    build_dataset_fn,
    embed_fn=None,
    flatten_fn=None,
    forward_fn=None,
    cross_entropy_fn=None,
    train_fn=None,
    sample_fn=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:   grade(build_dataset, embed, flatten_context, forward,
                             cross_entropy, train, sample)
                       grade(build_dataset, upto=1)   # only the first one
                       grade(build_dataset, embed, flatten_context, forward,
                             cross_entropy, train, sample, skip=(7,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global build_dataset, embed, flatten_context, forward, cross_entropy, train, sample
    build_dataset, embed, flatten_context = build_dataset_fn, embed_fn, flatten_fn
    forward, cross_entropy, train, sample = (
        forward_fn,
        cross_entropy_fn,
        train_fn,
        sample_fn,
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
            print(f"    {e}\n")
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
    """CLI route: grade makemore_mlp.py in this folder, if you keep one."""
    import makemore_mlp as m

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(
        m.build_dataset,
        m.embed,
        m.flatten_context,
        m.forward,
        m.cross_entropy,
        m.train,
        m.sample,
        upto=upto,
    )


if __name__ == "__main__":
    sys.exit(main())

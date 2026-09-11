"""
Grader for makemore part 5 (WaveNet). No pytest.

From the notebook:
    from test_makemore_wavenet import grade
    grade(Embedding=Embedding, Flatten=Flatten, upto=1)
    ...
    grade(Embedding, Flatten, Sequential, FlattenConsecutive, BatchNorm1d,
          build_wavenet, linear_as_conv)
    grade(..., upto=3)      # only the first three
    grade(..., skip=(7,))   # skip a stretch milestone

Milestones run in order and the grader stops at the first failure.
Every check is against torch's own implementation (or an explicit torch
reference) on seeded random inputs, never against magic constants.
"""

import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Filled in by grade(...) so the notebook can hand over its own classes.
Embedding = Flatten = Sequential = FlattenConsecutive = BatchNorm1d = None
build_wavenet = linear_as_conv = None


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
    """'BatchNorm1d.__call__' for the innermost frame that raised exc."""
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
    """Elementwise-close for tensors (or floats), relative to the larger scale."""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.shape != b.shape:
        return False
    scale = max(1.0, a.abs().max().item(), b.abs().max().item())
    return bool(((a - b).abs() <= tol * scale).all())


def shape(t):
    return tuple(t.shape)


# The two layers from lecture 4 that the notebook gives fully written. The
# grader keeps its own copies so it can build Sequentials without depending on
# the notebook's versions.


class _Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class _Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


def _call(layer, x, what):
    """Call a learner layer, turning a shape/broadcast crash into a Fail."""
    try:
        return layer(x)
    except NotImplementedError:
        raise
    except RuntimeError as e:
        raise Fail(
            f"{what}: calling the layer on input of shape {shape(x)} raised "
            f"RuntimeError: {e}\n    Some intermediate tensor has a shape that "
            "does not line up with x. Print the shapes inside __call__."
        )


# ----------------------------------------------------------------------------
# milestone 1: Embedding and Flatten as modules
# ----------------------------------------------------------------------------


def m1_embedding_flatten():
    torch.manual_seed(0)
    emb = Embedding(27, 5)
    ps = emb.parameters()
    check(
        isinstance(ps, list) and len(ps) == 1 and torch.is_tensor(ps[0]),
        "Embedding.parameters() should be a list holding exactly one tensor "
        "(the lookup table).",
    )
    check(
        shape(ps[0]) == (27, 5),
        f"Embedding(27, 5): the table has shape {shape(ps[0])}, expected (27, 5) "
        "-- one row per token, one column per embedding dimension.",
    )
    ix = torch.randint(0, 27, (4, 8))
    out = _call(emb, ix, "Embedding")
    check(
        torch.is_tensor(out) and shape(out) == (4, 8, 5),
        f"Embedding on an index tensor of shape (4, 8) returned shape "
        f"{shape(out) if torch.is_tensor(out) else type(out).__name__}, "
        "expected (4, 8, 5): every integer becomes its row of the table.",
    )
    check(
        torch.equal(out, ps[0][ix]),
        "Embedding output does not equal the table rows selected by the "
        "indices. Row i of the table must come back wherever the input is i.",
    )
    out2 = _call(emb, torch.tensor([3, 3, 7]), "Embedding")
    check(
        shape(out2) == (3, 5) and torch.equal(out2[0], out2[1]),
        "Embedding on a 1-D index tensor of shape (3,) should give (3, 5), "
        "and equal indices must give equal rows.",
    )

    fl = Flatten()
    check(fl.parameters() == [], "Flatten.parameters() should be an empty list.")
    x = torch.randn(4, 8, 5)
    y = _call(fl, x, "Flatten")
    check(
        torch.is_tensor(y) and shape(y) == (4, 40),
        f"Flatten on (4, 8, 5) returned shape "
        f"{shape(y) if torch.is_tensor(y) else type(y).__name__}, expected "
        "(4, 40): keep the batch dim, fold everything else into one vector.",
    )
    check(
        torch.equal(y, x.reshape(4, 40)),
        "Flatten has the right shape but the elements are in the wrong order. "
        "Position t's C values should sit next to each other, then position t+1's.",
    )
    check(
        shape(_call(fl, torch.randn(3, 2, 6, 7), "Flatten")) == (3, 84),
        "Flatten should fold ALL non-batch dims, not just one.",
    )


# ----------------------------------------------------------------------------
# milestone 2: Sequential container
# ----------------------------------------------------------------------------


def m2_sequential():
    torch.manual_seed(1)
    layers = [
        Embedding(27, 4),
        Flatten(),
        _Linear(3 * 4, 10),
        _Tanh(),
        _Linear(10, 27),
    ]
    model = Sequential(layers)
    check(
        hasattr(model, "layers") and list(model.layers) == layers,
        "Sequential should keep the given layers in order in an attribute "
        "called `layers` (the notebook's helpers walk it).",
    )
    ix = torch.randint(0, 27, (6, 3))
    out = _call(model, ix, "Sequential")
    want = ix
    for layer in layers:
        want = layer(want)
    check(
        torch.is_tensor(out) and shape(out) == (6, 27),
        f"Sequential output shape is "
        f"{shape(out) if torch.is_tensor(out) else type(out).__name__}, "
        "expected (6, 27). Each layer's output must become the next layer's input.",
    )
    check(
        torch.equal(out, want),
        "Sequential output differs from calling the layers by hand in order. "
        "Are the layers applied in the order given, each on the previous output?",
    )

    ps = model.parameters()
    check(
        isinstance(ps, list) and all(torch.is_tensor(p) for p in ps),
        "Sequential.parameters() must be ONE flat list of tensors -- not a list "
        f"of lists. Got: {[type(p).__name__ for p in ps] if isinstance(ps, list) else type(ps).__name__}",
    )
    want_ps = [p for layer in layers for p in layer.parameters()]
    check(
        len(ps) == len(want_ps),
        f"Sequential.parameters() has {len(ps)} tensors; the layers hold "
        f"{len(want_ps)} between them. Some layer's parameters are missing or "
        "counted twice.",
    )
    check(
        all(a is b for a, b in zip(ps, want_ps)),
        "Sequential.parameters() returns copies or different tensors than the "
        "layers own. The optimizer needs the SAME tensor objects the layers use.",
    )
    check(
        Sequential([]).parameters() == [],
        "Sequential of no layers should have an empty parameters list.",
    )


# ----------------------------------------------------------------------------
# milestone 3: FlattenConsecutive
# ----------------------------------------------------------------------------


def _fc_reference(x, n):
    B, T, C = x.shape
    out = torch.cat([x[:, i::n] for i in range(n)], dim=2)  # (B, T//n, C*n)
    if out.shape[1] == 1:
        out = out.squeeze(1)
    return out


def m3_flatten_consecutive():
    torch.manual_seed(2)
    fc = FlattenConsecutive(2)
    check(fc.parameters() == [], "FlattenConsecutive.parameters() should be [].")

    x = torch.randn(4, 8, 10)
    y = _call(fc, x, "FlattenConsecutive(2)")
    check(
        torch.is_tensor(y) and shape(y) == (4, 4, 20),
        f"FlattenConsecutive(2) on (4, 8, 10) returned shape "
        f"{shape(y) if torch.is_tensor(y) else type(y).__name__}, expected "
        "(4, 4, 20): half as many positions, each twice as wide.",
    )
    check(
        torch.equal(y, _fc_reference(x, 2)),
        "FlattenConsecutive(2): right shape, wrong contents. Output position j "
        "should be [x[:, 2j], x[:, 2j+1]] side by side, in that order. Check "
        "which elements end up adjacent after your reshape; a transpose "
        "somewhere, or viewing to the wrong intermediate shape, scrambles them.",
    )

    x3 = torch.randn(3, 6, 5)
    y3 = _call(FlattenConsecutive(3), x3, "FlattenConsecutive(3)")
    check(
        torch.is_tensor(y3)
        and shape(y3) == (3, 2, 15)
        and torch.equal(y3, _fc_reference(x3, 3)),
        f"FlattenConsecutive(3) on (3, 6, 5) should give (3, 2, 15) with "
        f"positions [0,1,2] then [3,4,5] concatenated; got shape "
        f"{shape(y3) if torch.is_tensor(y3) else type(y3).__name__}. Is n hard-coded as 2?",
    )

    # the squeeze: when only one position is left, drop that dim
    x1 = torch.randn(4, 2, 10)
    y1 = _call(FlattenConsecutive(2), x1, "FlattenConsecutive(2)")
    check(
        torch.is_tensor(y1) and shape(y1) == (4, 20),
        f"FlattenConsecutive(2) on (4, 2, 10) returned shape "
        f"{shape(y1) if torch.is_tensor(y1) else '?'}, expected (4, 20). When "
        "the middle dim would become 1 there is nothing sequential left, so the "
        "output should be a plain 2-D batch (the next Linear expects that).",
    )
    check(
        torch.equal(y1, _fc_reference(x1, 2)),
        "FlattenConsecutive(2) on (4, 2, 10): squeezed shape is right but contents are scrambled.",
    )
    # ...but ONLY then
    y_keep = _call(FlattenConsecutive(4), torch.randn(2, 8, 3), "FlattenConsecutive(4)")
    check(
        shape(y_keep) == (2, 2, 12),
        f"FlattenConsecutive(4) on (2, 8, 3) returned {shape(y_keep)}, expected "
        "(2, 2, 12). A dim should only disappear when it would be exactly 1.",
    )

    # chaining: 8 -> 4 -> 2 -> squeeze
    x = torch.randn(5, 8, 3)
    a = FlattenConsecutive(2)(x)
    b = FlattenConsecutive(2)(a)
    c = FlattenConsecutive(2)(b)
    check(
        shape(a) == (5, 4, 6) and shape(b) == (5, 2, 12) and shape(c) == (5, 24),
        f"chaining three FlattenConsecutive(2) on (5, 8, 3) gave shapes "
        f"{shape(a)}, {shape(b)}, {shape(c)}; expected (5,4,6), (5,2,12), (5,24).",
    )
    check(
        torch.equal(c, x.reshape(5, 24)),
        "three FlattenConsecutive(2) in a row should end up identical to a plain "
        "Flatten -- the same 24 numbers in the same order. They do not.",
    )


# ----------------------------------------------------------------------------
# milestone 4: BatchNorm1d that survives a 3-D input
# ----------------------------------------------------------------------------


def _bn_train_reference(x, gamma, beta, eps):
    """Batch-stat normalization, both the biased (torch) and unbiased (lecture)
    variance versions. Either is accepted."""
    dims = 0 if x.ndim == 2 else (0, 1)
    mean = x.mean(dims, keepdim=True)
    outs = []
    for unbiased in (False, True):
        var = x.var(dims, keepdim=True, unbiased=unbiased)
        outs.append(gamma * (x - mean) / torch.sqrt(var + eps) + beta)
    return outs


def _bn_torch_reference(x, gamma, beta, eps, running_mean=None, running_var=None):
    """nn.BatchNorm1d on the same input (it wants (N, C) or (N, C, L))."""
    C = x.shape[-1]
    ref = torch.nn.BatchNorm1d(C, eps=eps)
    with torch.no_grad():
        ref.weight.copy_(gamma.reshape(-1))
        ref.bias.copy_(beta.reshape(-1))
        if running_mean is not None:
            ref.running_mean.copy_(running_mean.reshape(-1))
            ref.running_var.copy_(running_var.reshape(-1))
            ref.eval()
    xin = x if x.ndim == 2 else x.transpose(1, 2)
    out = ref(xin)
    return out if x.ndim == 2 else out.transpose(1, 2)


def m4_batchnorm_3d():
    torch.manual_seed(3)
    C = 6
    bn = BatchNorm1d(C)
    check(
        hasattr(bn, "gamma")
        and hasattr(bn, "beta")
        and hasattr(bn, "running_mean")
        and hasattr(bn, "running_var")
        and hasattr(bn, "training"),
        "BatchNorm1d needs gamma, beta, running_mean, running_var and a "
        "`training` flag (the given __init__ sets these up).",
    )
    ps = bn.parameters()
    check(
        len(ps) == 2 and ps[0] is bn.gamma and ps[1] is bn.beta,
        "BatchNorm1d.parameters() should be [gamma, beta] -- running stats are "
        "not trained by gradient descent.",
    )

    # 2-D input, training mode: the lecture-4 case
    x = torch.randn(32, C) * 3 + 1
    bn.training = True
    y = _call(bn, x, "BatchNorm1d (2-D input)")
    check(
        torch.is_tensor(y) and shape(y) == (32, C),
        f"BatchNorm1d on (32, {C}) returned shape "
        f"{shape(y) if torch.is_tensor(y) else type(y).__name__}; the output "
        "must have the input's shape.",
    )
    refs = _bn_train_reference(x, bn.gamma, bn.beta, bn.eps)
    check(
        any(close(y, r, 1e-4) for r in refs),
        "BatchNorm1d on a 2-D (B, C) input does not match nn.BatchNorm1d in "
        "training mode. Every column (channel) should come out with mean 0 and "
        f"std 1 over the batch; got column means {y.mean(0).detach().numpy().round(3).tolist()} "
        f"and stds {y.std(0).detach().numpy().round(3).tolist()}.",
    )
    check(
        close(bn.running_mean.reshape(-1), 0.1 * x.mean(0), 1e-4),
        "After one training-mode call, running_mean should have moved 10% of the "
        "way (momentum 0.1) from 0 toward the batch mean. It did not.",
    )

    # 3-D input, training mode: THE bug
    bn = BatchNorm1d(C)
    x3 = torch.randn(8, 4, C) * 2
    x3[:, 1] += 5.0  # position 1 has a very different mean than the others
    x3[:, 3] *= 0.2  # position 3 has a very different spread
    y3 = _call(bn, x3, "BatchNorm1d (3-D input)")
    check(
        torch.is_tensor(y3) and shape(y3) == (8, 4, C),
        f"BatchNorm1d on (8, 4, {C}) returned shape "
        f"{shape(y3) if torch.is_tensor(y3) else type(y3).__name__}, expected (8, 4, {C}).",
    )
    refs3 = _bn_train_reference(x3, bn.gamma, bn.beta, bn.eps)
    torch_ref = _bn_torch_reference(x3, bn.gamma, bn.beta, bn.eps)
    check(close(refs3[0], torch_ref, 1e-4), "internal: reference mismatch")
    per_pos_means = y3.detach().mean(0)  # (T, C)
    check(
        any(close(y3, r, 1e-4) for r in refs3),
        "BatchNorm1d on a 3-D (B, T, C) input does not match nn.BatchNorm1d. "
        "The output has the right shape, so nothing crashed -- this bug is "
        "silent. Look at the per-position channel means of your output:\n"
        f"      {per_pos_means.numpy().round(2).tolist()}\n"
        "    Position 1 was fed a very different mean than the others, and in "
        "the correct output that difference must still be visible. A channel is "
        "one thing regardless of which position it sits at; its statistics "
        "should be pooled over every sample AND every position.",
    )
    rm = bn.running_mean.reshape(-1)
    check(
        shape(rm) == (C,) and close(rm, 0.1 * x3.mean((0, 1)), 1e-4),
        f"after the 3-D call, running_mean flattens to shape {shape(rm)}; it "
        f"should hold exactly {C} numbers (one per channel), each 10% of the way "
        "toward that channel's mean over all samples and positions.",
    )

    # eval mode uses the running stats, for both 2-D and 3-D
    bn.training = False
    with torch.no_grad():
        y_eval = _call(bn, x3, "BatchNorm1d (3-D, eval)")
        ref_eval = _bn_torch_reference(
            x3, bn.gamma, bn.beta, bn.eps, bn.running_mean, bn.running_var
        )
    check(
        close(y_eval, ref_eval, 1e-4),
        "In eval mode (training=False) on a 3-D input, the output does not match "
        "nn.BatchNorm1d.eval() using the same running stats. Eval must use the "
        "running mean/var, not the batch's.",
    )


# ----------------------------------------------------------------------------
# milestone 5: assemble the hierarchical net
# ----------------------------------------------------------------------------


def _expected_param_count(V, E, H, block_size):
    n_levels = block_size.bit_length() - 1  # log2
    total = V * E
    fan_in = 2 * E
    for _ in range(n_levels):
        total += fan_in * H + 2 * H  # Linear(no bias) + gamma, beta
        fan_in = 2 * H
    total += H * V + V
    return total


def _trace(model, ix):
    shapes = []
    x = ix
    for layer in model.layers:
        x = layer(x)
        shapes.append((type(layer).__name__, shape(x)))
    return shapes


def m5_build():
    torch.manual_seed(4)
    V, E, H, T = 27, 10, 16, 8
    model = build_wavenet(V, E, H, T)
    check(
        isinstance(model, Sequential),
        f"build_wavenet should return your Sequential, got {type(model).__name__}.",
    )
    ix = torch.randint(0, V, (5, T))
    trace = _trace(model, ix)
    names = [n for n, _ in trace]
    shapes = [s for _, s in trace]
    pretty = "\n      ".join(f"{n:<20} -> {s}" for n, s in trace)
    check(
        shapes[-1] == (5, V),
        f"the model's final output for a (5, {T}) batch has shape {shapes[-1]}, "
        f"expected (5, {V}) -- one logit per vocab entry. Layer trace:\n      {pretty}",
    )
    check(
        (5, T * E) not in shapes,
        f"a layer produced shape (5, {T*E}): all {T} positions were flattened "
        "into one vector in a single step. That is the lecture-3 MLP, not a "
        f"hierarchy. Fuse two positions at a time. Layer trace:\n      {pretty}",
    )
    seq_lens = []
    for s in shapes:
        if len(s) == 3 and (not seq_lens or seq_lens[-1] != s[1]):
            seq_lens.append(s[1])
    check(
        seq_lens == [8, 4, 2],
        f"the sequence dim should shrink 8 -> 4 -> 2 (then squeeze away); it went "
        f"{seq_lens}. Layer trace:\n      {pretty}",
    )
    n_bn = names.count("BatchNorm1d")
    n_tanh = names.count("Tanh")
    check(
        n_bn == 3 and n_tanh == 3,
        f"expected 3 BatchNorm1d and 3 Tanh (one per fusion level), found "
        f"{n_bn} and {n_tanh}. Layer trace:\n      {pretty}",
    )
    for i, n in enumerate(names):
        if n == "BatchNorm1d":
            check(
                i + 1 < len(names)
                and names[i + 1] == "Tanh"
                and names[i - 1] == "Linear",
                f"layer {i} is BatchNorm1d but its neighbours are {names[i-1]} / "
                f"{names[i+1] if i+1 < len(names) else 'nothing'}. The lecture's "
                "order at each level is Linear -> BatchNorm1d -> Tanh.",
            )
    want = _expected_param_count(V, E, H, T)
    got = sum(p.numel() for p in model.parameters())
    check(
        got == want,
        f"the model has {got} parameters, the lecture's recipe gives {want} for "
        f"V={V}, E={E}, H={H}, block_size={T}. Off by a small amount? A Linear "
        "feeding a BatchNorm1d has no bias (the bias would be subtracted out). "
        "Off by a lot? Check the fan_in of each Linear after a fusion step.",
    )
    lw = model.layers[-1].weight
    check(
        type(model.layers[-1]).__name__ == "Linear" and lw.std().item() < 0.05,
        f"the last Linear's weight has std {lw.std().item():.3f}. The lecture "
        "shrinks it (times 0.1) so the net starts out unconfident and the "
        "first loss is near ln(V).",
    )
    # gradients reach every parameter
    for p in model.parameters():
        p.requires_grad = True
    logits = model(ix)
    loss = F.cross_entropy(logits, torch.randint(0, V, (5,)))
    check(
        abs(loss.item() - 3.2958) < 0.35,
        f"initial loss is {loss.item():.3f}; with a shrunk last layer it should "
        f"sit near ln({V}) = 3.296.",
    )
    loss.backward()
    dead = [
        i
        for i, p in enumerate(model.parameters())
        if p.grad is None or (p.grad == 0).all()
    ]
    check(
        not dead,
        f"parameters at indices {dead} got no gradient. Every tensor in "
        "parameters() should be in the graph that produced the logits.",
    )

    # a different block_size must work too (nothing hard-coded to 8)
    m4 = build_wavenet(V, 3, 5, 4)
    out4 = m4(torch.randint(0, V, (2, 4)))
    got4 = sum(p.numel() for p in m4.parameters())
    check(
        shape(out4) == (2, V) and got4 == _expected_param_count(V, 3, 5, 4),
        f"build_wavenet(V, 3, 5, block_size=4): output {shape(out4)}, "
        f"{got4} params; expected (2, {V}) and {_expected_param_count(V, 3, 5, 4)}. "
        "The number of fusion levels should follow from block_size.",
    )


# ----------------------------------------------------------------------------
# milestone 6: it learns
# ----------------------------------------------------------------------------

_DATA = {}


def _dataset(block_size=8):
    if block_size in _DATA:
        return _DATA[block_size]
    import random

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "names.txt"
    )
    words = open(path).read().splitlines()
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0

    def build(ws):
        X, Y = [], []
        for w in ws:
            ctx = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                X.append(ctx)
                Y.append(ix)
                ctx = ctx[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)

    random.seed(42)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    _DATA[block_size] = (build(words[:n1]), build(words[n1:n2]))
    return _DATA[block_size]


def _train(model, Xtr, Ytr, steps, seed):
    g = torch.Generator().manual_seed(seed)
    params = model.parameters()
    for p in params:
        p.requires_grad = True
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < int(0.75 * steps) else 0.01
        for p in params:
            p.data += -lr * p.grad


@torch.no_grad()
def _eval(model, X, Y):
    for layer in model.layers:
        layer.training = False
    loss = F.cross_entropy(model(X), Y).item()
    for layer in model.layers:
        layer.training = True
    return loss


def m6_learns():
    (Xtr, Ytr), (Xdev, Ydev) = _dataset(8)
    torch.manual_seed(2147483647)
    model = build_wavenet(27, 10, 68, 8)
    first = _eval(model, Xdev, Ydev)
    _train(model, Xtr, Ytr, steps=2500, seed=7)
    last = _eval(model, Xdev, Ydev)
    print(f"       dev loss {first:.3f} -> {last:.3f} after 2500 steps")
    check(
        last < 2.45,
        f"dev loss went {first:.3f} -> {last:.3f} over 2500 steps; a working "
        "hierarchical net gets under 2.35 here (a unigram model sits at ~2.9, "
        "bigram ~2.45). Everything above passed shape checks, so suspect the "
        "silent things: does eval mode really use the running stats, and do "
        "those stats have one entry per channel?",
    )


# ----------------------------------------------------------------------------
# milestone 7 (stretch): FlattenConsecutive + Linear IS a strided convolution
# ----------------------------------------------------------------------------


def m7_conv():
    torch.manual_seed(6)
    for n, C, H, T in [(2, 3, 5, 8), (3, 4, 2, 6)]:
        W = torch.randn(n * C, H)
        w = linear_as_conv(W, n)
        check(
            torch.is_tensor(w) and shape(w) == (H, C, n),
            f"linear_as_conv on a ({n*C}, {H}) Linear weight with n={n} returned "
            f"shape {shape(w) if torch.is_tensor(w) else type(w).__name__}; "
            f"F.conv1d wants (out_channels, in_channels, kernel_size) = ({H}, {C}, {n}).",
        )
        x = torch.randn(4, T, C)
        want = FlattenConsecutive(n)(x) @ W  # (4, T//n, H)
        got = F.conv1d(x.transpose(1, 2), w, stride=n).transpose(1, 2)
        check(
            close(got, want, 1e-4),
            f"n={n}: conv1d with your reshaped weight does not reproduce "
            "FlattenConsecutive(n) followed by the Linear. Shape is right, so "
            "it is the element order: row k*C + c of the Linear weight is the "
            "weight for channel c of the k-th position in the window.",
        )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "Embedding and Flatten as modules", m1_embedding_flatten),
    (2, "Sequential container", m2_sequential),
    (3, "FlattenConsecutive: fuse n neighbours at a time", m3_flatten_consecutive),
    (4, "BatchNorm1d on a 3-D input", m4_batchnorm_3d),
    (5, "assemble the hierarchical net", m5_build),
    (6, "it learns", m6_learns),
    (7, "STRETCH: the Linear is a strided conv", m7_conv),
]


def grade(
    Embedding_cls=None,
    Flatten_cls=None,
    Sequential_cls=None,
    FlattenConsecutive_cls=None,
    BatchNorm1d_cls=None,
    build_wavenet_fn=None,
    linear_as_conv_fn=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:
        grade(Embedding, Flatten, upto=1)
        grade(Embedding, Flatten, Sequential, FlattenConsecutive, BatchNorm1d,
              build_wavenet, linear_as_conv)
        grade(Embedding, Flatten, Sequential, FlattenConsecutive, BatchNorm1d,
              build_wavenet, skip=(7,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global Embedding, Flatten, Sequential, FlattenConsecutive, BatchNorm1d
    global build_wavenet, linear_as_conv
    Embedding, Flatten, Sequential = Embedding_cls, Flatten_cls, Sequential_cls
    FlattenConsecutive, BatchNorm1d = FlattenConsecutive_cls, BatchNorm1d_cls
    build_wavenet, linear_as_conv = build_wavenet_fn, linear_as_conv_fn
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
                    "    class or function not passed to grade() yet -- hand it over once it exists.\n"
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


if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)

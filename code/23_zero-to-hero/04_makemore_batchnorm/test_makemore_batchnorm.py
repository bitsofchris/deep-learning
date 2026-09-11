"""
Grader for makemore part 3 (activations, gradients, batchnorm). No pytest.

From the notebook:   from test_makemore_batchnorm import grade; grade(globals(), upto=2)
                     grade(globals(), skip=(6,))   # skip a stretch milestone

grade() takes the notebook's namespace (globals()) so it can pick out whichever
functions and classes exist so far. Milestones run in order and the grader stops
at the first failure. A name that is not defined yet shows up as [TODO].

Everything is checked numerically against torch on the same seeded inputs.
"""

import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

# Filled in by grade(...) from the notebook's namespace.
ns = {}

NAMES = [
    "uniform_loss",
    "init_params",
    "saturation_fraction",
    "tanh_local_grad",
    "dead_units",
    "kaiming_std",
    "activation_std_through_stack",
    "BatchNorm1d",
    "Linear",
    "Tanh",
    "update_to_data_ratio",
    "bias_grad_through_bn",
]


class Missing(Exception):
    """A learner object the milestone needs is not defined yet -> [TODO]."""


def need(name):
    """Fetch a learner object from the namespace; missing -> [TODO]."""
    obj = ns.get(name)
    if obj is None:
        raise Missing(f"{name} is not defined in the notebook yet")
    return obj


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


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def tclose(a, b, tol=1e-5):
    """Tensor closeness, relative to the larger magnitude."""
    if not torch.is_tensor(a):
        return False
    if a.shape != b.shape:
        return False
    return torch.allclose(a.detach().double(), b.detach().double(), rtol=tol, atol=tol)


# ----------------------------------------------------------------------------
# shared plumbing: the same dataset the notebook builds
# ----------------------------------------------------------------------------

VOCAB = 27
BLOCK = 3
N_EMBD = 10
N_HIDDEN = 200

_data = {}


def dataset():
    """(Xtr, Ytr, Xdev, Ydev) for ../data/names.txt, block_size 3, seed 42 split."""
    if _data:
        return _data["tr"]
    import random

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "names.txt")
    words = open(path).read().splitlines()
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0

    def build(ws):
        X, Y = [], []
        for w in ws:
            ctx = [0] * BLOCK
            for ch in w + ".":
                ix = stoi[ch]
                X.append(ctx)
                Y.append(ix)
                ctx = ctx[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build(words[:n1])
    Xdev, Ydev = build(words[n1:n2])
    _data["tr"] = (Xtr, Ytr, Xdev, Ydev)
    return _data["tr"]


def forward_manual(params, Xb):
    """Reference forward for the 1-hidden-layer MLP: returns (hpreact, h, logits)."""
    C, W1, b1, W2, b2 = params
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return hpreact, h, logits


def get_init_params(seed=2147483647):
    init_params = need("init_params")
    g = torch.Generator().manual_seed(seed)
    params = init_params(g)
    check(
        isinstance(params, (list, tuple)) and len(params) == 5,
        f"init_params should return a list of 5 tensors [C, W1, b1, W2, b2], "
        f"got {type(params).__name__} of length {len(params) if hasattr(params, '__len__') else '?'}",
    )
    want = [
        (VOCAB, N_EMBD),
        (BLOCK * N_EMBD, N_HIDDEN),
        (N_HIDDEN,),
        (N_HIDDEN, VOCAB),
        (VOCAB,),
    ]
    names = ["C", "W1", "b1", "W2", "b2"]
    for p, shape, name in zip(params, want, names):
        check(
            torch.is_tensor(p) and tuple(p.shape) == shape,
            f"{name} should have shape {shape}, got "
            f"{tuple(p.shape) if torch.is_tensor(p) else type(p).__name__}",
        )
        check(p.requires_grad, f"{name} does not require grad; nothing can train it")
    return params


# ----------------------------------------------------------------------------
# milestone 1: the loss at init
# ----------------------------------------------------------------------------


def m1_initial_loss():
    uniform_loss = need("uniform_loss")
    for n in (27, 10, 2):
        got = uniform_loss(n)
        check(
            close(float(got), -math.log(1.0 / n), 1e-4),
            f"uniform_loss({n}) returned {float(got):.4f}. Think about what the "
            f"cross-entropy is when the model assigns the same probability to "
            f"every one of the {n} classes.",
        )

    Xtr, Ytr, _, _ = dataset()
    g = torch.Generator().manual_seed(0)
    ix = torch.randint(0, Xtr.shape[0], (256,), generator=g)
    params = get_init_params()
    with torch.no_grad():
        _, _, logits = forward_manual(params, Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix]).item()
    want = -math.log(1.0 / VOCAB)
    check(
        loss < want + 0.25,
        f"loss at init is {loss:.3f} but a model that knows nothing should get "
        f"about {want:.3f}. The logits are too confident before any training: "
        f"logits std is {logits.std():.3f}. Which layer sets their scale?",
    )
    check(
        loss > want - 0.5,
        f"loss at init is {loss:.3f}, well under the uninformed {want:.3f}. "
        f"Something is leaking label information, or the forward pass is off.",
    )
    W2, b2 = params[3], params[4]
    check(
        W2.abs().max() > 0,
        "W2 is exactly zero. The loss looks right but the lecture wants small, "
        "not zero: think about what the gradient into a layer of exact zeros "
        "looks like on step 1.",
    )
    check(
        b2.abs().max() < 0.05,
        f"b2 has entries as large as {b2.abs().max():.3f}. At init the output "
        "bias should not be voting for any character.",
    )


# ----------------------------------------------------------------------------
# milestone 2: the saturated tanh
# ----------------------------------------------------------------------------


def m2_saturated_tanh():
    saturation_fraction = need("saturation_fraction")
    tanh_local_grad = need("tanh_local_grad")
    dead_units = need("dead_units")

    g = torch.Generator().manual_seed(3)
    h = torch.tanh(torch.randn(64, 50, generator=g) * 3.0)
    for thresh in (0.99, 0.9):
        want = (h.abs() > thresh).float().mean().item()
        got = saturation_fraction(h, thresh)
        check(
            close(float(got), want, 1e-4),
            f"saturation_fraction(h, {thresh}) returned {float(got):.4f}, expected "
            f"{want:.4f}. It should be the fraction of ALL entries whose |value| "
            f"exceeds the threshold (a single number in [0, 1]).",
        )
    # a column of all saturated values is a dead unit
    h2 = torch.randn(32, 6, generator=g) * 0.3
    h2[:, 1] = 0.999
    h2[:, 4] = -0.995
    h2[3, 4] = 0.2  # one example keeps unit 4 alive
    d = dead_units(h2, 0.99)
    check(
        torch.is_tensor(d) and d.dtype == torch.bool and tuple(d.shape) == (6,),
        "dead_units should return a bool tensor with one entry per hidden unit "
        "(shape (n_hidden,)).",
    )
    check(
        d.tolist() == [False, True, False, False, False, False],
        f"dead_units gave {d.tolist()} for a batch where only unit 1 is saturated "
        "on EVERY example (unit 4 has one live example). A unit is dead only if "
        "no example in the batch can push gradient through it.",
    )

    # local derivative of tanh, expressed in terms of its output
    x = torch.randn(40, generator=g) * 2.5
    x.requires_grad_(True)
    hh = torch.tanh(x)
    hh.sum().backward()
    got = tanh_local_grad(hh.detach())
    check(
        tclose(got, x.grad, 1e-5),
        "tanh_local_grad(h) does not match what autograd says d tanh(x)/dx is. "
        "The function receives the OUTPUT h = tanh(x), not x.",
    )
    check(
        tanh_local_grad(torch.tensor([0.999])).item() < 0.01,
        "tanh_local_grad(0.999) should be almost zero: a saturated unit passes "
        "almost no gradient.",
    )

    # the init should not saturate the hidden layer
    Xtr, Ytr, _, _ = dataset()
    gg = torch.Generator().manual_seed(0)
    ix = torch.randint(0, Xtr.shape[0], (256,), generator=gg)
    params = get_init_params()
    with torch.no_grad():
        hpreact, h, _ = forward_manual(params, Xtr[ix])
    sat = (h.abs() > 0.99).float().mean().item()
    std = hpreact.std().item()
    check(
        sat < 0.20,
        f"{100*sat:.0f}% of the hidden tanh units are saturated (|h| > 0.99) at "
        f"init; pre-activation std is {std:.2f}. Those units pass almost no "
        f"gradient back. Which tensors set the scale of the pre-activations?",
    )
    check(
        std > 0.3,
        f"pre-activation std at init is {std:.3f}. Nothing saturates, but tanh "
        f"of a number that small is just the number: the hidden layer has gone "
        f"nearly linear. Aim for order 1, not order 0.",
    )
    check(
        (params[2].abs().max() < 0.5).item(),
        f"b1 has entries up to {params[2].abs().max():.2f}; a large hidden bias "
        "at init pushes units into saturation by itself.",
    )


# ----------------------------------------------------------------------------
# milestone 3: Kaiming init
# ----------------------------------------------------------------------------


def m3_kaiming():
    kaiming_std = need("kaiming_std")
    stack = need("activation_std_through_stack")

    for fan_in, gain in [(30, 1.0), (30, 5 / 3), (200, 5 / 3), (4, 2.0**0.5)]:
        got = float(kaiming_std(fan_in, gain))
        want = gain / math.sqrt(fan_in)
        check(
            close(got, want, 1e-6),
            f"kaiming_std({fan_in}, {gain:.4g}) returned {got:.5f}. Ask: if x has "
            f"{fan_in} entries of unit variance and each is multiplied by an "
            f"independent weight, what is the variance of the sum?",
        )
    check(
        close(float(kaiming_std(30)), 1 / math.sqrt(30), 1e-6),
        "kaiming_std(fan_in) with no gain given should use gain 1.",
    )

    g = torch.Generator().manual_seed(0)
    stds = stack(g, 10, 200, 1.0, None)
    check(
        isinstance(stds, (list, tuple)) and len(stds) == 10,
        f"activation_std_through_stack(g, depth=10, ...) should return one std "
        f"per layer (a list of 10 floats), got {type(stds).__name__} of length "
        f"{len(stds) if hasattr(stds, '__len__') else '?'}",
    )
    stds = [float(s) for s in stds]
    check(
        all(0.8 < s < 1.25 for s in stds),
        f"linear stack, gain 1: per-layer stds are {[round(s, 2) for s in stds]}. "
        "With the right weight scale a purely linear stack should hold the "
        "activation std near 1.0 at every depth. Either the weights are not "
        "scaled by kaiming_std(width, gain), or x is not standard normal to start.",
    )

    g = torch.Generator().manual_seed(1)
    stds_t = [float(s) for s in stack(g, 10, 200, 5 / 3, torch.tanh)]
    check(
        all(0.5 < s < 0.85 for s in stds_t[-5:]),
        f"tanh stack, gain 5/3: per-layer stds are {[round(s, 2) for s in stds_t]}. "
        "Expected them to settle in a band around 0.65 and stay there. Check "
        "that the nonlinearity is applied after every matmul, and that the "
        "std you record is of the post-nonlinearity output.",
    )
    g = torch.Generator().manual_seed(1)
    stds_1 = [float(s) for s in stack(g, 10, 200, 1.0, torch.tanh)]
    check(
        stds_1[-1] < 0.35 and stds_1[-1] < stds_t[-1] and stds_1[0] > stds_1[-1],
        f"tanh stack, gain 1: per-layer stds are {[round(s, 2) for s in stds_1]}. "
        "Without the extra gain, tanh squashes a little at every layer and the "
        "signal should visibly shrink with depth (down to ~0.23 by layer 10). "
        "Your measurement doesn't show that -- is the gain actually reaching the "
        "weights?",
    )

    # and init_params should now use the principled scale for W1
    params = get_init_params()
    W1 = params[1]
    want = (5 / 3) / math.sqrt(BLOCK * N_EMBD)
    got = W1.std().item()
    check(
        abs(got - want) / want < 0.3,
        f"W1.std() is {got:.3f}; the Kaiming scale for a tanh layer with fan_in "
        f"{BLOCK * N_EMBD} is {want:.3f}. Use kaiming_std in init_params rather "
        f"than a hand-tuned constant.",
    )


# ----------------------------------------------------------------------------
# milestone 4: BatchNorm1d from scratch
# ----------------------------------------------------------------------------


def m4_batchnorm():
    BatchNorm1d = need("BatchNorm1d")
    dim = 20
    bn = BatchNorm1d(dim)
    ref = torch.nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1)

    ps = bn.parameters()
    check(
        isinstance(ps, (list, tuple)) and len(ps) == 2,
        f"BatchNorm1d(dim).parameters() should be [gamma, beta] (2 tensors), got "
        f"{len(ps) if hasattr(ps, '__len__') else type(ps).__name__}. The running "
        "statistics are buffers, not parameters: nothing should train them.",
    )
    check(
        all(torch.is_tensor(p) and tuple(p.shape) in ((1, dim), (dim,)) for p in ps),
        f"gamma and beta should each have {dim} entries (shape (1, {dim}) or ({dim},))",
    )
    check(
        torch.allclose(ps[0].detach().flatten(), torch.ones(dim))
        and torch.allclose(ps[1].detach().flatten(), torch.zeros(dim)),
        "at init gamma should be all ones and beta all zeros (the identity affine)",
    )
    check(
        hasattr(bn, "training") and bn.training is True,
        "a fresh BatchNorm1d should have training=True",
    )

    g = torch.Generator().manual_seed(7)
    for step in range(5):
        x = torch.randn(32, dim, generator=g) * 3.0 + 2.0
        x.requires_grad_(True)
        y = bn(x)
        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = ref(x_ref)
        check(
            torch.is_tensor(y) and tuple(y.shape) == (32, dim),
            f"output shape should be (32, {dim}), got "
            f"{tuple(y.shape) if torch.is_tensor(y) else type(y).__name__}",
        )
        if step == 0:
            check(
                tclose(y.mean(0), torch.zeros(dim), 1e-4),
                f"in training mode each column of the output should have mean ~0 "
                f"over the batch; column means are up to {y.mean(0).abs().max():.3f}. "
                "Which dimension are you averaging over?",
            )
            check(
                tclose(y.std(0, unbiased=False), torch.ones(dim), 2e-2),
                f"in training mode each column of the output should have std ~1 "
                f"over the batch; got stds around {y.std(0, unbiased=False).mean():.3f}.",
            )
        check(
            tclose(y, y_ref, 1e-4),
            f"training-mode output differs from torch.nn.BatchNorm1d by up to "
            f"{(y - y_ref).abs().max():.2e} (step {step}). The mean and std look "
            "right, so it's a detail: which variance estimate (biased or "
            "unbiased) normalizes the batch, and where eps sits.",
        )
        w = torch.randn(32, dim, generator=g)
        (y * w).sum().backward()
        (y_ref * w).sum().backward()
        check(
            tclose(x.grad, x_ref.grad, 1e-4),
            f"gradient w.r.t. the input differs from torch by up to "
            f"{(x.grad - x_ref.grad).abs().max():.2e}. The batch mean/std must "
            "stay in the graph -- did something get .detach()ed or computed "
            "under no_grad?",
        )
        rm = getattr(bn, "running_mean", None)
        rv = getattr(bn, "running_var", None)
        check(
            torch.is_tensor(rm) and torch.is_tensor(rv),
            "the layer needs running_mean and running_var tensors, updated on "
            "every training-mode call",
        )
        check(
            tclose(rm.flatten(), ref.running_mean, 1e-4),
            f"after {step + 1} training call(s) running_mean differs from torch by "
            f"up to {(rm.flatten() - ref.running_mean).abs().max():.2e}. It should "
            "move a fraction `momentum` of the way toward the batch mean each call.",
        )
        check(
            tclose(rv.flatten(), ref.running_var, 1e-3),
            f"after {step + 1} training call(s) running_var differs from torch by "
            f"up to {(rv.flatten() - ref.running_var).abs().max():.2e}, while "
            "running_mean is right. torch tracks the UNBIASED batch variance in "
            "running_var (even though it normalizes with the biased one).",
        )
        check(
            not rm.requires_grad and not rv.requires_grad,
            "running_mean / running_var should not require grad (they are buffers)",
        )

    # eval mode: use the running stats, do not touch them
    bn.training = False
    ref.eval()
    rm0 = bn.running_mean.detach().clone()
    x = torch.randn(8, dim, generator=g) * 3.0 + 2.0
    with torch.no_grad():
        y = bn(x)
        y_ref = ref(x)
    check(
        tclose(y, y_ref, 1e-4),
        f"eval-mode output differs from torch by up to {(y - y_ref).abs().max():.2e}. "
        "With training=False the layer must normalize with the running "
        "statistics, not the current batch's.",
    )
    check(
        torch.equal(bn.running_mean.detach(), rm0),
        "an eval-mode call changed running_mean. The running stats only update "
        "while training.",
    )
    # eval mode must work on a batch of one (train mode cannot)
    with torch.no_grad():
        y1 = bn(x[:1])
        y1_ref = ref(x[:1])
    check(
        tclose(y1, y1_ref, 1e-4),
        "eval-mode call on a single example gives the wrong answer",
    )

    # gamma / beta are used
    bn2 = BatchNorm1d(dim)
    with torch.no_grad():
        bn2.parameters()[0].mul_(3.0)
        bn2.parameters()[1].add_(0.5)
    x = torch.randn(32, dim, generator=g)
    with torch.no_grad():
        y = bn2(x)
    xhat = (x - x.mean(0, keepdim=True)) / torch.sqrt(
        x.var(0, keepdim=True, unbiased=False) + 1e-5
    )
    check(
        tclose(y, xhat * 3.0 + 0.5, 1e-4),
        "gamma and beta don't seem to scale and shift the normalized output. "
        "After normalizing, out = gamma * xhat + beta.",
    )


# ----------------------------------------------------------------------------
# milestone 5: Linear / Tanh, and the pytorch-ified model learns
# ----------------------------------------------------------------------------

_trained = {}


def build_model(Linear, BatchNorm1d, Tanh, g, n_hidden=100):
    C = torch.randn((VOCAB, N_EMBD), generator=g)
    layers = [
        Linear(BLOCK * N_EMBD, n_hidden, bias=False, generator=g),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, n_hidden, bias=False, generator=g),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, VOCAB, bias=False, generator=g),
        BatchNorm1d(VOCAB),
    ]
    with torch.no_grad():
        layers[-1].parameters()[0].mul_(0.1)
    params = [C] + [p for l in layers for p in l.parameters()]
    for p in params:
        p.requires_grad = True
    return C, layers, params


def run_model(C, layers, Xb):
    x = C[Xb].view(Xb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return x


def m5_layers_and_learning():
    Linear = need("Linear")
    Tanh = need("Tanh")
    BatchNorm1d = need("BatchNorm1d")

    g = torch.Generator().manual_seed(5)
    lin = Linear(30, 100, generator=g)
    ps = lin.parameters()
    check(
        isinstance(ps, (list, tuple)) and len(ps) == 2,
        f"Linear(30, 100).parameters() should be [weight, bias]; got "
        f"{len(ps) if hasattr(ps, '__len__') else type(ps).__name__} item(s)",
    )
    W, b = ps
    check(
        tuple(W.shape) == (30, 100) and tuple(b.shape) == (100,),
        f"weight should be (fan_in, fan_out) = (30, 100) and bias (100,); got "
        f"{tuple(W.shape)} and {tuple(b.shape)}",
    )
    check(
        torch.all(b == 0).item(),
        "bias should start at zero",
    )
    want = 1 / math.sqrt(30)
    check(
        abs(W.std().item() - want) / want < 0.2,
        f"weight std is {W.std():.3f}; expected about {want:.3f} for fan_in 30. "
        "Linear should scale its weights by 1/sqrt(fan_in) at init (gain 1 -- "
        "any gain for the nonlinearity is left to the layers around it).",
    )
    x = torch.randn(16, 30, generator=g)
    y = lin(x)
    check(
        tclose(y, x @ W + b, 1e-5),
        "Linear(x) does not equal x @ weight + bias",
    )
    check(
        hasattr(lin, "out") and torch.is_tensor(lin.out) and torch.equal(lin.out, y),
        "after a call, Linear must keep its output in self.out (the histogram "
        "plots read it)",
    )
    lin2 = Linear(30, 100, bias=False, generator=g)
    check(
        len(lin2.parameters()) == 1,
        "Linear(..., bias=False) should have exactly one parameter (the weight)",
    )
    check(
        tclose(lin2(x), x @ lin2.parameters()[0], 1e-5),
        "Linear with bias=False should compute x @ weight",
    )

    t = Tanh()
    x = torch.randn(16, 30, generator=g) * 2
    y = t(x)
    check(tclose(y, torch.tanh(x), 1e-6), "Tanh()(x) should equal torch.tanh(x)")
    check(
        hasattr(t, "out") and torch.equal(t.out, y),
        "after a call, Tanh must keep its output in self.out",
    )
    check(len(t.parameters()) == 0, "Tanh has no parameters; return an empty list")

    # the assembled model at init: the loss should start near uniform
    Xtr, Ytr, Xdev, Ydev = dataset()
    g = torch.Generator().manual_seed(2147483647)
    C, layers, params = build_model(Linear, BatchNorm1d, Tanh, g)
    ix = torch.randint(0, Xtr.shape[0], (256,), generator=g)
    with torch.no_grad():
        loss0 = F.cross_entropy(run_model(C, layers, Xtr[ix]), Ytr[ix]).item()
    check(
        loss0 < 3.6,
        f"the assembled model's loss at init is {loss0:.3f}, expected ~3.3 with "
        "the last BatchNorm's gamma scaled by 0.1. Something in your layers is "
        "not doing what the grader's build expects.",
    )

    # it learns
    losses = []
    for i in range(400):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        logits = run_model(C, layers, Xtr[ix])
        loss = F.cross_entropy(logits, Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        missing = [i for i, p in enumerate(params) if p.grad is None]
        check(
            not missing,
            f"after loss.backward(), parameters {missing} (index into [C] + all "
            "layer parameters) received no gradient at all. They are not in the "
            "graph: somewhere a layer is computing with .data / .detach() / under "
            "no_grad instead of the live tensor.",
        )
        with torch.no_grad():
            for p in params:
                p -= 0.1 * p.grad
        losses.append(loss.item())
    tail = sum(losses[-50:]) / 50
    if os.environ.get("GRADER_DEBUG"):
        print(f"    [debug] init {losses[0]:.3f} tail {tail:.3f}")
    check(
        tail < 2.75,
        f"after 400 steps the loss is averaging {tail:.3f} (started {losses[0]:.3f}); "
        "expected under 2.75. The forward pass matches torch piece by piece, so "
        "suspect the gradient path: is anything in Linear / BatchNorm1d built "
        "from .data or under no_grad, or missing from parameters()?",
    )
    # eval on dev with running stats
    for l in layers:
        l.training = False
    with torch.no_grad():
        dev = F.cross_entropy(run_model(C, layers, Xdev), Ydev).item()
    if os.environ.get("GRADER_DEBUG"):
        print(f"    [debug] dev {dev:.3f}")
    check(
        dev < 2.85,
        f"dev loss in eval mode is {dev:.3f} while the training loss got to "
        f"{tail:.3f}. In eval mode BatchNorm normalizes with the running stats; "
        "if those never tracked the batches, the whole distribution is off.",
    )
    _trained["model"] = (C, layers, params)


# ----------------------------------------------------------------------------
# milestone 6 (stretch): diagnostics
# ----------------------------------------------------------------------------


def m6_diagnostics():
    update_to_data_ratio = need("update_to_data_ratio")
    bias_grad_through_bn = need("bias_grad_through_bn")
    Linear = need("Linear")
    BatchNorm1d = need("BatchNorm1d")

    g = torch.Generator().manual_seed(11)
    for scale in (1.0, 0.01, 100.0):
        p = torch.randn(50, 20, generator=g) * scale
        p.grad = torch.randn(50, 20, generator=g) * scale * 0.3
        lr = 0.1
        want = math.log10((lr * p.grad).std().item() / p.std().item())
        got = float(update_to_data_ratio(p, lr))
        check(
            close(got, want, 1e-4),
            f"update_to_data_ratio returned {got:.4f}, expected {want:.4f}. It is "
            "log10 of (std of the update the optimizer is about to apply) over "
            "(std of the parameter itself). Note the ratio is per parameter "
            "tensor, and the learning rate is part of the update.",
        )

    if "model" in _trained:
        C, layers, params = _trained["model"]
        ratios = [float(update_to_data_ratio(p, 0.1)) for p in params if p.ndim == 2]
        check(
            all(-4.5 < r < -1.0 for r in ratios),
            f"on the model trained in milestone 5, the weight ratios are "
            f"{[round(r, 2) for r in ratios]}; healthy training sits around -3.",
        )

    lin = Linear(30, 64, bias=True, generator=g)
    with torch.no_grad():
        lin.parameters()[1].add_(torch.randn(64, generator=g))
    bn = BatchNorm1d(64)
    x = torch.randn(32, 30, generator=g)
    got = bias_grad_through_bn(lin, bn, x)
    check(
        torch.is_tensor(got) and got.numel() == 64,
        "bias_grad_through_bn should return the gradient tensor that lands on "
        "lin's bias (64 entries)",
    )
    check(
        got.abs().max().item() < 1e-5,
        f"the gradient reaching the Linear bias through BatchNorm has entries up "
        f"to {got.abs().max():.2e}; it should be ~0. Did the bias survive the "
        "batch-mean subtraction somehow, or is the bn placed after something "
        "else?",
    )
    # and it must be a real backward, not a torch.zeros
    lin_nb = Linear(30, 64, bias=True, generator=g)
    check(
        lin_nb.parameters()[1].grad is None,
        "the function should not touch layers it wasn't given",
    )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "the loss at init (uniform_loss, init_params)", m1_initial_loss),
    (
        2,
        "the saturated tanh (saturation_fraction, tanh_local_grad, dead_units)",
        m2_saturated_tanh,
    ),
    (3, "Kaiming init (kaiming_std, activation_std_through_stack)", m3_kaiming),
    (4, "BatchNorm1d from scratch, vs torch in train and eval", m4_batchnorm),
    (5, "Linear / Tanh layers, and the model learns", m5_layers_and_learning),
    (
        6,
        "STRETCH: update/data ratio, bias is redundant before BatchNorm",
        m6_diagnostics,
    ),
]


def grade(namespace, upto=99, skip=()):
    """Run milestones in order, stopping at the first failure.

    From a notebook:   grade(globals())
                       grade(globals(), upto=3)     # only the first three
                       grade(globals(), skip=(6,))  # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    global ns
    ns = dict(namespace)
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
        except Missing as e:
            print(f"\n[TODO] milestone {num}: {title}")
            print(f"    {e} -- define it in the notebook, then grade again.")
            _progress()
            print()
            return 1
        except Exception:
            print(f"\n[ERROR] milestone {num}: {title}\n")
            traceback.print_exc()
            return 1

        print(f"[ok]   milestone {num}: {title}")
    print("\nall milestones passed.")
    return 0


def main():
    """CLI route: grade a module named makemore_batchnorm.py in this folder."""
    import importlib

    mod = importlib.import_module("makemore_batchnorm")
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(vars(mod), upto=upto)


if __name__ == "__main__":
    sys.exit(main())

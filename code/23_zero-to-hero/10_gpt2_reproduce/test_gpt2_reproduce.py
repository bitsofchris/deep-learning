"""
Grader for the GPT-2 (124M) reproduce unit. No pytest.

From the notebook:
    from test_gpt2_reproduce import grade
    grade(CausalSelfAttention=CausalSelfAttention, upto=1)
    grade(CausalSelfAttention=..., DataLoaderLite=..., GPT=..., ...)   # later

Milestones run in order and the grader stops at the first failure.
Every check is numerical against a straightforward reference computed from
the learner's own objects on seeded inputs. Nothing here needs a GPU, an
internet connection, tiktoken, or transformers.
"""

import math
import os
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Filled in by grade(...) so the notebook can hand over its own objects.
CausalSelfAttention = None
DataLoaderLite = None
GPT = None
grad_accum_step = None
get_lr = None
configure_optimizers = None
train_step = None


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
    """'CausalSelfAttention.forward' for the innermost frame that raised exc."""
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
    """Tensor closeness with a relative tolerance on the larger magnitude."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.shape != b.shape:
        return False
    scale = max(1.0, a.abs().max().item(), b.abs().max().item())
    return (a - b).abs().max().item() <= tol * scale


def cfg(**kw):
    """A stand-in for GPTConfig: the model classes only read attributes."""
    base = dict(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32)
    base.update(kw)
    return SimpleNamespace(**base)


def _require(obj, name):
    if obj is None:
        # mimics "class not passed yet" so grade() prints [TODO]
        raise TypeError(f"'NoneType' object is not callable ({name} not handed over)")


# ----------------------------------------------------------------------------
# milestone 1: CausalSelfAttention, fused qkv + the (B, nh, T, hs) dance
# ----------------------------------------------------------------------------


def ref_attention(attn, x, n_head):
    """Per-head reference built from the learner's own c_attn / c_proj weights."""
    B, T, C = x.shape
    hs = C // n_head
    qkv = x @ attn.c_attn.weight.T + attn.c_attn.bias
    q, k, v = qkv.split(C, dim=2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    outs = []
    for h in range(n_head):
        sl = slice(h * hs, (h + 1) * hs)
        qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
        att = (qh @ kh.transpose(1, 2)) / math.sqrt(hs)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        outs.append(att @ vh)
    y = torch.cat(outs, dim=2)
    return y @ attn.c_proj.weight.T + attn.c_proj.bias


def m1_attention():
    _require(CausalSelfAttention, "CausalSelfAttention")
    torch.manual_seed(0)
    c = cfg(n_embd=32, n_head=4, block_size=16)
    attn = CausalSelfAttention(c)
    check(
        hasattr(attn, "c_attn") and hasattr(attn, "c_proj"),
        "CausalSelfAttention needs submodules named c_attn and c_proj. The "
        "names are part of the contract: they are what let the lecture load "
        "the OpenAI checkpoint by name.",
    )
    check(
        tuple(attn.c_attn.weight.shape) == (3 * c.n_embd, c.n_embd),
        f"c_attn.weight has shape {tuple(attn.c_attn.weight.shape)}; one fused "
        f"Linear should produce q, k and v together from n_embd={c.n_embd}.",
    )
    check(
        tuple(attn.c_proj.weight.shape) == (c.n_embd, c.n_embd),
        f"c_proj.weight has shape {tuple(attn.c_proj.weight.shape)}, expected "
        f"({c.n_embd}, {c.n_embd}).",
    )
    check(
        attn.c_attn.bias is not None and attn.c_proj.bias is not None,
        "GPT-2's c_attn and c_proj both have biases.",
    )

    # full-length context
    x = torch.randn(3, c.block_size, c.n_embd)
    y = attn(x)
    check(
        isinstance(y, torch.Tensor) and tuple(y.shape) == tuple(x.shape),
        f"forward returned shape {tuple(y.shape) if isinstance(y, torch.Tensor) else type(y)}, "
        f"expected {tuple(x.shape)}. The heads have to be merged back into one C-wide vector per token.",
    )
    want = ref_attention(attn, x, c.n_head)
    diff = (y - want).abs().max().item()
    check(
        tclose(y, want, 1e-5),
        f"output differs from a per-head reference computed from your own "
        f"c_attn/c_proj weights (max abs diff {diff:.3g}). The q/k/v split, "
        "the head reshape/transpose, the 1/sqrt(head_size) scale, the mask, "
        "or the merge back to (B, T, C) is off.",
    )

    # shorter-than-block_size context
    x2 = torch.randn(2, 5, c.n_embd)
    try:
        y2 = attn(x2)
    except NotImplementedError:
        raise
    except Exception as e:  # noqa: BLE001
        raise Fail(
            f"forward crashed on T=5 < block_size={c.block_size}: {type(e).__name__}: {e}. "
            "Whatever you use to hide the future must be cut down to the "
            "actual sequence length."
        ) from None
    want2 = ref_attention(attn, x2, c.n_head)
    check(
        tclose(y2, want2, 1e-5),
        "output matches at T=block_size but not at T=5. Something is sized to "
        "block_size rather than to the sequence actually passed in.",
    )

    # causality: perturbing the future must not move the past
    x3 = x.clone()
    x3[:, 9:, :] = torch.randn_like(x3[:, 9:, :])
    y3 = attn(x3)
    check(
        tclose(y3[:, :9], y[:, :9], 1e-5),
        "changing tokens 9.. changed the outputs at positions 0..8. The future "
        "is leaking into the past.",
    )
    check(
        not tclose(y3[:, 9:], y[:, 9:], 1e-3),
        "changing tokens 9.. did not change outputs at 9.. at all. Is the mask "
        "hiding the present too, or is v not being used?",
    )

    # the graph is intact
    y.sum().backward()
    check(
        attn.c_attn.weight.grad is not None and attn.c_proj.weight.grad is not None,
        "no gradient reached c_attn / c_proj. Something detached the graph.",
    )


# ----------------------------------------------------------------------------
# milestone 2: DataLoaderLite
# ----------------------------------------------------------------------------


def ref_batches(tokens, B, T, n):
    """The batches a lecture-style loader yields: B*T+1 window, advance B*T,
    reset to 0 when the next window would run off the end."""
    out, pos = [], 0
    for _ in range(n):
        buf = tokens[pos : pos + B * T + 1]
        out.append((buf[:-1].view(B, T), buf[1:].view(B, T)))
        pos += B * T
        if pos + (B * T + 1) > len(tokens):
            pos = 0
    return out


def m2_dataloader():
    _require(DataLoaderLite, "DataLoaderLite")
    tokens = torch.arange(
        48, dtype=torch.long
    )  # exactly 6 windows of B*T, then 1 short
    B, T = 2, 4
    loader = DataLoaderLite(tokens, B, T)
    want = ref_batches(tokens, B, T, 12)
    for i, (wx, wy) in enumerate(want):
        try:
            got = loader.next_batch()
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001
            raise Fail(
                f"next_batch() crashed on batch {i}: {type(e).__name__}: {e}. "
                "With 48 tokens and B*T=8, that call's window would have run "
                "past the end of the stream. The reset has to happen BEFORE "
                "a window that cannot be filled, not after."
            ) from None
        check(
            isinstance(got, tuple) and len(got) == 2,
            "next_batch() should return a pair (x, y).",
        )
        x, y = got
        check(
            isinstance(x, torch.Tensor) and tuple(x.shape) == (B, T),
            f"batch {i}: x has shape {tuple(x.shape) if isinstance(x, torch.Tensor) else type(x)}, expected {(B, T)}.",
        )
        check(
            tuple(y.shape) == (B, T),
            f"batch {i}: y has shape {tuple(y.shape)}, expected {(B, T)}.",
        )
        check(
            torch.equal(x.long(), wx),
            f"batch {i}: x is\n{x.tolist()}\nexpected\n{wx.tolist()}\n"
            "Check how far the cursor advances per batch and where the window "
            "starts; batches should tile the stream with no gaps and no "
            "repeats until the wraparound.",
        )
        check(
            torch.equal(y.long(), wy),
            f"batch {i}: y is\n{y.tolist()}\nfor x\n{x.tolist()}\n"
            "y should be x shifted by exactly one token.",
        )
        check(
            x.dtype == torch.long and y.dtype == torch.long,
            f"batch {i}: dtype is {x.dtype}; token ids must be int64 for the "
            "embedding table and cross_entropy.",
        )
    # wraparound happened exactly where the lecture's rule says
    x, _ = loader.next_batch()
    x_want, _ = ref_batches(tokens, B, T, 13)[-1]
    check(
        torch.equal(x.long(), x_want),
        f"batch 12: x is {x.tolist()}, expected {x_want.tolist()}. The reset "
        "to the start of the stream happened one batch early or late. The rule: "
        "reset when the NEXT window of B*T+1 tokens would run off the end.",
    )


# ----------------------------------------------------------------------------
# milestone 3: weight tying + init
# ----------------------------------------------------------------------------


def _std(t):
    return t.detach().double().std().item()


def m3_tie_and_init():
    _require(GPT, "GPT")
    torch.manual_seed(0)
    c = cfg(n_layer=4, n_head=4, n_embd=64, block_size=16, vocab_size=65)
    model = GPT(c)
    tr = model.transformer

    check(
        model.lm_head.weight is tr.wte.weight,
        "lm_head.weight and transformer.wte.weight are two different tensors. "
        "They should be one and the same Parameter, not a copy of each other.",
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_unique = len({p.data_ptr() for p in model.parameters()})
    check(
        n_unique == len(list(model.parameters())),
        "model.parameters() lists the same tensor twice.",
    )

    # embeddings
    for name, w in [("wte", tr.wte.weight), ("wpe", tr.wpe.weight)]:
        s = _std(w)
        check(
            abs(s - 0.02) < 0.004,
            f"{name}.weight has std {s:.4f}; GPT-2 initialises every embedding "
            "with std 0.02. (PyTorch's default for nn.Embedding is std 1.0.)",
        )

    # per-layer linears
    for i, block in enumerate(tr.h):
        for name, lin, scaled in [
            ("attn.c_attn", block.attn.c_attn, False),
            ("mlp.c_fc", block.mlp.c_fc, False),
            ("attn.c_proj", block.attn.c_proj, True),
            ("mlp.c_proj", block.mlp.c_proj, True),
        ]:
            s = _std(lin.weight)
            want = 0.02 * (2 * c.n_layer) ** -0.5 if scaled else 0.02
            check(
                abs(s - want) < 0.2 * want,
                f"h[{i}].{name}.weight has std {s:.4f}, expected about {want:.4f}. "
                + (
                    "This Linear writes into the residual stream; its init "
                    "should shrink with the number of such writes."
                    if scaled
                    else "This Linear does not write into the residual stream; "
                    "plain std 0.02."
                ),
            )
            check(
                lin.bias is not None and lin.bias.abs().max().item() == 0.0,
                f"h[{i}].{name}.bias is not all zeros. (PyTorch's default bias "
                "init is uniform, not zero.)",
            )
        for ln_name, ln in [("ln_1", block.ln_1), ("ln_2", block.ln_2)]:
            check(
                torch.all(ln.weight == 1.0) and torch.all(ln.bias == 0.0),
                f"h[{i}].{ln_name} was touched by init. LayerNorm keeps its "
                "PyTorch default (weight 1, bias 0).",
            )
    check(
        torch.all(tr.ln_f.weight == 1.0),
        "ln_f.weight was touched by init. LayerNorm keeps its default.",
    )

    # loss at init should be about ln(V): the model starts out unsure
    torch.manual_seed(1)
    idx = torch.randint(0, c.vocab_size, (4, c.block_size))
    tgt = torch.randint(0, c.vocab_size, (4, c.block_size))
    logits, loss = model(idx, tgt)
    check(
        tuple(logits.shape) == (4, c.block_size, c.vocab_size),
        f"logits shape {tuple(logits.shape)}, expected (4, {c.block_size}, {c.vocab_size}).",
    )
    lnv = math.log(c.vocab_size)
    check(
        abs(loss.item() - lnv) < 0.15,
        f"loss at init is {loss.item():.3f}; ln(vocab)={lnv:.3f}. A freshly "
        "initialised GPT should assign roughly uniform probability to every "
        "token. Some weight is much larger than std 0.02 intends.",
    )

    # two seeds, two different models (init actually draws random numbers)
    torch.manual_seed(2)
    other = GPT(c)
    check(
        not torch.equal(other.transformer.wte.weight, tr.wte.weight),
        "two models built under different seeds have identical wte weights.",
    )
    check(n_params > 0, "model has no parameters?")


# ----------------------------------------------------------------------------
# milestone 4: gradient accumulation
# ----------------------------------------------------------------------------


def _grads(model):
    return {
        n: (None if p.grad is None else p.grad.detach().clone())
        for n, p in model.named_parameters()
    }


def m4_grad_accum():
    _require(grad_accum_step, "grad_accum_step")
    torch.manual_seed(0)
    c = cfg(n_layer=2, n_head=2, n_embd=32, block_size=8, vocab_size=65)
    model = GPT(c)
    tokens = torch.randint(0, c.vocab_size, (400,))
    B, T, N = 2, 8, 4
    loader = DataLoaderLite(tokens, B, T)

    # plant garbage so we can tell if grads were cleared at the start
    for p in model.parameters():
        p.grad = torch.full_like(p, 7.0)

    loss_acc = grad_accum_step(model, loader, N)
    check(
        loss_acc is not None,
        "grad_accum_step returned None; it should return the loss of the whole "
        "accumulated batch (a detached scalar tensor or float).",
    )
    loss_acc = float(loss_acc)
    got = _grads(model)
    check(
        all(g is not None for g in got.values()),
        "some parameters have no gradient after grad_accum_step.",
    )

    # the loader must have been consumed exactly N times
    nx, _ = loader.next_batch()
    wx, _ = ref_batches(tokens, B, T, N + 1)[-1]
    check(
        torch.equal(nx.long(), wx),
        f"after grad_accum_step(..., {N}) the loader is not exactly {N} batches "
        "further along. Each micro-step should pull exactly one batch.",
    )

    # reference: one big batch of the same tokens, one backward
    batches = ref_batches(tokens, B, T, N)
    x_big = torch.cat([b[0] for b in batches], dim=0)
    y_big = torch.cat([b[1] for b in batches], dim=0)
    model.zero_grad(set_to_none=True)
    _, loss_big = model(x_big, y_big)
    loss_big.backward()
    want = _grads(model)

    # per-micro-batch reference for diagnostics
    model.zero_grad(set_to_none=True)
    _, loss_last = model(batches[-1][0], batches[-1][1])
    loss_last.backward()
    last = _grads(model)

    names = list(want)
    ratio = None
    for n in names:
        if want[n].abs().max() > 1e-3:
            ratio = (got[n].norm() / want[n].norm()).item()
            break
    ok = all(tclose(got[n], want[n], 1e-4) for n in names)
    if not ok:
        if all(tclose(got[n], last[n], 1e-4) for n in names):
            raise Fail(
                "the accumulated gradient equals the gradient of the LAST "
                "micro-batch alone. Something between micro-steps is wiping "
                "what the earlier ones deposited in .grad."
            )
        if ratio is not None and abs(ratio - N) < 0.05 * N:
            raise Fail(
                f"the accumulated gradient is about {ratio:.2f}x the gradient of "
                f"one big batch of the same {N} micro-batches. Each micro-batch "
                "loss is a MEAN over its own tokens; think about what the mean "
                "over all the tokens should look like once the pieces are summed."
            )
        if all(tclose(got[n] - 7.0, want[n], 1e-4) for n in names):
            raise Fail(
                "gradients still contain the garbage that was in .grad before "
                "the call. Grads must be cleared at the start of the step, "
                "not left to accumulate across steps."
            )
        worst = max(names, key=lambda n: (got[n] - want[n]).abs().max().item())
        raise Fail(
            f"accumulated gradient differs from one-big-batch gradient (worst: "
            f"{worst}, max abs diff {(got[worst]-want[worst]).abs().max().item():.3g}, "
            f"norm ratio {ratio}). Micro-batches should add up to exactly the big batch."
        )
    check(
        close(loss_acc, loss_big.item(), 1e-4),
        f"returned loss {loss_acc:.5f} but the big-batch loss is {loss_big.item():.5f}. "
        "The gradients are right; the reported loss is not the same average.",
    )


# ----------------------------------------------------------------------------
# milestone 5: LR schedule, and it learns
# ----------------------------------------------------------------------------


def ref_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def _char_tokens():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "tinyshakespeare.txt"
    )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long), len(chars)


def m5_lr_and_learns():
    _require(get_lr, "get_lr")
    max_lr, min_lr, warm, mx = 6e-4, 6e-5, 10, 50
    lr0 = get_lr(0, max_lr, min_lr, warm, mx)
    check(lr0 is not None, "get_lr returned None.")
    check(
        lr0 > 0,
        f"get_lr(0) = {lr0}. A step with learning rate 0 is a wasted step; "
        "the lecture's warmup never returns zero.",
    )
    check(
        close(lr0, max_lr / warm, 1e-6),
        f"get_lr(0) = {lr0:.3g}, expected {max_lr/warm:.3g} (one warmup-step's worth).",
    )
    check(
        close(get_lr(warm - 1, max_lr, min_lr, warm, mx), max_lr, 1e-6),
        f"get_lr({warm-1}) = {get_lr(warm-1, max_lr, min_lr, warm, mx):.3g}; the "
        f"last warmup step should land exactly on max_lr={max_lr:.3g}.",
    )
    mid = warm + (mx - warm) // 2
    check(
        close(get_lr(mid, max_lr, min_lr, warm, mx), (max_lr + min_lr) / 2, 1e-6),
        f"get_lr({mid}) (halfway through decay) = "
        f"{get_lr(mid, max_lr, min_lr, warm, mx):.3g}, expected the midpoint "
        f"{(max_lr+min_lr)/2:.3g}. Cosine decay is symmetric about its middle.",
    )
    for it in (mx, mx + 1, mx + 100):
        check(
            close(get_lr(it, max_lr, min_lr, warm, mx), min_lr, 1e-6),
            f"get_lr({it}) = {get_lr(it, max_lr, min_lr, warm, mx):.3g}; at and "
            f"after max_steps the schedule should sit at min_lr={min_lr:.3g}.",
        )
    prev = 0.0
    for it in range(warm):
        v = get_lr(it, max_lr, min_lr, warm, mx)
        check(v > prev, f"warmup is not strictly increasing at step {it}.")
        prev = v
    for it in range(warm, mx):
        v = get_lr(it, max_lr, min_lr, warm, mx)
        check(v <= prev + 1e-12, f"decay is not monotone at step {it}.")
        prev = v
    for it in range(0, mx + 5):
        v = get_lr(it, max_lr, min_lr, warm, mx)
        w = ref_lr(it, max_lr, min_lr, warm, mx)
        check(
            close(v, w, 1e-6),
            f"get_lr({it}) = {v:.6g}, reference {w:.6g}. The endpoints are right "
            "but the curve between them is not a half-cosine from max_lr to min_lr.",
        )

    # ---- it learns: everything so far, wired together
    _require(grad_accum_step, "grad_accum_step")
    tokens, V = _char_tokens()
    torch.manual_seed(1337)
    c = cfg(n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=V)
    model = GPT(c)
    loader = DataLoaderLite(tokens, 8, 32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    steps, warm = 300, 20
    first = None
    for step in range(steps):
        lr = get_lr(step, 3e-3, 3e-4, warm, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = float(grad_accum_step(model, loader, 2))
        if first is None:
            first = loss
        opt.step()
    # evaluate on the next few batches
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(10):
            x, y = loader.next_batch()
            losses.append(model(x, y)[1].item())
    last = sum(losses) / len(losses)
    model.train()
    check(
        last < 2.95,
        f"loss went {first:.3f} -> {last:.3f} over {steps} steps (B=8, T=32, "
        "2 micro-steps). A working tiny GPT gets well under 2.95 here. Pieces "
        "that pass in isolation can still be wired wrong: check what the loss "
        "print in your own loop does over time.",
    )
    check(
        first > math.log(V) - 0.3,
        f"first loss was {first:.3f}, well below ln(V)={math.log(V):.3f}: init is not fresh.",
    )


# ----------------------------------------------------------------------------
# milestone 6 (stretch): AdamW param groups + gradient clipping placement
# ----------------------------------------------------------------------------


def m6_optimizer_and_clip():
    _require(configure_optimizers, "configure_optimizers")
    torch.manual_seed(0)
    c = cfg(n_layer=2, n_head=2, n_embd=32, block_size=8, vocab_size=65)
    model = GPT(c)
    opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4)
    check(
        isinstance(opt, torch.optim.AdamW),
        f"configure_optimizers returned {type(opt).__name__}, expected torch.optim.AdamW.",
    )
    groups = opt.param_groups
    check(len(groups) == 2, f"optimizer has {len(groups)} param groups, expected 2.")
    want_decay = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    want_nodecay = sum(p.numel() for p in model.parameters() if p.dim() < 2)
    sizes = {g["weight_decay"]: sum(p.numel() for p in g["params"]) for g in groups}
    check(
        0.1 in sizes and 0.0 in sizes,
        f"param groups have weight_decay values {sorted(sizes)}; expected one group "
        "at 0.1 and one at 0.0.",
    )
    check(
        sizes[0.1] == want_decay,
        f"the decayed group holds {sizes[0.1]} parameters, expected {want_decay}. "
        "Decay every matrix (weights, embeddings); never decay a vector (biases, "
        "LayerNorm). Also: is the tied wte/lm_head weight counted once?",
    )
    check(
        sizes[0.0] == want_nodecay,
        f"the no-decay group holds {sizes[0.0]} parameters, expected {want_nodecay}.",
    )
    seen = [id(p) for g in groups for p in g["params"]]
    check(
        len(seen) == len(set(seen)) == len(list(model.parameters())),
        "the two groups together should cover every parameter exactly once.",
    )
    for g in groups:
        check(close(g["lr"], 6e-4), f"group lr is {g['lr']}, expected 6e-4.")
        check(
            tuple(g["betas"]) == (0.9, 0.95),
            f"betas are {tuple(g['betas'])}; GPT-3's table says (0.9, 0.95).",
        )
        check(close(g["eps"], 1e-8), f"eps is {g['eps']}, expected 1e-8.")

    # ---- clipping placement, checked with SGD so the update is transparent
    _require(train_step, "train_step")
    torch.manual_seed(0)
    model = GPT(c)
    tokens = torch.randint(0, c.vocab_size, (400,))
    B, T, N = 2, 8, 4
    lr, max_norm = 0.1, 0.05

    # reference: full accumulated grad on a twin model
    twin = GPT(c)
    twin.load_state_dict(model.state_dict())
    grad_accum_step(twin, DataLoaderLite(tokens, B, T), N)
    full_norm = math.sqrt(
        sum(p.grad.double().pow(2).sum().item() for p in twin.parameters())
    )
    check(full_norm > max_norm * 2, "test setup: grad norm too small to test clipping")

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    sgd = torch.optim.SGD(model.parameters(), lr=lr)
    out = train_step(model, sgd, DataLoaderLite(tokens, B, T), N, lr, max_norm)
    check(
        isinstance(out, tuple) and len(out) == 2,
        "train_step should return (loss, norm): the accumulated loss and the "
        "gradient norm BEFORE clipping (what clip_grad_norm_ returns).",
    )
    loss, norm = out
    norm = float(norm)
    check(
        close(norm, full_norm, 1e-3),
        f"train_step reported grad norm {norm:.4f}; the norm of the full "
        f"accumulated gradient is {full_norm:.4f}. Clipping is measuring something "
        "other than the whole accumulated gradient -- is it inside the micro-step loop?",
    )
    post = math.sqrt(
        sum(p.grad.double().pow(2).sum().item() for p in model.parameters())
    )
    check(
        post <= max_norm * 1.01,
        f"after train_step the gradient norm is {post:.4f} > max_norm={max_norm}. "
        "The gradient was never actually clipped.",
    )
    # SGD update must equal -lr * clipped grad, i.e. clip happened before step
    scale = max_norm / (full_norm + 1e-6)
    for n, p in model.named_parameters():
        want = before[n] - lr * scale * dict(twin.named_parameters())[n].grad
        check(
            tclose(p.detach(), want, 1e-4),
            f"parameter {n} moved by something other than -lr * clipped_grad. "
            "The optimizer stepped with the unclipped gradient: check what "
            "happens between the last backward and optimizer.step().",
        )
    for g in sgd.param_groups:
        check(
            close(g["lr"], lr),
            f"train_step did not set lr={lr} on the param groups (got {g['lr']}).",
        )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "CausalSelfAttention: fused qkv, (B, nh, T, hs), causal mask", m1_attention),
    (2, "DataLoaderLite: B*T+1 windows over a token stream", m2_dataloader),
    (3, "weight tying + GPT-2 init (0.02, residual scaling)", m3_tie_and_init),
    (4, "gradient accumulation == one big batch", m4_grad_accum),
    (5, "warmup + cosine LR, and it learns", m5_lr_and_learns),
    (6, "STRETCH: AdamW param groups + clipping placement", m6_optimizer_and_clip),
]


def grade(
    CausalSelfAttention=None,
    DataLoaderLite=None,
    GPT=None,
    grad_accum_step=None,
    get_lr=None,
    configure_optimizers=None,
    train_step=None,
    upto=99,
    skip=(),
):
    """Run milestones in order, stopping at the first failure.

    From a notebook:
        grade(CausalSelfAttention=CausalSelfAttention, upto=1)
        grade(CausalSelfAttention=..., DataLoaderLite=..., GPT=..., upto=3)
        grade(..., configure_optimizers=..., train_step=..., skip=(6,))   # skip a stretch milestone
    Returns 0 on all-pass, 1 otherwise.
    """
    g = globals()
    g["CausalSelfAttention"] = CausalSelfAttention
    g["DataLoaderLite"] = DataLoaderLite
    g["GPT"] = GPT
    g["grad_accum_step"] = grad_accum_step
    g["get_lr"] = get_lr
    g["configure_optimizers"] = configure_optimizers
    g["train_step"] = train_step
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
                _progress()
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
    """CLI route: grade gpt2_reproduce.py in this folder (if you export one)."""
    import gpt2_reproduce as m

    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    return grade(
        CausalSelfAttention=getattr(m, "CausalSelfAttention", None),
        DataLoaderLite=getattr(m, "DataLoaderLite", None),
        GPT=getattr(m, "GPT", None),
        grad_accum_step=getattr(m, "grad_accum_step", None),
        get_lr=getattr(m, "get_lr", None),
        configure_optimizers=getattr(m, "configure_optimizers", None),
        train_step=getattr(m, "train_step", None),
        upto=upto,
    )


if __name__ == "__main__":
    sys.exit(main())

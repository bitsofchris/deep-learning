"""
Grader for micrograd.py. No pytest. Run: python test_micrograd.py

Milestones run in order and the grader stops at the first failure.
Pass a milestone number to run only up to it:  python test_micrograd.py 3
"""

import math
import random
import sys
import traceback

from micrograd import Value, Neuron, Layer, MLP


# ----------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def numeric_grad(f, xs, i, h=1e-6):
    """Central finite difference of f(*xs) w.r.t. xs[i]. xs are floats."""
    xp = list(xs)
    xp[i] += h
    xm = list(xs)
    xm[i] -= h
    return (f(*xp) - f(*xm)) / (2 * h)


def grad_check(name, f, xs, tol=1e-4):
    """Build the graph from f on Values, run backward, compare each input's
    .grad against finite differences on the same f applied to plain floats."""
    vals = [Value(x) for x in xs]
    out = f(*vals)
    check(
        isinstance(out, Value),
        f"{name}: expression did not return a Value (got {type(out).__name__})",
    )
    out.backward()

    def fnum(*fs):
        r = f(*[Value(v) for v in fs])
        return r.data

    for i, v in enumerate(vals):
        want = numeric_grad(fnum, xs, i)
        check(
            close(v.grad, want, tol),
            f"{name}: d/dx{i} is {v.grad:.6g} but finite differences say "
            f"{want:.6g} (inputs {[round(x, 4) for x in xs]}). One local "
            f"derivative is wrong, or gradient is not reaching this input.",
        )


# ----------------------------------------------------------------------------
# milestone 1: forward pass + graph bookkeeping
# ----------------------------------------------------------------------------


def m1_forward_and_graph():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    check(isinstance(c, Value), "a * b did not return a Value")
    check(close(c.data, -6.0), f"a * b: data is {c.data}, expected -6.0")
    check(
        a in c._prev and b in c._prev,
        "a * b: the result does not remember its inputs in _prev. "
        "It won't be able to send gradient anywhere later.",
    )
    check(c._op == "*", f"a * b: _op is {c._op!r}, expected '*'")

    d = a + b
    check(close(d.data, -1.0), f"a + b: data is {d.data}, expected -1.0")
    check(a in d._prev and b in d._prev, "a + b: _prev does not contain both inputs")
    check(d._op == "+", f"a + b: _op is {d._op!r}, expected '+'")

    e = a + 1
    check(
        isinstance(e, Value) and close(e.data, 3.0),
        "a + 1 (plain int on the right) should still produce a Value(3.0)",
    )
    f = a * 2
    check(
        isinstance(f, Value) and close(f.data, 4.0),
        "a * 2 (plain int on the right) should still produce a Value(4.0)",
    )

    g = (a * b + a) * b
    check(close(g.data, 12.0), f"(a*b + a)*b: data is {g.data}, expected 12.0")
    check(
        g.grad == 0.0 and a.grad == 0.0,
        "grads should all still be 0.0 -- nothing has called backward yet",
    )


# ----------------------------------------------------------------------------
# milestone 2: one hop of gradient via _backward
# ----------------------------------------------------------------------------


def m2_one_hop():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.grad = 1.0
    c._backward()
    check(
        close(a.grad, -3.0) and close(b.grad, 2.0),
        f"c = a*b, c.grad=1: got a.grad={a.grad}, b.grad={b.grad}. "
        "Think about how much c moves if you nudge a by a tiny bit.",
    )

    # incoming gradient is NOT always 1
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    c.grad = 5.0
    c._backward()
    check(
        close(a.grad, -15.0) and close(b.grad, 10.0),
        f"c = a*b, c.grad=5: got a.grad={a.grad}, b.grad={b.grad}. "
        "The closure is ignoring the gradient that arrived at c from "
        "downstream.",
    )

    a, b = Value(2.0), Value(-3.0)
    d = a + b
    d.grad = 7.0
    d._backward()
    check(
        close(a.grad, 7.0) and close(b.grad, 7.0),
        f"d = a+b, d.grad=7: got a.grad={a.grad}, b.grad={b.grad}.",
    )

    # leaf's _backward is a no-op
    leaf = Value(1.0)
    leaf.grad = 3.0
    leaf._backward()
    check(leaf.grad == 3.0, "a leaf's _backward should do nothing")

    # a closure must only push ONE hop, not recurse
    a, b, c = Value(2.0), Value(-3.0), Value(4.0)
    d = a * b
    e = d + c
    e.grad = 1.0
    e._backward()
    check(
        a.grad == 0.0 and b.grad == 0.0,
        "e._backward() pushed gradient more than one hop (a.grad or b.grad "
        "changed). _backward should touch only the direct inputs; ordering "
        "the whole graph is a later milestone's job.",
    )
    check(
        close(d.grad, 1.0) and close(c.grad, 1.0),
        f"e = d + c: after e._backward(), d.grad={d.grad}, c.grad={c.grad}",
    )


# ----------------------------------------------------------------------------
# milestone 3: full backward pass (topological order)
# ----------------------------------------------------------------------------


def m3_backward():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L.backward()
    check(close(L.grad, 1.0), f"L.grad is {L.grad} after L.backward(). What is dL/dL?")
    check(
        close(f.grad, 4.0) and close(d.grad, -2.0),
        f"one hop from L: f.grad={f.grad}, d.grad={d.grad}, expected 4, -2",
    )
    check(
        close(c.grad, -2.0) and close(e.grad, -2.0),
        f"two hops: c.grad={c.grad}, e.grad={e.grad}, expected -2, -2. "
        "Gradient stopped one hop early, or the nodes ran in the wrong order.",
    )
    check(
        close(a.grad, 6.0) and close(b.grad, -4.0),
        f"three hops: a.grad={a.grad}, b.grad={b.grad}, expected 6, -4.",
    )

    # deeper chains, checked numerically, with plain-number mixing.
    # each input is used exactly once so this passes without accumulation.
    random.seed(0)
    for _ in range(5):
        xs = [random.uniform(-2, 2) for _ in range(6)]
        grad_check(
            "chain",
            lambda a, b, c, d, e, f: ((a * b + c) * d + 1) * 3 + (e * f) * 2,
            xs,
        )


# ----------------------------------------------------------------------------
# milestone 4: gradient accumulation when a node is used twice
# ----------------------------------------------------------------------------


def m4_accumulation():
    a = Value(3.0)
    b = a + a
    b.backward()
    check(
        close(a.grad, 2.0),
        f"b = a + a: a.grad is {a.grad}, expected 2. One of the two edges "
        "into a is overwriting the other's contribution.",
    )

    a = Value(3.0)
    b = a * a
    b.backward()
    check(close(a.grad, 6.0), f"b = a * a: a.grad is {a.grad}, expected 6.")

    a, b = Value(-2.0), Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()
    check(
        close(a.grad, -3.0) and close(b.grad, -8.0),
        f"f = (a*b)*(a+b): a.grad={a.grad}, b.grad={b.grad}, expected -3, -8.",
    )

    random.seed(1)
    for _ in range(5):
        xs = [random.uniform(-2, 2) for _ in range(3)]
        grad_check("reuse", lambda a, b, c: (a * b + a) * (a + c) + b * b * c, xs)

    # ordering trap: an intermediate node with a long path and a short path
    # to the root. accumulation alone isn't enough -- if that node's
    # _backward runs before all the gradient flowing into it has arrived,
    # you get a partial answer.
    a, b, c = Value(1.5), Value(-0.5), Value(2.0)
    x = a * b
    y = x * c
    z = y + x  # x reaches root via y (long) and directly (short)
    z.backward()
    check(
        close(a.grad, -0.5 * 3.0) and close(b.grad, 1.5 * 3.0),
        f"z = (a*b)*c + (a*b): a.grad={a.grad}, b.grad={b.grad}, expected "
        f"{-1.5}, {4.5}. Either a contribution is being overwritten, or the "
        "shared node x had its _backward run before its grad was final.",
    )

    # same graph twice gives the same answer
    a = Value(1.5)
    (a * a * a).backward()
    g1 = a.grad
    a = Value(1.5)
    (a * a * a).backward()
    check(close(g1, a.grad), "same graph twice gave different gradients")


# ----------------------------------------------------------------------------
# milestone 5: tanh as a single op
# ----------------------------------------------------------------------------


def m5_tanh():
    x = Value(0.8814)
    y = x.tanh()
    check(isinstance(y, Value), "tanh did not return a Value")
    check(
        close(y.data, math.tanh(0.8814), 1e-9),
        f"tanh(0.8814).data is {y.data}, expected {math.tanh(0.8814)}",
    )
    check(x in y._prev, "tanh's output does not remember its input")
    y.backward()
    want = 1 - math.tanh(0.8814) ** 2
    check(
        close(x.grad, want),
        f"d tanh/dx at 0.8814 is {x.grad:.6g}, expected {want:.6g}.",
    )

    random.seed(2)
    for _ in range(5):
        xs = [random.uniform(-2, 2) for _ in range(3)]
        grad_check(
            "tanh-in-graph", lambda a, b, c: (a * b + c).tanh() * a + (b * c).tanh(), xs
        )

    # the neuron shape from the lecture
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()
    check(close(o.data, 0.7071, 1e-3), f"neuron output {o.data}, expected ~0.7071")
    check(
        close(w1.grad, 1.0, 1e-3)
        and close(x2.grad, 0.5, 1e-3)
        and close(x1.grad, -1.5, 1e-3)
        and close(w2.grad, 0.0, 1e-3),
        f"neuron grads: x1={x1.grad:.4f} w1={w1.grad:.4f} "
        f"x2={x2.grad:.4f} w2={w2.grad:.4f}; expected -1.5, 1.0, 0.5, 0.0",
    )


# ----------------------------------------------------------------------------
# milestone 6 (stretch): exp, pow, and the ops built from them
# ----------------------------------------------------------------------------


def m6_more_ops():
    x = Value(1.3)
    y = x.exp()
    check(close(y.data, math.exp(1.3), 1e-9), f"exp(1.3).data is {y.data}")
    y.backward()
    check(close(x.grad, math.exp(1.3)), f"d exp/dx at 1.3 is {x.grad}")

    x = Value(2.0)
    y = x**3
    check(close(y.data, 8.0), f"2**3 .data is {y.data}")
    y.backward()
    check(close(x.grad, 12.0), f"d(x**3)/dx at 2 is {x.grad}, expected 12")

    x = Value(4.0)
    y = x**-0.5
    y.backward()
    check(
        close(x.grad, -0.5 * 4.0**-1.5),
        f"d(x**-0.5)/dx at 4 is {x.grad}, expected {-0.5 * 4.0 ** -1.5}",
    )

    # derived ops: forward values
    a, b = Value(5.0), Value(2.0)
    check(close((-a).data, -5.0), "-a")
    check(close((a - b).data, 3.0), "a - b")
    check(close((a - 1).data, 4.0), "a - 1")
    check(close((a / b).data, 2.5), "a / b")
    check(close((a / 4).data, 1.25), "a / 4")
    check(close((1 + a).data, 6.0), "1 + a  (float on the left)")
    check(close((3 * a).data, 15.0), "3 * a  (float on the left)")
    check(
        close((1 - a).data, -4.0),
        "1 - a should be -4, not 4. Subtraction does not commute.",
    )
    check(
        close((10 / a).data, 2.0),
        "10 / a should be 2, not 0.5. Division does not commute.",
    )
    for expr, name in [
        (-a, "-a"),
        (a - b, "a - b"),
        (a / b, "a / b"),
        (1 - a, "1 - a"),
        (10 / a, "10 / a"),
    ]:
        check(isinstance(expr, Value), f"{name} did not return a Value")

    # derived ops: gradients, numerically
    random.seed(3)
    for _ in range(5):
        xs = [random.uniform(0.5, 2.0) for _ in range(3)]
        grad_check("sub/neg", lambda a, b, c: (a - b) * c - (-a) * b + (1 - c), xs)
        grad_check("div", lambda a, b, c: a / b + c / 2 + 3 / (a + c), xs)
        grad_check("exp/pow", lambda a, b, c: (a * b).exp() / (c**2 + 1) + a**0.5, xs)
        grad_check("r-ops", lambda a, b, c: 2 * a + 1 + (3 - b) * (4 / c), xs)

    # a hand-rolled sigmoid
    z = Value(0.7)
    sig = 1 / (1 + (-z).exp())
    sig.backward()
    s = 1 / (1 + math.exp(-0.7))
    check(
        close(sig.data, s) and close(z.grad, s * (1 - s)),
        f"sigmoid via exp/div: value {sig.data}, grad {z.grad}; expected "
        f"{s}, {s*(1-s)}",
    )


# ----------------------------------------------------------------------------
# milestone 7 (stretch): neuron -> layer -> MLP, and it learns
# ----------------------------------------------------------------------------


def m7_mlp():
    random.seed(4)
    n = Neuron(3)
    ps = n.parameters()
    check(len(ps) == 4, f"Neuron(3) has {len(ps)} parameters, expected 4 (3 w + 1 b)")
    check(all(isinstance(p, Value) for p in ps), "parameters must be Values")
    check(all(-1 <= p.data <= 1 for p in ps), "init should be uniform in [-1, 1]")
    out = n([1.0, -2.0, 0.5])
    check(isinstance(out, Value), "Neuron.__call__ should return a single Value")
    check(-1 < out.data < 1, "neuron output should be squashed into (-1, 1)")
    out.backward()
    check(
        any(p.grad != 0 for p in ps),
        "after backward, no parameter got any gradient. Are the weights "
        "actually in the graph, or did you multiply their .data?",
    )

    layer = Layer(3, 4)
    check(
        len(layer.parameters()) == 16,
        f"Layer(3,4) has {len(layer.parameters())} params, expected 16",
    )
    outs = layer([1.0, -2.0, 0.5])
    check(
        isinstance(outs, list) and len(outs) == 4,
        "Layer(3,4) should return a list of 4 Values",
    )
    single = Layer(3, 1)([1.0, -2.0, 0.5])
    check(
        isinstance(single, Value),
        "Layer(_, 1) should return a bare Value, not a 1-element list",
    )

    m = MLP(3, [4, 4, 1])
    check(
        len(m.parameters()) == 41,
        f"MLP(3,[4,4,1]) has {len(m.parameters())} params, expected 41",
    )
    y = m([2.0, 3.0, -1.0])
    check(isinstance(y, Value), "MLP(...,[...,1]) should return a bare Value")
    y2 = m([-2.0, 0.5, 1.0])
    check(not close(y.data, y2.data), "MLP gives the same output for different inputs")

    # the lecture's toy dataset. training-loop boilerplate is given; the
    # thing under test is that the gradients your Value computes make it learn.
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    m = MLP(3, [4, 4, 1])

    def loss_fn():
        return sum((m(x) - y) ** 2 for x, y in zip(xs, ys))

    first = loss_fn().data
    for _ in range(60):
        loss = loss_fn()
        for p in m.parameters():
            p.grad = 0.0
        loss.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
    last = loss_fn().data
    check(
        last < first * 0.25,
        f"loss went {first:.4f} -> {last:.4f} over 60 steps. It should drop "
        "by well over 4x. Either gradients are wrong somewhere upstream or "
        "the parameters list is missing some Values.",
    )
    preds = [m(x).data for x in xs]
    check(
        all((p > 0) == (y > 0) for p, y in zip(preds, ys)),
        f"after training, predictions {[round(p, 2) for p in preds]} do not "
        f"match the signs of {ys}",
    )


# ----------------------------------------------------------------------------

MILESTONES = [
    (1, "forward pass + graph bookkeeping (__add__, __mul__)", m1_forward_and_graph),
    (2, "one hop of gradient (_backward closures)", m2_one_hop),
    (3, "full backward() in the right order", m3_backward),
    (4, "gradient accumulation on reused nodes", m4_accumulation),
    (5, "tanh", m5_tanh),
    (6, "STRETCH: exp, pow, neg, sub, div, r-ops", m6_more_ops),
    (7, "STRETCH: Neuron / Layer / MLP, and it learns", m7_mlp),
]


def main():
    upto = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    for num, title, fn in MILESTONES:
        if num > upto:
            break
        try:
            fn()
        except Fail as e:
            print(f"\n[FAIL] milestone {num}: {title}\n")
            print(f"    {e}\n")
            return 1
        except NotImplementedError:
            print(f"\n[TODO] milestone {num}: {title}")
            print("    hit a NotImplementedError -- this is the next thing to write.\n")
            return 1
        except Exception:
            print(f"\n[ERROR] milestone {num}: {title}\n")
            traceback.print_exc()
            return 1
        print(f"[ok]   milestone {num}: {title}")
    print("\nall milestones passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

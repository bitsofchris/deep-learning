"""
micrograd, from scratch.

OPTIONAL .py route. The main exercise is micrograd.ipynb; this file is the
same stubs if you would rather work in an editor.

Fill in every `raise NotImplementedError`. Run `python test_micrograd.py`
after each one. The grader stops at your first failure so there is always
exactly one thing in front of you.

Rules of engagement:
  - Don't open the lecture. Don't open the real micrograd repo.
  - Stuck on an IDEA for 20 min -> read one tier of HINTS.md, or ask.
  - Stuck on PYTHON SYNTAX -> ask immediately, zero learning value in that.
  - Before you run the tests, say out loud what you expect to happen.
"""

import math
import random


class Value:
    """A scalar that remembers the operation that produced it.

    Fields:
      data       float, the actual number
      grad       float, d(final output) / d(self). starts at zero.
      _prev      set of Values that were the inputs to the op producing self
      _op        str, debug label for that op
      _backward  a closure that takes self.grad and pushes gradient ONE HOP
                 back into each element of _prev. does nothing for a leaf.
    """

    # --- given to you: the container. the ideas are all below it. -----------
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def dump(self, indent=0):
        """Plumbing: print the expression graph as text. For debugging."""
        pad = "  " * indent
        label = self._op or "leaf"
        print(f"{pad}{label:>6} data={self.data:<12.6g} grad={self.grad:<12.6g}")
        for child in self._prev:
            child.dump(indent + 1)

    # --- MILESTONE 1 & 2 ----------------------------------------------------
    # Each op does two jobs: compute the forward number, and attach a closure
    # that knows the local derivative. Write both in the same method.

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        raise NotImplementedError

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        raise NotImplementedError

    # --- MILESTONE 3 & 4 ----------------------------------------------------

    def backward(self):
        """Run backprop from self through the entire graph behind it.

        Two questions to answer before you write a line:
          1. In what order must the nodes be visited, and why that order?
          2. What is self.grad, before any of this starts?
        """
        raise NotImplementedError

    # --- MILESTONE 5 --------------------------------------------------------

    def tanh(self):
        """Note you get to treat this as a single atomic op with one local
        derivative, even though it is really exp/div/sub underneath."""
        raise NotImplementedError

    # --- MILESTONE 6 (stretch) ---------------------------------------------
    # exp and pow need real _backward closures.
    # Everything after them does NOT -- each is a one-liner built out of ops
    # you already have. If you find yourself writing a closure for __sub__,
    # stop and think.

    def exp(self):
        raise NotImplementedError

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only scalar exponents"
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    # these four fire when the Value is on the RIGHT of the operator,
    # e.g. `2.0 * v` or `2.0 - v`. pure python trivia, ask me if annoying.
    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError


# --- MILESTONE 7 (stretch) --------------------------------------------------
# Nothing below here knows anything about calculus. That is the point: once
# Value works, a neural net is just arithmetic on top of it.


class Neuron:
    def __init__(self, nin):
        """nin weights and one bias. init weights uniform in [-1, 1]."""
        raise NotImplementedError

    def __call__(self, x):
        """x is a list of nin floats or Values. Return one Value: the
        squashed weighted sum."""
        raise NotImplementedError

    def parameters(self):
        """Flat list of every Value the optimizer is allowed to nudge."""
        raise NotImplementedError


class Layer:
    def __init__(self, nin, nout):
        raise NotImplementedError

    def __call__(self, x):
        """Return a list of Values -- or a bare Value if nout == 1."""
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class MLP:
    def __init__(self, nin, nouts):
        """nouts is a list of layer widths, e.g. MLP(3, [4, 4, 1])."""
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

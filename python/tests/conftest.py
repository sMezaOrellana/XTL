import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import xtl


def flat(t):
    if isinstance(t[0], list):
        return [x for row in t for x in flat(row)]
    return t


def allclose(a, b, tol=1e-4):
    fa = flat(a) if isinstance(a[0], list) else a
    fb = flat(b) if isinstance(b[0], list) else b
    return all(abs(x - y) < tol for x, y in zip(fa, fb))

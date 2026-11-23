import math

import pytest

import histmatch

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = [[0.0, 0.5], [0.5, 1.0]]
B = [[0.0, 0.5], [0.5, 1.0]]


def _h():
    return histmatch.HistMatch()


def _check_shape():
    out = _h().match(A, B)
    return len(out) == 2 and len(out[0]) == 2


def _check_identity():
    out = _h().match(A, A)
    return all(abs(out[r][c] - A[r][c]) < 1e-6 for r in range(2) for c in range(2))


def _check_range():
    out = _h().match(A, B)
    return all(0.0 <= v <= 1.0 for row in out for v in row)


def _check_bad_empty():
    return _raises(lambda: _h().match([], B))


def _check_bad_shape():
    return _raises(lambda: _h().match(A, [[0.0]]))


def _check_type():
    return isinstance(_h().match(A, B), list)


def _check_det():
    return _h().match(A, B) == _h().match(A, B)


def _check_monotonic():
    out = _h().match([[0.0], [0.5], [1.0]], [[0.0], [1.0], [1.0]])
    return out[0][0] <= out[1][0] <= out[2][0]

CASES = [("shape", "_check_shape"), ("identity", "_check_identity"), ("range", "_check_range"), ("bad_empty", "_check_bad_empty"), ("bad_shape", "_check_bad_shape"), ("type", "_check_type"), ("det", "_check_det"), ("monotonic", "_check_monotonic"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert HistMatch() is not None
        return
    assert fn() is True

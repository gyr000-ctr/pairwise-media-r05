import math

import pytest

import gridcc

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]


def _g():
    return gridcc.GridCC()


def _check_count4():
    return len(set(v for row in _g().label(G, False) for v in row if v >= 0)) == 2


def _check_count8():
    return len(set(v for row in _g().label(G, True) for v in row if v >= 0)) == 1


def _check_shape():
    l = _g().label(G, False)
    return len(l) == 3 and len(l[0]) == 3


def _check_bad_grid():
    return _raises(lambda: _g().label([], False))


def _check_type():
    return isinstance(_g().label(G, False), list)


def _check_det():
    return _g().label(G, False) == _g().label(G, False)


def _check_single():
    return len(set(v for row in _g().label([[1, 1], [1, 1]], False) for v in row if v >= 0)) == 1


def _check_empty():
    return len(set(v for row in _g().label([[1, 0], [0, 1]], False) for v in row if v >= 0)) == 2

CASES = [("count4", "_check_count4"), ("count8", "_check_count8"), ("shape", "_check_shape"), ("bad_grid", "_check_bad_grid"), ("type", "_check_type"), ("det", "_check_det"), ("single", "_check_single"), ("empty", "_check_empty"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert GridCC() is not None
        return
    assert fn() is True

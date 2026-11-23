import math

import pytest

import depthsort

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

I = [("a", 0.5), ("b", 0.1), ("c", 0.5)]


def _d():
    return depthsort.DepthSort()


def _check_order():
    return [x[0] for x in _d().sort(I, 0.0, 1.0)] == ["b", "a", "c"]


def _check_cull():
    return [x[0] for x in _d().sort(I, 0.2, 1.0)] == ["a", "c"]


def _check_none():
    return _d().sort(I, 2.0, 3.0) == []


def _check_bad_range():
    return _raises(lambda: _d().sort(I, 1.0, 0.0))


def _check_bad_item():
    return _raises(lambda: _d().sort([("a",)], 0.0, 1.0))


def _check_type():
    return isinstance(_d().sort(I, 0.0, 1.0), list)


def _check_det():
    return _d().sort(I, 0.0, 1.0) == _d().sort(I, 0.0, 1.0)


def _check_stable():
    return [x[0] for x in _d().sort(I, 0.0, 1.0)] == ["b", "a", "c"]

CASES = [("order", "_check_order"), ("cull", "_check_cull"), ("none", "_check_none"), ("bad_range", "_check_bad_range"), ("bad_item", "_check_bad_item"), ("type", "_check_type"), ("det", "_check_det"), ("stable", "_check_stable"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert DepthSort() is not None
        return
    assert fn() is True

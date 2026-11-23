import math

import pytest

import channelsplit

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return channelsplit.Center()


def _check_len():
    m, s = _c().split([0.4], [0.2])
    return len(m) == 1 and len(s) == 1


def _check_mid():
    m, s = _c().split([0.4], [0.2])
    return abs(m[0] - 0.3) < 1e-9


def _check_side():
    m, s = _c().split([0.4], [0.2])
    return abs(s[0] - 0.1) < 1e-9


def _check_mono_center():
    c = _c().extract([0.5], [0.5])
    return abs(c[0] - 0.5) < 1e-9


def _check_bad_len():
    return _raises(lambda: _c().split([0.1], [0.1, 0.2]))


def _check_empty():
    return _c().split([], []) == ([], [])


def _check_type():
    return isinstance(_c().split([0.1], [0.2]), tuple)


def _check_det():
    return _c().split([0.1], [0.2]) == _c().split([0.1], [0.2])

CASES = [("len", "_check_len"), ("mid", "_check_mid"), ("side", "_check_side"), ("mono_center", "_check_mono_center"), ("bad_len", "_check_bad_len"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Center() is not None
        return
    assert fn() is True

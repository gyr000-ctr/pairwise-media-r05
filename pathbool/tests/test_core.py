import math

import pytest

import pathbool

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [((0.0, 0.0), (1.0, 0.0)), ((0.5, 0.0), (1.5, 0.0))]


def _p():
    return pathbool.PathBool()


def _check_single():
    return abs(_p().union_length([((0.0, 0.0), (1.0, 0.0))]) - 1.0) < 1e-9


def _check_overlap():
    return abs(_p().union_length(S) - 1.5) < 1e-9


def _check_disjoint():
    return abs(_p().union_length([((0.0, 0.0), (1.0, 0.0)), ((2.0, 0.0), (3.0, 0.0))]) - 2.0) < 1e-9


def _check_touch():
    return abs(_p().union_length([((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (2.0, 0.0))]) - 2.0) < 1e-9


def _check_empty():
    return _p().union_length([]) == 0.0


def _check_bad_seg():
    return _raises(lambda: _p().union_length([((0.0, 0.0),)]))


def _check_type():
    return isinstance(_p().union_length(S), float)


def _check_det():
    return _p().union_length(S) == _p().union_length(S)

CASES = [("single", "_check_single"), ("overlap", "_check_overlap"), ("disjoint", "_check_disjoint"), ("touch", "_check_touch"), ("empty", "_check_empty"), ("bad_seg", "_check_bad_seg"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert PathBool() is not None
        return
    assert fn() is True

"""depthsort 扩展验收测试（不改动既有 tests/test_core.py）。

覆盖题面隐藏要点：
- 与内置稳定排序 sorted(..., key=...) 逐元素一致（同深度插入序）
- 闭区间边界归属确定
- NaN / inf / 缺失深度 -> DepthError，区间错误 -> DepthRangeError，可区分
- 自定义比较键（构造器级与单次调用级）
- 10 万项 <= 100 ms
- 多线程并发排序安全、结果确定
"""

import math
import random
import threading

import pytest

import depthsort
from depthsort import DepthError, DepthRangeError, DepthSort


def _ref(items, near, far, key=lambda x: x[1]):
    """真值参照：过滤后走内置稳定排序。"""
    kept = [x for x in items if near <= key(x) <= far]
    return sorted(kept, key=key)


# --------------------------------------------------------------------------- #
# 稳定性：与 sorted 稳定排序完全一致
# --------------------------------------------------------------------------- #
def test_matches_builtin_stable_sort_on_ties():
    items = [(f"id{i:04d}", round(i % 7, 1)) for i in range(500)]
    got = DepthSort().sort(items, 0.0, 6.0)
    assert got == _ref(items, 0.0, 6.0)


def test_equal_depths_keep_insertion_order():
    items = [("z", 1.0), ("a", 1.0), ("m", 1.0), ("b", 2.0), ("k", 2.0)]
    assert [x[0] for x in DepthSort().sort(items, 0.0, 9.0)] == [
        "z", "a", "m", "b", "k",
    ]


def test_does_not_mutate_input():
    items = [("a", 0.5), ("b", 0.1)]
    snapshot = list(items)
    DepthSort().sort(items, 0.0, 1.0)
    assert items == snapshot


def test_accepts_generators_and_returns_list():
    gen = ((c, d) for c, d in [("a", 0.9), ("b", 0.1)])
    out = DepthSort().sort(gen, 0.0, 1.0)
    assert isinstance(out, list)
    assert [x[0] for x in out] == ["b", "a"]


def test_deterministic_across_instances():
    items = [(str(i), (i * 37) % 13 / 13) for i in range(300)]
    a = DepthSort().sort(items, 0.1, 0.9)
    b = DepthSort().sort(items, 0.1, 0.9)
    assert a == b


# --------------------------------------------------------------------------- #
# 闭区间剔除与边界归属
# --------------------------------------------------------------------------- #
def test_closed_interval_includes_both_bounds():
    items = [("lo", 0.2), ("mid", 0.5), ("hi", 1.0), ("out", 1.0001)]
    got = DepthSort().sort(items, 0.2, 1.0)
    assert [x[0] for x in got] == ["lo", "mid", "hi"]


def test_cull_outside_range():
    items = [("a", -0.1), ("b", 0.0), ("c", 1.0), ("d", 1.1)]
    got = DepthSort().sort(items, 0.0, 1.0)
    assert [x[0] for x in got] == ["b", "c"]


def test_empty_when_nothing_in_range():
    assert DepthSort().sort([("a", 5.0)], 0.0, 1.0) == []


def test_empty_input():
    assert DepthSort().sort([], 0.0, 1.0) == []


def test_negative_and_integer_depths():
    items = [("a", -3), ("b", -3.0), ("c", 0), ("d", 2)]
    got = DepthSort().sort(items, -3, 2)
    assert [x[0] for x in got] == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------- #
# 非法深度分类：DepthError vs DepthRangeError，均为 ValueError
# --------------------------------------------------------------------------- #
def test_nan_depth_raises_depth_error():
    with pytest.raises(DepthError):
        DepthSort().sort([("a", float("nan"))], 0.0, 1.0)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_infinite_depth_raises_depth_error(bad):
    with pytest.raises(DepthError):
        DepthSort().sort([("a", bad)], 0.0, 1.0)


def test_nan_and_inf_are_distinguishable_from_range_error():
    with pytest.raises(DepthError):
        DepthSort().sort([("a", float("nan"))], 0.0, 1.0)
    with pytest.raises(DepthRangeError):
        DepthSort().sort([("a", 0.5)], 1.0, 0.0)
    # 两类异常各自都是 ValueError，兼容旧契约
    assert issubclass(DepthError, ValueError)
    assert issubclass(DepthRangeError, ValueError)
    assert DepthError is not DepthRangeError


def test_missing_depth_field_raises_depth_error():
    with pytest.raises(DepthError):
        DepthSort().sort([("only-one",)], 0.0, 1.0)


def test_non_numeric_depth_raises_depth_error():
    with pytest.raises(DepthError):
        DepthSort().sort([("a", "0.5")], 0.0, 1.0)


@pytest.mark.parametrize("near,far", [(float("nan"), 1.0), (0.0, float("inf"))])
def test_non_finite_bounds_raise_range_error(near, far):
    with pytest.raises(DepthRangeError):
        DepthSort().sort([("a", 0.5)], near, far)


def test_inverted_range_raises_range_error():
    with pytest.raises(DepthRangeError):
        DepthSort().sort([("a", 0.5)], 1.0, 0.0)


# --------------------------------------------------------------------------- #
# 自定义比较键
# --------------------------------------------------------------------------- #
def test_custom_key_constructor_level():
    items = [{"id": "a", "z": 0.9}, {"id": "b", "z": 0.1}]
    got = DepthSort(key=lambda o: o["z"]).sort(items, 0.0, 1.0)
    assert [o["id"] for o in got] == ["b", "a"]


def test_custom_key_per_call_overrides_constructor():
    items = [{"id": "a", "z": 0.9, "w": 0.2}, {"id": "b", "z": 0.1, "w": 0.8}]
    ds = DepthSort(key=lambda o: o["z"])
    got = ds.sort(items, 0.0, 1.0, key=lambda o: o["w"])
    assert [o["id"] for o in got] == ["a", "b"]
    # 构造器键不受单次调用影响
    got2 = ds.sort(items, 0.0, 1.0)
    assert [o["id"] for o in got2] == ["b", "a"]


def test_custom_key_nan_classified():
    with pytest.raises(DepthError):
        DepthSort(key=lambda o: o[1]).sort([("a", float("nan"))], 0.0, 1.0)


def test_custom_key_exception_classified_as_depth_error():
    def boom(_item):
        raise RuntimeError("key exploded")

    with pytest.raises(DepthError):
        DepthSort(key=boom).sort([("a", 0.5)], 0.0, 1.0)


def test_non_comparable_items_sort_fine_with_depth_key():
    # 元素本身不可比较（dict 间 < 会报错），深度键排序不得触碰元素比较
    items = [{"v": i, "d": i % 5} for i in range(50)]
    got = DepthSort(key=lambda o: o["d"]).sort(items, 0.0, 4.0)
    assert got == sorted(items, key=lambda o: o["d"])


# --------------------------------------------------------------------------- #
# 性能：10 万项 <= 100 ms
# --------------------------------------------------------------------------- #
def test_perf_100k_under_100ms():
    rng = random.Random(42)
    items = [(i, rng.random() * 100.0) for i in range(100_000)]
    import time

    t0 = time.perf_counter()
    out = DepthSort().sort(items, 10.0, 90.0)
    elapsed = time.perf_counter() - t0
    assert elapsed <= 0.1, f"耗时 {elapsed * 1000:.1f} ms 超过 100 ms"
    assert out == _ref(items, 10.0, 90.0)


# --------------------------------------------------------------------------- #
# 并发安全：无共享可变状态，多线程结果一致且等于单线程参照
# --------------------------------------------------------------------------- #
def test_concurrent_sort_is_safe():
    rng = random.Random(7)
    items = [(i, rng.random()) for i in range(20_000)]
    expected = _ref(items, 0.1, 0.9)
    shared = DepthSort()
    errors = []
    results = []
    lock = threading.Lock()

    def worker():
        try:
            local = shared.sort(items, 0.1, 0.9)
            with lock:
                results.append(local)
        except Exception as exc:  # pragma: no cover - 并发下不应出现
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 8
    for r in results:
        assert r == expected


def test_concurrent_with_distinct_keys_same_instance():
    items = [{"id": i, "a": i % 11 / 11, "b": (100 - i) % 13 / 13} for i in range(5_000)]
    ds = DepthSort()
    outcomes = {}
    barrier = threading.Barrier(2)

    def worker(name, k):
        barrier.wait()
        outcomes[name] = ds.sort(items, 0.0, 1.0, key=k)

    t1 = threading.Thread(target=worker, args=("a", lambda o: o["a"]))
    t2 = threading.Thread(target=worker, args=("b", lambda o: o["b"]))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert outcomes["a"] == sorted(items, key=lambda o: o["a"])
    assert outcomes["b"] == sorted(items, key=lambda o: o["b"])

"""直方图规定化验收测试（新增，不改动既有 test_core.py）。

覆盖题面约束：
1. 规定化后 CDF 与目标 CDF 差 <= 0.02
2. 查找表与逐像素解析结果一致 <= 1e-9
3. 映射单调、同分布时恒等 <= 1e-6
4. 空图与形状不一致可区分报错（非有限输入另有独立类型）
5. 100 万像素 <= 500 ms
6. 并发安全
"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import histmatch
from histmatch import (
    EmptyImageError,
    HistMatch,
    LookupTableError,
    NonFiniteError,
    PixelTypeError,
    ShapeMismatchError,
)
from histmatch.core import _np


from bisect import bisect_right


def _cdf(values):
    """经验 CDF：对阈值 t 返回 #(<= t)/n。"""
    s = sorted(values)
    n = len(s)

    def cdf(t):
        return bisect_right(s, t) / n

    return cdf


def _quantile(values, p):
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s)) - 1)]


@pytest.fixture
def hm():
    return HistMatch()


# ---------------------------------------------------------------------------
# 1. 映射后 CDF 与目标 CDF 差 <= 0.02
# ---------------------------------------------------------------------------


def test_cdf_matches_target(hm):
    rng = random.Random(7)
    src = [[rng.random() for _ in range(100)] for _ in range(100)]
    tgt = [[rng.betavariate(2.0, 5.0) for _ in range(100)] for _ in range(100)]
    out = hm.match(src, tgt)

    flat_src = [v for row in src for v in row]
    flat_tgt = [v for row in tgt for v in row]
    flat_out = [v for row in out for v in row]
    assert len(flat_out) == len(flat_src)

    cdf_out = _cdf(flat_out)
    worst = 0.0
    for k in range(1, 100):
        p = k / 100.0
        threshold = _quantile(flat_tgt, p)
        worst = max(worst, abs(cdf_out(threshold) - p))
    assert worst <= 0.02, f"规定化后 CDF 与目标最大偏差 {worst:.6f}"


def test_cdf_matches_target_disjoint_support(hm):
    """源与目标支撑几乎不重叠时仍须满足 CDF 约束。"""
    rng = random.Random(11)
    src = [[rng.random() for _ in range(80)] for _ in range(80)]
    tgt = [[0.45 + 0.1 * rng.random() for _ in range(80)] for _ in range(80)]
    out = hm.match(src, tgt)
    flat_out = [v for row in out for v in row]
    flat_tgt = [v for row in tgt for v in row]
    cdf_out = _cdf(flat_out)
    worst = max(
        abs(cdf_out(_quantile(flat_tgt, k / 100.0)) - k / 100.0)
        for k in range(1, 100)
    )
    assert worst <= 0.02
    assert min(flat_out) >= min(flat_tgt) and max(flat_out) <= max(flat_tgt)


# ---------------------------------------------------------------------------
# 2. 查找表与逐像素解析结果一致 <= 1e-9
# ---------------------------------------------------------------------------


def test_lookup_table_matches_analytic(hm):
    rng = random.Random(3)
    src = [[rng.random() for _ in range(64)] for _ in range(48)]
    tgt = [[rng.gauss(0.5, 0.2) for _ in range(64)] for _ in range(48)]
    lut = hm.lookup_table(src, tgt)
    assert isinstance(lut, dict) and lut

    by_lut = hm.apply_lut(src, lut)
    by_parse = hm.match(src, tgt)
    assert by_lut == by_parse

    worst = 0.0
    for y, row in enumerate(src):
        for x, v in enumerate(row):
            analytic = hm.map_value(v, src, tgt)
            worst = max(worst, abs(by_lut[y][x] - analytic))
            worst = max(worst, abs(by_parse[y][x] - analytic))
    assert worst <= 1e-9, f"查表与解析结果最大偏差 {worst:.3e}"


def test_lookup_table_matches_match_for_flat_input(hm):
    src = [i / 100.0 for i in range(101)]
    tgt = [min(1.0, i / 50.0) for i in range(101)]
    lut = hm.lookup_table(src, tgt)
    assert hm.apply_lut(src, lut) == hm.match(src, tgt)
    assert hm.map_value(0.5, src, tgt) == lut[0.5]


def test_lookup_table_is_readonly_shared(hm):
    """查找表可跨实例复用，结果为只读语义（重复使用不改变输出）。"""
    src = [[0.0, 0.5], [0.5, 1.0]]
    tgt = [[0.2, 0.4], [0.6, 0.8]]
    lut = hm.lookup_table(src, tgt)
    first = hm.apply_lut(src, lut)
    other = HistMatch()
    assert other.apply_lut(src, lut) == first == hm.match(src, tgt)


def test_lookup_table_off_grid_values(hm):
    """查找表对任意有限值给出解析式结果（与逐像素解析一致）。"""
    img = [[0.0, 1.0]]
    tgt = [[0.0, 1.0]]
    lut = hm.lookup_table(img, tgt)
    # 0.5 不在源灰度级里：LookupTable 按解析式回退，结果与 map_value 一致。
    out = hm.apply_lut([[0.0, 0.5]], lut)
    analytic = hm.map_value(0.5, img, tgt)
    assert out == [[0.0, analytic]]
    assert lut(0.5) == analytic
    assert lut[0.0] == 0.0 and lut[1.0] == 1.0


def test_plain_dict_missing_level_raises(hm):
    """直接传入不覆盖灰度级的普通 dict 时，缺失键显式报错。"""
    lut = {"0.0": 0.0}  # 非数值键，无法查询 0.5
    with pytest.raises(LookupTableError):
        hm.apply_lut([[0.0, 0.5]], lut)
    plain = {0.0: 0.0, 1.0: 1.0}
    with pytest.raises(LookupTableError):
        hm.apply_lut([[0.0, 0.5]], plain)


# ---------------------------------------------------------------------------
# 3. 单调与同分布恒等
# ---------------------------------------------------------------------------


def test_mapping_monotonic(hm):
    src = [[0.0], [0.1], [0.5], [0.9], [1.0]]
    tgt = [[0.3], [0.3], [0.6], [1.0], [1.0]]
    out = hm.match(src, tgt)
    flat = [row[0] for row in out]
    assert all(flat[i] <= flat[i + 1] for i in range(len(flat) - 1))


def test_mapping_monotonic_random(hm):
    rng = random.Random(5)
    src_flat = [rng.random() for _ in range(400)]
    tgt_flat = [rng.betavariate(5.0, 2.0) for _ in range(400)]
    lut = hm.lookup_table(src_flat, tgt_flat)
    keys = sorted(lut)
    values = [lut[k] for k in keys]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_identity_same_distribution(hm):
    rng = random.Random(9)
    img = [[rng.random() for _ in range(50)] for _ in range(40)]
    out = hm.match(img, img)
    worst = max(
        abs(out[y][x] - img[y][x])
        for y in range(len(img))
        for x in range(len(img[0]))
    )
    assert worst <= 1e-6
    assert out == img


def test_identity_permuted_same_distribution(hm):
    """目标只是同一多重集的乱序/异形排列时，映射仍为恒等。"""
    rng = random.Random(13)
    flat = [rng.random() for _ in range(200)]
    img = [flat[:100], flat[100:]]
    shuffled = flat[:]
    rng.shuffle(shuffled)
    tgt = [shuffled[:100], shuffled[100:]]
    out = hm.match(img, tgt)
    assert out == img


# ---------------------------------------------------------------------------
# 4. 可区分的错误
# ---------------------------------------------------------------------------


def test_empty_vs_shape_errors_are_distinguishable(hm):
    good = [[0.0, 0.5], [0.5, 1.0]]
    with pytest.raises(EmptyImageError):
        hm.match([], good)
    with pytest.raises(EmptyImageError):
        hm.match(good, [])
    with pytest.raises(ShapeMismatchError):
        hm.match(good, [[0.0]])
    assert EmptyImageError is not ShapeMismatchError
    assert issubclass(EmptyImageError, ValueError)
    assert issubclass(ShapeMismatchError, ValueError)


def test_ragged_and_non_finite_and_type_errors(hm):
    good = [[0.0, 0.5], [0.5, 1.0]]
    with pytest.raises(ShapeMismatchError):
        hm.match([[0.0, 0.5], [0.5]], good)
    with pytest.raises(NonFiniteError):
        hm.match([[0.0, float("nan")], [0.5, 1.0]], good)
    with pytest.raises(NonFiniteError):
        hm.match(good, [[0.0, float("inf")], [0.5, 1.0]])
    with pytest.raises(PixelTypeError):
        hm.match([[0.0, "x"], [0.5, 1.0]], good)
    assert len({EmptyImageError, ShapeMismatchError, NonFiniteError, PixelTypeError}) == 4


# ---------------------------------------------------------------------------
# 5. 性能：100 万像素 <= 500 ms
# ---------------------------------------------------------------------------


def test_performance_1m_pixels(hm):
    """100 万像素须在 500 ms 内完成（numpy 加速路径）。

    没有 numpy 时只有纯 Python 回退（结果相同但明显更慢），性能上界
    在题面所指的运行环境里以加速路径为准，故此处跳过。
    """
    if _np is None:
        pytest.skip("numpy 不可用：纯 Python 回退路径不承诺该性能上界")
    rng = random.Random(21)
    n = 1000
    src = [[rng.random() for _ in range(n)] for _ in range(n)]
    tgt = [[rng.betavariate(2.0, 5.0) for _ in range(n)] for _ in range(n)]
    start = time.perf_counter()
    out = hm.match(src, tgt)
    elapsed = time.perf_counter() - start
    assert len(out) == n and len(out[0]) == n
    assert elapsed <= 0.5, f"100 万像素耗时 {elapsed * 1000:.1f} ms"

# ---------------------------------------------------------------------------
# 6. 并发安全
# ---------------------------------------------------------------------------


def test_concurrent_match_and_lookup(hm):
    rng = random.Random(33)
    src = [[rng.random() for _ in range(40)] for _ in range(30)]
    tgt = [[rng.gauss(0.5, 0.2) for _ in range(40)] for _ in range(30)]
    lut = hm.lookup_table(src, tgt)
    expected_match = hm.match(src, tgt)
    expected_lut = hm.apply_lut(src, lut)

    def work(i):
        if i % 3 == 0:
            return hm.match(src, tgt)
        if i % 3 == 1:
            return hm.apply_lut(src, lut)
        return HistMatch().lookup_table(src, tgt)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, range(96)))

    for i, res in enumerate(results):
        if i % 3 == 0:
            assert res == expected_match
        elif i % 3 == 1:
            assert res == expected_lut
        else:
            assert res == lut


# ---------------------------------------------------------------------------
# 7. numpy / 纯 Python 两条路径结果一致
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_np is None, reason="numpy 不可用")
def test_numpy_and_python_paths_agree(hm):
    from histmatch import core

    rng = random.Random(41)
    src_flat = [rng.random() for _ in range(300)]
    tgt_flat = [rng.betavariate(3.0, 2.0) for _ in range(300)]

    np_table = core._table_numpy(
        core._np.asarray(src_flat), core._np.asarray(tgt_flat)
    )
    py_table = core._table_python(src_flat, tgt_flat)
    np_dict = dict(zip(np_table[0].tolist(), np_table[1].tolist()))
    assert np_dict == py_table
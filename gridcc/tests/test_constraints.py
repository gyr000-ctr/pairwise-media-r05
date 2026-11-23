"""gridcc 约束级测试：邻接语义、标签稳定性、面积、边界、报错、性能、并发。

参考实现独立于 core.py：另写一份教科书两遍扫描（等价表即并查集），
临时标签稠密分配、等价关系单向指向较小标签、末遍仅解析一层，
用于与生产实现逐像素对照。另用完整并查集与 BFS 校验语义不变量。
"""

import math
import random
import threading
import time
from collections import Counter, deque

import pytest

import gridcc
from gridcc import GridCC, GridShapeError, PixelValueError, BACKGROUND


# --------------------------------------------------------------------- #
# 独立参考实现 A：教科书两遍扫描（等价表 = 并查集，末遍解析一层）
# 与 core.py 数据结构、控制流均不同，仅语义对齐。
# --------------------------------------------------------------------- #
def _reference_two_pass(grid, diagonal):
    h, w = len(grid), len(grid[0])
    provisional = [[0] * w for _ in range(h)]
    equivalence = {}          # 临时标签 -> 指向的较小临时标签
    fresh = 1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            seen = []
            if r > 0 and grid[r - 1][c]:
                seen.append(provisional[r - 1][c])
            if c > 0 and grid[r][c - 1]:
                seen.append(provisional[r][c - 1])
            if diagonal:
                if r > 0 and c > 0 and grid[r - 1][c - 1]:
                    seen.append(provisional[r - 1][c - 1])
                if r > 0 and c + 1 < w and grid[r - 1][c + 1]:
                    seen.append(provisional[r - 1][c + 1])
            seen = [t for t in seen if t]
            if not seen:
                provisional[r][c] = fresh
                equivalence[fresh] = fresh
                fresh += 1
            else:
                low = min(seen)
                provisional[r][c] = low
                for t in set(seen):
                    if t != low:
                        equivalence[t] = low
    resolved = [[BACKGROUND] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            t = provisional[r][c]
            if t:
                resolved[r][c] = equivalence[t]
    return resolved


# --------------------------------------------------------------------- #
# 独立参考实现 B：完整并查集（路径压缩 + 最小根），给出"真"连通划分
# --------------------------------------------------------------------- #
def _reference_full_union_find(grid, diagonal):
    h, w = len(grid), len(grid[0])
    n = h * w
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            i = r * w + c
            if r > 0 and grid[r - 1][c]:
                union(i, (r - 1) * w + c)
            if c > 0 and grid[r][c - 1]:
                union(i, r * w + c - 1)
            if diagonal:
                if r > 0 and c > 0 and grid[r - 1][c - 1]:
                    union(i, (r - 1) * w + c - 1)
                if r > 0 and c + 1 < w and grid[r - 1][c + 1]:
                    union(i, (r - 1) * w + c + 1)
    return [[find(r * w + c) if grid[r][c] else BACKGROUND for c in range(w)]
            for r in range(h)]


def _bfs_component(grid, start, diagonal):
    h, w = len(grid), len(grid[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        if diagonal:
            candidates += [(r - 1, c - 1), (r - 1, c + 1),
                           (r + 1, c - 1), (r + 1, c + 1)]
        for nr, nc in candidates:
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


# --------------------------------------------------------------------- #
# 邻接语义：与两遍扫描参考逐像素一致；与完整并查集保持"细分"关系
# --------------------------------------------------------------------- #
def test_4_neighbor_fixture_two_labels():
    g = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    out = GridCC().label(g, False)
    assert out == [[-1, -1, 1], [-1, 1, 1], [2, 1, 1]]
    assert len({t for row in out for t in row if t >= 0}) == 2


def test_8_neighbor_fixture_one_label():
    g = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    out = GridCC().label(g, True)
    assert out == [[-1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    assert len({t for row in out for t in row if t >= 0}) == 1


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("diagonal", [False, True])
def test_random_grids_pixel_match_two_pass(seed, diagonal):
    rng = random.Random(seed)
    h, w = rng.randint(1, 14), rng.randint(1, 14)
    grid = [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
    out = GridCC().label(grid, diagonal)
    assert out == _reference_two_pass(grid, diagonal)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("diagonal", [False, True])
def test_output_is_a_refinement_of_full_union_find(seed, diagonal):
    """两遍扫描的每个标签必落在完整并查集的同一真分量内（无错误合并），
    且每个真分量至少被一个标签覆盖（无遗漏）。"""
    rng = random.Random(1000 + seed)
    h, w = rng.randint(1, 12), rng.randint(1, 12)
    grid = [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
    out = GridCC().label(grid, diagonal)
    full = _reference_full_union_find(grid, diagonal)

    label_to_root = {}
    covered_roots = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                assert out[r][c] == BACKGROUND
                continue
            tag, root = out[r][c], full[r][c]
            covered_roots.add(root)
            if tag in label_to_root:
                assert label_to_root[tag] == root
            else:
                label_to_root[tag] = root
    # 覆盖全部真分量
    true_roots = {full[r][c] for r in range(h) for c in range(w) if grid[r][c]}
    assert covered_roots == true_roots


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("diagonal", [False, True])
def test_each_label_is_truly_connected_by_bfs(seed, diagonal):
    """同标签的所有格在对应邻接下确实互相可达（绝不会把不相邻格并为一域）。"""
    rng = random.Random(2000 + seed)
    h, w = rng.randint(1, 10), rng.randint(1, 10)
    grid = [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
    out = GridCC().label(grid, diagonal)
    cells_by_tag = {}
    for r in range(h):
        for c in range(w):
            if out[r][c] != BACKGROUND:
                cells_by_tag.setdefault(out[r][c], []).append((r, c))
    for tag, cells in cells_by_tag.items():
        component = _bfs_component(grid, cells[0], diagonal)
        assert set(cells) <= component


def test_checkerboard_4_vs_8():
    grid = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    assert GridCC().count(grid, False) == 5
    assert GridCC().count(grid, True) == 1
    # 8 邻接分量数永不超过 4 邻接
    rng = random.Random(3)
    for _ in range(30):
        g = [[rng.randint(0, 1) for _ in range(8)] for _ in range(8)]
        assert GridCC().count(g, True) <= GridCC().count(g, False)


# --------------------------------------------------------------------- #
# 标签稳定性：确定性、最小标签、背景约定
# --------------------------------------------------------------------- #
def test_label_deterministic_across_calls_and_instances():
    grid = [[1, 0, 1], [1, 1, 0], [0, 0, 1]]
    cc = GridCC()
    first = cc.label(grid, False)
    for _ in range(5):
        assert cc.label(grid, False) == first
    assert GridCC().label(grid, False) == first


def test_minimum_provisional_label_rule():
    # 行主序下三个彼此分离的前景依次取得标签 1、2、3
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 0]]
    assert GridCC().label(grid, False) == [[1, -1, 2], [-1, -1, -1], [3, -1, -1]]


def test_background_is_negative_one():
    out = GridCC().label([[0, 1], [0, 0]], False)
    assert out == [[-1, 1], [-1, -1]]
    assert BACKGROUND == -1


# --------------------------------------------------------------------- #
# 数量与面积统计
# --------------------------------------------------------------------- #
def test_count_and_areas_are_self_consistent():
    rng = random.Random(42)
    for _ in range(50):
        h, w = rng.randint(1, 10), rng.randint(1, 10)
        grid = [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
        diagonal = rng.random() < 0.5
        cc = GridCC()
        out, areas = cc.label_with_areas(grid, diagonal)
        # 面积与标注网格逐格统计完全一致
        assert areas == dict(Counter(t for row in out for t in row if t != BACKGROUND))
        assert cc.count(grid, diagonal) == len(areas)
        # 面积之和 == 前景格总数
        assert sum(areas.values()) == sum(sum(row) for row in grid)
        # 与独立参考的面积多重集合一致（同一划分）
        ref = _reference_two_pass(grid, diagonal)
        ref_areas = Counter(t for row in ref for t in row if t != BACKGROUND)
        assert sorted(areas.values()) == sorted(ref_areas.values())


def test_areas_label_keys_match_output():
    grid = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    out, areas = GridCC().label_with_areas(grid, False)
    assert set(areas) == {t for row in out for t in row if t != BACKGROUND}
    assert areas == {1: 3, 2: 1}


def test_all_background_zero_components():
    grid = [[0, 0], [0, 0]]
    cc = GridCC()
    assert cc.count(grid, False) == 0
    assert cc.areas(grid, False) == {}
    assert cc.label(grid, False) == [[-1, -1], [-1, -1]]


def test_all_foreground_single_component():
    grid = [[1, 1, 1], [1, 1, 1]]
    for diagonal in (False, True):
        cc = GridCC()
        assert cc.count(grid, diagonal) == 1
        assert cc.areas(grid, diagonal) == {1: 6}
        assert cc.label(grid, diagonal) == [[1, 1, 1], [1, 1, 1]]


# --------------------------------------------------------------------- #
# 边界：空图、单行、单列、单像素
# --------------------------------------------------------------------- #
def test_empty_grid_raises_shape_error():
    with pytest.raises(GridShapeError):
        GridCC().label([], False)


def test_empty_row_raises_shape_error():
    with pytest.raises(GridShapeError):
        GridCC().label([[]], False)
    with pytest.raises(GridShapeError):
        GridCC().label([[1], []], False)


def test_ragged_rows_raise_shape_error():
    with pytest.raises(GridShapeError):
        GridCC().label([[1, 0], [1]], False)


def test_single_row_segmentation():
    grid = [[1, 1, 0, 1, 0, 1, 1, 1]]
    expected = [[1, 1, -1, 2, -1, 3, 3, 3]]
    assert GridCC().label(grid, False) == expected
    assert GridCC().label(grid, True) == expected
    assert GridCC().areas(grid, False) == {1: 2, 2: 1, 3: 3}


def test_single_column_segmentation():
    assert GridCC().label([[1], [0], [1], [1]], False) == [[1], [-1], [2], [2]]


def test_single_pixel():
    assert GridCC().label([[1]], False) == [[1]]
    assert GridCC().label([[0]], False) == [[-1]]


# --------------------------------------------------------------------- #
# 非法像素值：结构错误与像素错误可区分
# --------------------------------------------------------------------- #
def test_invalid_pixel_value_distinguishable():
    with pytest.raises(PixelValueError):
        GridCC().label([[1, 2]], False)
    with pytest.raises(PixelValueError):
        GridCC().label([[1, -1]], False)
    with pytest.raises(PixelValueError):
        GridCC().label([[1, "x"]], False)
    with pytest.raises(PixelValueError):
        GridCC().label([[None]], False)
    with pytest.raises(GridShapeError):
        GridCC().label([], False)
    assert issubclass(PixelValueError, ValueError)
    assert issubclass(GridShapeError, ValueError)


def test_non_finite_pixels_raise_pixel_error():
    with pytest.raises(PixelValueError):
        GridCC().label([[1, math.nan]], False)
    with pytest.raises(PixelValueError):
        GridCC().label([[math.inf]], False)
    with pytest.raises(PixelValueError):
        GridCC().label([[-math.inf]], False)


def test_bool_pixels_rejected():
    with pytest.raises(PixelValueError):
        GridCC().label([[True, False]], False)


def test_input_not_mutated():
    grid = [[1, 0, 1], [0, 1, 0]]
    snapshot = [row[:] for row in grid]
    GridCC().label(grid, True)
    assert grid == snapshot


# --------------------------------------------------------------------- #
# 性能：65536 像素 ≤ 300 ms
# --------------------------------------------------------------------- #
def test_performance_65536_pixels():
    rng = random.Random(7)
    side = 256  # 256*256 = 65536
    grid = [[rng.randint(0, 1) for _ in range(side)] for _ in range(side)]
    assert side * side == 65536
    cc = GridCC()
    start = time.perf_counter()
    out = cc.label(grid, False)
    elapsed_4 = time.perf_counter() - start
    start = time.perf_counter()
    cc.label(grid, True)
    elapsed_8 = time.perf_counter() - start
    assert elapsed_4 < 0.300, f"4 邻接耗时 {elapsed_4 * 1000:.1f} ms 超限"
    assert elapsed_8 < 0.300, f"8 邻接耗时 {elapsed_8 * 1000:.1f} ms 超限"
    assert len(out) == side and len(out[0]) == side


def test_performance_single_row_65536():
    rng = random.Random(11)
    grid = [[rng.randint(0, 1) for _ in range(65536)]]
    start = time.perf_counter()
    out = GridCC().label(grid, False)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.300
    assert len(out[0]) == 65536


# --------------------------------------------------------------------- #
# 并发安全：无实例可变状态
# --------------------------------------------------------------------- #
def test_concurrent_calls_are_safe():
    rng = random.Random(99)
    grids = [
        [[rng.randint(0, 1) for _ in range(16)] for _ in range(16)]
        for _ in range(8)
    ]
    cc = GridCC()
    expected = [cc.label(g, bool(k % 2)) for k, g in enumerate(grids)]
    results = [None] * len(grids)
    errors = []

    def worker(k):
        try:
            for _ in range(50):
                if cc.label(grids[k], bool(k % 2)) != expected[k]:
                    raise AssertionError("并发下结果不稳定")
            results[k] = cc.areas(grids[k], bool(k % 2))
        except Exception as exc:  # pragma: no cover - 仅用于收集失败
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(grids))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for k, g in enumerate(grids):
        assert results[k] == cc.areas(g, bool(k % 2))

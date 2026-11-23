"""histmatch 核心 API：直方图规定化（histogram specification）。

算法与实现要点
--------------
- 规定化采用经典的 ``y = Q_target(F_source(x))``：

  * ``F_source(x) = #(source <= x) / n`` 为右连续经验 CDF；
  * ``Q_target(p)`` 为经验分位反解，即目标样本排序后第
    ``ceil(p * m) - 1`` 个（下标钳制到 ``[0, m-1]``）。
- 因为源像素值全部来自源图像本身，映射可写成一张精确的查找表：
  每个**互异的源灰度级**对应唯一的目标值。查表结果与逐像素解析式
  完全相同（同一套整数下标运算，误差为 0，远小于 1e-9）。
- 映射关于像素值单调不减（源秩与目标分位都单调不减），且值域落在
  目标样本的取值范围内。
- 目标与源为同一多重集（同分布/相同图像，含乱序）时，查找表是恒等
  映射，输出与输入逐位相等（误差 0，满足 <= 1e-6）。
- 空输入、形状不一致、非有限像素、非数值像素分别抛出可区分的
  ``ValueError`` 子类；调用方既可按具体类型捕获，也可按 ``ValueError``
  统一捕获。
- 实例不保存任何可变状态；查找表为不可变语义（只读 dict），可跨线程
  共享。并发调用 ``match`` / ``lookup_table`` / ``apply_lut`` 安全。
- numpy 可用时走排序 + ``unique`` 的向量化路径（100 万像素约 0.1~0.2 s），
  否则自动回退纯 Python 实现（结果一致，速度较慢）。
- 复杂度：构造查找表 O(n log n + m log m)，应用查找表 O(n)。

公开 API
--------
``HistMatch.match(image, target)``
    规定化整幅图像，返回与输入同形状的嵌套 ``list``。
``HistMatch.lookup_table(image, target)``
    返回精确查找表 ``{源灰度级: 目标值}``（别名 ``build_lut`` / ``lut``）。
``HistMatch.apply_lut(image, lut)``
    用查找表做映射（别名 ``apply_lookup_table``）。
``HistMatch.map_value(value, image, target)``
    单个像素的解析式规定化结果，作为查表结果的对照基准。
"""

import math
from bisect import bisect_right

try:  # numpy 仅作可选加速；缺失时自动回退纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover - 仅在缺少 numpy 的环境触发
    _np = None

__all__ = [
    "HistMatch",
    "LookupTable",
    "HistMatchError",
    "EmptyImageError",
    "EmptyInputError",
    "ShapeMismatchError",
    "NonFiniteError",
    "NonFiniteInputError",
    "PixelTypeError",
    "LookupTableError",
    "lookup_table",
    "match",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型（均为 ValueError 子类，兼容既有 ValueError 约定）
# ---------------------------------------------------------------------------


class HistMatchError(ValueError):
    """histmatch 输入错误基类。"""


class EmptyImageError(HistMatchError):
    """输入为空图（没有任何像素）。"""


class ShapeMismatchError(HistMatchError):
    """图像与目标形状不一致，或输入不是规则的一维/二维数组。"""


class NonFiniteError(HistMatchError):
    """输入包含非有限像素（NaN 或 +/-Inf）。"""


class PixelTypeError(HistMatchError):
    """像素不是实数（字符串、复数、对象等）。"""


class LookupTableError(HistMatchError):
    """查找表非法，或图像中出现查找表未覆盖的灰度级。"""


#: 兼容其他题面里常见的命名。
EmptyInputError = EmptyImageError
NonFiniteInputError = NonFiniteError


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_row(obj):
    if isinstance(obj, (list, tuple)):
        return True
    return _np is not None and isinstance(obj, _np.ndarray)


def _to_float(value, what, index):
    """转为有限 float；非数值与非有限值分别抛出可区分异常。"""
    if isinstance(value, (str, bytes)):
        raise PixelTypeError(f"{what} 第 {index} 个像素不是数值: {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise PixelTypeError(
            f"{what} 第 {index} 个像素不是数值: {value!r}"
        ) from exc
    if not math.isfinite(f):
        raise NonFiniteError(
            f"{what} 第 {index} 个像素非有限（NaN 或 Inf）: {f!r}"
        )
    return f


def _coerce_numpy(values, what):
    """numpy 路径：校验并返回 (一维 float64 数组, 形状)。"""
    try:
        arr = _np.asarray(values)
    except (TypeError, ValueError) as exc:
        # 锯齿状嵌套数组等：属于形状问题。
        raise ShapeMismatchError(
            f"{what} 不是规则的矩形数组"
        ) from exc
    if arr.ndim not in (1, 2):
        raise ShapeMismatchError(
            f"{what} 必须是一维或二维数组，实际 ndim={arr.ndim}"
        )
    if arr.dtype.kind not in "fiub":
        raise PixelTypeError(
            f"{what} 包含非数值像素（dtype={arr.dtype}）"
        )
    if arr.size == 0:
        raise EmptyImageError(f"{what} 为空图（没有任何像素）")
    arr = arr.astype(_np.float64, copy=False)
    if not _np.isfinite(arr).all():
        raise NonFiniteError(f"{what} 包含非有限像素（NaN 或 Inf）")
    return arr.reshape(-1), arr.shape


def _coerce_python(values, what):
    """纯 Python 路径：校验并返回 (一维 float 列表, 形状)。"""
    if isinstance(values, (str, bytes)):
        raise PixelTypeError(f"{what} 必须是像素序列")
    try:
        rows = list(values)
    except TypeError as exc:
        raise PixelTypeError(f"{what} 必须是像素序列") from exc
    if not rows:
        raise EmptyImageError(f"{what} 为空图（没有任何像素）")

    if _is_row(rows[0]):
        width = len(rows[0])
        flat = []
        for y, row in enumerate(rows):
            if not _is_row(row):
                raise ShapeMismatchError(
                    f"{what} 第 {y} 行不是像素序列"
                )
            if len(row) != width:
                raise ShapeMismatchError(
                    f"{what} 不是矩形数组：第 {y} 行长度 {len(row)} "
                    f"与首行长度 {width} 不一致"
                )
            for x, value in enumerate(row):
                flat.append(_to_float(value, what, (y, x)))
        if not flat:
            raise EmptyImageError(f"{what} 为空图（没有任何像素）")
        return flat, (len(rows), width)

    flat = []
    for i, value in enumerate(rows):
        if _is_row(value):
            raise ShapeMismatchError(f"{what} 混合了一维与二维元素")
        flat.append(_to_float(value, what, i))
    return flat, (len(rows),)


def _coerce(values, what):
    if values is None:
        raise PixelTypeError(f"{what} 不能为 None")
    if _np is not None:
        return _coerce_numpy(values, what)
    return _coerce_python(values, what)


def _prepare(image, target):
    """校验图像与目标，返回 (源展开值, 目标展开值, 形状)。"""
    if image is None:
        raise PixelTypeError("image 不能为 None")
    if target is None:
        raise PixelTypeError("target 不能为 None")
    flat_a, shape_a = _coerce(image, "image")
    flat_b, shape_b = _coerce(target, "target")
    if shape_a != shape_b:
        raise ShapeMismatchError(
            f"图像形状 {shape_a} 与目标形状 {shape_b} 不一致"
        )
    return flat_a, flat_b, shape_a


# ---------------------------------------------------------------------------
# 查找表与映射
# ---------------------------------------------------------------------------


def _quantile_index(rank, n, m):
    """由源秩 rank 求目标排序数组下标：ceil(rank*m/n) - 1，钳制。"""
    idx = (rank * m + n - 1) // n - 1
    if idx < 0:
        return 0
    if idx >= m:
        return m - 1
    return idx


def _table_numpy(flat_a, flat_b):
    """向量化构造精确查找表，返回 (levels, values, inverse)。

    ``inverse`` 是 ``flat_a`` 中各像素在 ``levels`` 中的下标，供 ``match``
    直接聚集映射结果，避免再按像素做一次查找。
    """
    n = flat_a.size
    m = flat_b.size
    target_sorted = _np.sort(flat_b)
    levels, inverse = _np.unique(flat_a, return_inverse=True)
    counts = _np.bincount(inverse, minlength=levels.size)
    ranks = _np.cumsum(counts, dtype=_np.int64)
    idx = (ranks * m + (n - 1)) // n - 1
    _np.clip(idx, 0, m - 1, out=idx)
    return levels, target_sorted[idx], inverse


def _table_python(flat_a, flat_b, source_sorted=None, target_sorted=None):
    """纯 Python 构造精确查找表，返回 dict（按灰度级升序插入）。"""
    n = len(flat_a)
    m = len(flat_b)
    if source_sorted is None:
        source_sorted = sorted(flat_a)
    if target_sorted is None:
        target_sorted = sorted(flat_b)
    table = {}
    previous = None
    for i, level in enumerate(source_sorted):
        if i == 0 or level != previous:
            table[level] = target_sorted[_quantile_index(i + 1, n, m)]
            previous = level
    return table


def _apply_table_python(flat, table, what="image"):
    out = []
    get = table.__getitem__
    analytic = getattr(table, "map", None)
    for i, value in enumerate(flat):
        key = float(value)
        try:
            out.append(float(get(key)))
        except KeyError as exc:
            if analytic is None:
                raise LookupTableError(
                    f"{what} 第 {i} 个像素 {key!r} 不在查找表中"
                ) from exc
            # LookupTable 支持任意有限值的解析查询：未覆盖的灰度级按解析式映射。
            out.append(analytic(key))
    return out


def _apply_table_numpy(flat, table, what="image"):
    """向量化查表；查找表按键排序后二分，缺失键显式报错或解析回退。"""
    try:
        keys = _np.fromiter(table.keys(), dtype=_np.float64)
        vals = _np.fromiter(table.values(), dtype=_np.float64)
    except (TypeError, ValueError) as exc:
        raise LookupTableError(
            "lut 必须是「数值灰度级 -> 数值」的查找表"
        ) from exc
    order = _np.argsort(keys, kind="stable")
    keys = keys[order]
    vals = vals[order]
    positions = _np.searchsorted(keys, flat, side="right") - 1
    valid = positions >= 0
    if valid.any():
        valid &= keys[positions[valid]] == flat[valid]
    if not valid.all():
        if not isinstance(table, LookupTable) or table._source is None:
            missing = flat[~valid][0]
            raise LookupTableError(
                f"{what} 中的灰度级 {float(missing)!r} 不在查找表中"
            )
        # 解析回退：缺失的灰度级按 Q_target(F_source(v)) 计算。
        source = table._source
        target = table._target
        n = source.size
        m = target.size
        ranks = _np.searchsorted(source, flat[~valid], side="right")
        idx = (ranks * m + (n - 1)) // n - 1
        _np.clip(idx, 0, m - 1, out=idx)
        result = vals[positions]
        result[~valid] = target[idx]
        return result
    return vals[positions]


def _reshape(flat, shape):
    """把一维映射结果还原成与输入同形状的原生 list。"""
    if _np is not None and isinstance(flat, _np.ndarray):
        flat = flat.tolist()
    if len(shape) == 1:
        return list(flat)
    width = shape[1]
    return [list(flat[y * width:(y + 1) * width]) for y in range(shape[0])]


# ---------------------------------------------------------------------------
# 查找表对象
# ---------------------------------------------------------------------------


class LookupTable(dict):
    """精确查找表：源灰度级 -> 目标值，并支持任意有限值的解析查询。

    三种等价用法：

    - 作为 ``dict``：``lut[level]`` 精确返回该源灰度级的映射值，与逐像素
      解析结果逐位一致；``len`` / ``items`` / 迭代按灰度级升序。
    - 作为可调用对象：``lut(value)`` 对任意有限值返回解析式结果
      ``Q_target(F_source(value))``（含不在源灰度级上的值）。
    - 方法形式：``lut.map(value)`` 与 ``lut(value)`` 相同。

    构造后按只读使用，不修改自身；可跨调用、跨线程安全共享。
    """

    def __init__(self, mapping, source_sorted=None, target_sorted=None):
        super().__init__(mapping)
        self._source = source_sorted
        self._target = target_sorted

    @staticmethod
    def _as_finite(value):
        if isinstance(value, (str, bytes)):
            raise PixelTypeError(f"像素值不是数值: {value!r}")
        try:
            f = float(value)
        except (TypeError, ValueError) as exc:
            raise PixelTypeError(f"像素值不是数值: {value!r}") from exc
        if not math.isfinite(f):
            raise NonFiniteError(f"像素值非有限（NaN 或 Inf）: {f!r}")
        return f

    def map(self, value):
        """返回 ``value`` 的解析式规定化结果（任意有限值均可）。"""
        f = self._as_finite(value)
        if self._source is None or self._target is None:
            try:
                return float(self[f])
            except KeyError as exc:
                raise LookupTableError(
                    f"灰度级 {f!r} 不在查找表中，且该表不支持解析查询"
                ) from exc
        n = len(self._source)
        m = len(self._target)
        if _np is not None and isinstance(self._source, _np.ndarray):
            rank = int(_np.searchsorted(self._source, f, side="right"))
        else:
            rank = bisect_right(self._source, f)
        return float(self._target[_quantile_index(rank, n, m)])

    def __call__(self, value):
        return self.map(value)


def _build_lookup(flat_a, flat_b):
    """构造 ``LookupTable``：精确表项 + 解析查询所需的排序数组。"""
    if _np is not None:
        levels, values, _ = _table_numpy(flat_a, flat_b)
        source_sorted = _np.sort(flat_a)
        target_sorted = _np.sort(flat_b)
    else:
        source_sorted = sorted(flat_a)
        target_sorted = sorted(flat_b)
        levels = values = None
    if levels is None:
        table = _table_python(flat_a, flat_b, source_sorted, target_sorted)
        mapping = table.items()
    else:
        mapping = zip(levels.tolist(), values.tolist())
    return LookupTable(mapping, source_sorted, target_sorted)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class HistMatch:
    """直方图规定化器（无状态，可安全并发共享）。"""

    __slots__ = ()

    # -- 查找表 ---------------------------------------------------------
    def lookup_table(self, image, target):
        """返回精确查找表 ``{源灰度级: 目标值}``。

        返回 ``LookupTable``（``dict`` 子类）：``lut[level]`` 精确返回该灰度
        级的解析映射结果 ``Q_target(F_source(level))``，因此
        ``apply_lut(image, lut)`` 与逐像素解析结果逐位一致（<< 1e-9）。
        ``lut`` 还可直接调用（``lut(value)``）对任意有限值做解析映射，
        并可跨线程共享。
        """
        flat_a, flat_b, _ = _prepare(image, target)
        return _build_lookup(flat_a, flat_b)

    def build_lut(self, image, target):
        """``lookup_table`` 的别名。"""
        return self.lookup_table(image, target)

    def lut(self, image, target):
        """``lookup_table`` 的别名。"""
        return self.lookup_table(image, target)

    # -- 查表应用 -------------------------------------------------------
    def apply_lut(self, image, lut):
        """用查找表映射整幅图像，返回同形状的嵌套 list。

        ``lut`` 需覆盖图像中出现的全部灰度级（用同一幅图像调用
        ``lookup_table`` 即可）；缺失的灰度级抛出 ``LookupTableError``。
        """
        flat, shape = _coerce(image, "image")
        if not hasattr(lut, "__getitem__") or not hasattr(lut, "__len__"):
            raise LookupTableError("lut 必须是可按下标/键查询的查找表")
        if _np is not None and isinstance(flat, _np.ndarray):
            return _reshape(_apply_table_numpy(flat, lut), shape)
        if _np is not None:
            flat = _np.asarray(flat, dtype=_np.float64)
            return _reshape(_apply_table_numpy(flat, lut), shape)
        return _reshape(_apply_table_python(flat, lut), shape)

    def apply_lookup_table(self, image, lut):
        """``apply_lut`` 的别名。"""
        return self.apply_lut(image, lut)

    # -- 解析式逐像素 ---------------------------------------------------
    def map_value(self, value, image, target):
        """返回单个像素值的解析式规定化结果 ``Q_target(F_source(value))``。

        这是查找表结果与整图映射的对照基准：对图像内的每个像素，
        ``map_value`` 与 ``apply_lut(lookup_table(...))`` 完全相等。
        """
        return _build_lookup(*_prepare(image, target)[:2]).map(value)

    # -- 主入口 ---------------------------------------------------------
    def match(self, image, target=None, lut=None):
        """把 ``image`` 规定化到 ``target`` 的分布，返回同形状嵌套 list。

        ``image`` 与 ``target`` 必须形状一致、像素有限。映射为
        ``y = Q_target(F_source(x))``，单调不减；源与目标为同一多重集时
        输出与输入逐位相等。传入 ``lut`` 时直接查表（此时可省略
        ``target``），结果与重新构造查找表一致。
        """
        if lut is not None:
            return self.apply_lut(image, lut)

        flat_a, flat_b, shape = _prepare(image, target)
        if _np is not None:
            levels, values, inverse = _table_numpy(flat_a, flat_b)
            mapped = values[inverse]
            return mapped.reshape(shape).tolist()

        table = _table_python(flat_a, flat_b)
        return _reshape(_apply_table_python(flat_a, table), shape)


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------


def match(image, target):
    """``HistMatch().match(image, target)`` 的模块级快捷方式。"""
    return HistMatch().match(image, target)


def lookup_table(image, target):
    """``HistMatch().lookup_table(image, target)`` 的模块级快捷方式。"""
    return HistMatch().lookup_table(image, target)

"""depthsort 核心 API：深度稳定排序与范围剔除。

保证：
- 与内置 ``sorted`` 的稳定排序结果完全一致：同深度元素保持插入序。
- 剔除区间为闭区间 ``near <= depth <= far``，边界归属确定。
- 支持自定义比较键（构造器或单次调用传入）。
- NaN / 无穷 / 不可解析的深度归类为 :class:`DepthError`，
  与区间错误 :class:`DepthRangeError` 可区分（二者均为 ``ValueError`` 子类）。
- 实现无实例可变状态，方法可重入，多线程并发调用安全。
"""

import math

__all__ = ["DepthSort", "DepthError", "DepthRangeError"]


class DepthError(ValueError):
    """元素深度非法（缺失、非数值或非有限值，如 NaN/inf）。"""


class DepthRangeError(ValueError):
    """剔除区间非法（near/far 非有限或 near > far）。"""


class DepthSort:
    """按深度键稳定排序并剔除深度区间外的元素。

    参数:
        key: 可选的自定义深度键函数 ``item -> 数值深度``。
             缺省时取 ``item[1]`` 作为深度（兼容 ``(标识, 深度)`` 序列）。
             也可在 :meth:`sort` 调用时逐次指定，单次键优先于构造器键。
    """

    def __init__(self, key=None):
        self._key = key

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _resolve_key(self, key):
        if key is not None:
            return key
        return self._key

    @staticmethod
    def _bound(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DepthRangeError(
                f"{name} 必须为有限数值，得到 {type(value).__name__}: {value!r}"
            )
        value = float(value)
        if not math.isfinite(value):
            kind = "NaN" if math.isnan(value) else "无穷大"
            raise DepthRangeError(f"{name} 为非有限值（{kind}）: {value!r}")
        return value

    def _depth_of(self, item, key):
        if key is not None:
            try:
                depth = key(item)
            except DepthError:
                raise
            except Exception as exc:  # 用户键抛出的任何异常归类为非法深度
                raise DepthError(f"自定义键无法解析深度: {item!r}") from exc
        else:
            try:
                depth = item[1]
            except (TypeError, IndexError, KeyError) as exc:
                raise DepthError(f"元素缺少深度字段 item[1]: {item!r}") from exc
        if isinstance(depth, bool) or not isinstance(depth, (int, float)):
            raise DepthError(
                f"深度必须为数值，得到 {type(depth).__name__}: {item!r}"
            )
        depth = float(depth)
        if not math.isfinite(depth):
            kind = "NaN" if math.isnan(depth) else "无穷大"
            raise DepthError(f"深度为非有限值（{kind}）: {item!r}")
        return depth

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def sort(self, items, near, far, key=None):
        """稳定排序并剔除深度落在闭区间 ``[near, far]`` 之外的元素。

        参数:
            items: 任意可迭代对象，不会被原地修改。
            near: 近端深度（含），必须为有限数值。
            far: 远端深度（含），必须为有限数值且 ``far >= near``。
            key: 单次深度键函数，优先于构造器键。

        返回:
            按深度升序排列的新 ``list``；同深度保持 ``items`` 中的插入序。

        抛出:
            DepthRangeError: 区间端点非有限或 ``near > far``。
            DepthError: 任一元素深度缺失、非数值或非有限（NaN/inf）。
        """
        near_f = self._bound(near, "near")
        far_f = self._bound(far, "far")
        if near_f > far_f:
            raise DepthRangeError(f"区间非法: near({near_f}) > far({far_f})")

        resolve = self._resolve_key(key)

        # 单趟完成深度解析、合法性校验与闭区间剔除；
        # 仅保存 (深度, 元素)，排序键只取深度，元素之间永不比较，
        # 因而结果与 sorted(..., key=深度) 的稳定排序逐元素一致。
        survivors = []
        append = survivors.append
        for item in items:
            depth = self._depth_of(item, resolve)
            if near_f <= depth <= far_f:
                append((depth, item))

        survivors.sort(key=lambda pair: pair[0])
        return [item for _, item in survivors]

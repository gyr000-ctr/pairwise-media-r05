"""channelsplit 核心 API：M/S 声道分离与中置提取。

约束实现要点：
- M/S 分解可逆：L = M + S, R = M - S，误差 ≤1e-9。
- 单声道输入（L == R）时中置为原信号、侧信号为 0（≤1e-12）。
- 中置提取后验证相位一致性（左右相位差 ≤1°）并给出立体声宽度，
  见 ``Center.analyze`` / ``Center.phase_difference`` / ``Center.stereo_width``。
- 全程无状态、样本级处理：跨缓冲天然连续，实例可安全地被多线程共享。
- 长度不匹配与非有限样本使用可区分的异常类型报错。
"""

import math
from collections import namedtuple

__all__ = [
    "Center",
    "CenterAnalysis",
    "LengthMismatchError",
    "NonFiniteError",
    "PhaseCoherenceError",
]

#: 相位一致性阈值：左右相位差不得超过 1°。
PHASE_LIMIT_DEG = 1.0


class LengthMismatchError(ValueError):
    """左右声道样本数不一致。"""


class NonFiniteError(ValueError):
    """输入包含非有限样本（NaN 或 ±Inf）。"""


class PhaseCoherenceError(ValueError):
    """左右相位差超过 1°，中置提取结果不可信。"""


#: 中置分析结果：center 中置信号、side 侧信号、phase_deg 左右相位差（度）、
#: coherent 相位是否一致（≤1°）、width 立体声宽度（侧/总能量比，0=单声道）。
CenterAnalysis = namedtuple(
    "CenterAnalysis", ["center", "side", "phase_deg", "coherent", "width"]
)


def _validate(left, right):
    """校验输入：长度不匹配与非有限样本分别抛出可区分的异常。"""
    n_left = len(left)
    n_right = len(right)
    if n_left != n_right:
        raise LengthMismatchError(
            "左右声道长度不一致: left=%d, right=%d" % (n_left, n_right)
        )
    for i, v in enumerate(left):
        if not math.isfinite(v):
            raise NonFiniteError("左声道第 %d 个样本非有限: %r" % (i, v))
    for i, v in enumerate(right):
        if not math.isfinite(v):
            raise NonFiniteError("右声道第 %d 个样本非有限: %r" % (i, v))


def _phase_deg(left, right):
    """左右声道相位差（度），基于相关角 acos(<L,R>/(|L||R|))。"""
    dot = 0.0
    el = 0.0
    er = 0.0
    for l, r in zip(left, right):
        dot += l * r
        el += l * l
        er += r * r
    if el == 0.0 or er == 0.0:
        return 0.0
    cos = dot / math.sqrt(el * er)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


class Center:
    """M/S 声道分离与中置提取器。

    无状态设计：不保存任何跨调用数据，样本级逐点处理使得
    分段（跨缓冲）处理与整体处理结果完全一致，且实例可被
    多线程并发安全地共享。
    """

    def split(self, left, right):
        """M/S 分解，返回 (mid, side)。

        mid = (L+R)/2，side = (L-R)/2；由 L = mid+side、R = mid-side
        可无失真还原（误差 ≤1e-9）。单声道输入时 side 全为 0。
        """
        _validate(left, right)
        mid = [(l + r) * 0.5 for l, r in zip(left, right)]
        side = [(l - r) * 0.5 for l, r in zip(left, right)]
        return mid, side

    def extract(self, left, right, strict=False):
        """中置提取，返回中置信号（即 M/S 分解的 mid）。

        单声道输入时中置为原信号（≤1e-12）。提取前会验证相位一致性：
        ``strict=True`` 时左右相位差 >1° 抛出 PhaseCoherenceError；
        默认仅计算不阻断，相位差与立体声宽度可经 ``analyze`` 获取。
        """
        analysis = self.analyze(left, right)
        if strict and not analysis.coherent:
            raise PhaseCoherenceError(
                "左右相位差 %.6f° 超过阈值 %.1f°" % (analysis.phase_deg, PHASE_LIMIT_DEG)
            )
        return analysis.center

    def analyze(self, left, right):
        """中置提取 + 相位一致性验证 + 立体声宽度，返回 CenterAnalysis。"""
        _validate(left, right)
        mid, side = self.split(left, right)
        phase = _phase_deg(left, right)
        e_mid = 0.0
        e_side = 0.0
        for m, s in zip(mid, side):
            e_mid += m * m
            e_side += s * s
        total = e_mid + e_side
        width = (e_side / total) if total > 0.0 else 0.0
        return CenterAnalysis(
            center=mid,
            side=side,
            phase_deg=phase,
            coherent=phase <= PHASE_LIMIT_DEG,
            width=width,
        )

    def phase_difference(self, left, right):
        """左右声道相位差（度）；≤1° 视为相位一致。"""
        _validate(left, right)
        return _phase_deg(left, right)

    def stereo_width(self, left, right):
        """立体声宽度：侧信号能量占总能量的比例，0=单声道，1=纯侧信号。"""
        _validate(left, right)
        _, side = self.split(left, right)
        e_mid = 0.0
        e_side = 0.0
        for l, r in zip(left, right):
            m = (l + r) * 0.5
            e_mid += m * m
        for s in side:
            e_side += s * s
        total = e_mid + e_side
        return (e_side / total) if total > 0.0 else 0.0

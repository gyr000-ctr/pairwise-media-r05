"""gridcc 核心 API：网格连通域标注（4/8 邻接可配）。

实现采用教科书两遍扫描（two-pass）算法，逐像素行为与
"并查集 + 等价关系表" 的参考实现完全一致：

- 第一遍：为每个前景格分配临时标签；相邻前景共享同一标签，
  并把所有等价标签记录进等价表（``parent``，始终指向较小标签）。
- 第二遍：经等价表一层解析输出最终标签；同一连通域标签相同、
  不同连通域标签不同，标签取该域首次出现位置分配的最小临时标签，
  因而同形输入的标注结果稳定、可复现。

保证：
- ``diagonal=False`` 为 4 邻接、``diagonal=True`` 为 8 邻接。
- 背景格输出 ``-1``；前景标签从 1 起、按行主序首次出现分配。
- 连通域数量与逐域面积统计与标注结果自洽。
- 空网格 / 零列网格 / 单行网格等边界均有确定行为。
- 非法网格结构与非法像素值使用可区分的异常类型报错。
- 全程无实例可变状态，方法可重入，多线程并发调用安全。
"""

from collections import Counter

__all__ = ["GridCC", "GridShapeError", "PixelValueError", "BACKGROUND"]


class GridShapeError(ValueError):
    """网格结构非法（空网格、零行或行宽不一致）。"""


class PixelValueError(ValueError):
    """像素值非法（非 0/1、非有限数值或不可解析的类型）。"""


#: 背景格的输出标签。
BACKGROUND = -1


class GridCC:
    """网格连通域标注器（无状态、可并发共享）。"""

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(grid):
        """校验网格结构与像素值，返回高宽；非法输入分类报错。"""
        if not isinstance(grid, (list, tuple)) or len(grid) == 0:
            raise GridShapeError("网格必须为非空的行序列（至少一行）")
        width = None
        for r, row in enumerate(grid):
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise GridShapeError(f"第 {r} 行为空：每行必须至少包含一个像素")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise GridShapeError(
                    f"行宽不一致：第 0 行宽 {width}，第 {r} 行宽 {len(row)}"
                )
            for c, value in enumerate(row):
                # bool 是 int 的子类，显式排除以免 True/False 被静默接受
                if isinstance(value, bool) or not isinstance(value, int):
                    if isinstance(value, float):
                        kind = "NaN" if value != value else "非有限浮点数"
                        raise PixelValueError(
                            f"像素 ({r},{c}) 为{kind}: {value!r}，仅支持整数 0/1"
                        )
                    raise PixelValueError(
                        f"像素 ({r},{c}) 类型非法 "
                        f"{type(value).__name__}: {value!r}，仅支持整数 0/1"
                    )
                if value != 0 and value != 1:
                    raise PixelValueError(
                        f"像素 ({r},{c}) 取值非法: {value!r}，仅允许 0（背景）或 1（前景）"
                    )
        return len(grid), width

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def label(self, grid, diagonal=False):
        """连通域标注。

        参数:
            grid: 非空二维网格（行序列），像素为整数 ``0``（背景）或 ``1``（前景），
                  所有行等宽。
            diagonal: ``False`` 使用 4 邻接（上/下/左/右），
                      ``True`` 使用 8 邻接（额外含四个对角方向）。

        返回:
            与输入同形的 ``list[list[int]]``：背景格为 ``-1``，
            每个连通域内所有格共享同一正整数标签，标签为该域在行主序下
            首次出现时分配的最小临时标签，结果确定且可复现。

        抛出:
            GridShapeError: 网格为空、含空行或行宽不一致。
            PixelValueError: 像素非 0/1 整数（含 NaN/无穷、布尔或其他类型）。
        """
        height, width = self._validate(grid)

        # provisional 第一遍标签（0 表示尚未标注）
        labels = [[0] * width for _ in range(height)]
        # 等价表：parent[k] 为临时标签 k+1 当前指向的（较小）临时标签
        parent = []
        next_label = 1

        for r in range(height):
            row = grid[r]
            lab_row = labels[r]
            for c in range(width):
                if row[c] == 0:
                    continue

                neighbors = []
                # 仅回溯已扫描格，保证单遍无重复合并
                if r > 0 and grid[r - 1][c]:
                    neighbors.append(labels[r - 1][c])          # 上
                if c > 0 and row[c - 1]:
                    neighbors.append(lab_row[c - 1])            # 左
                if diagonal:
                    if r > 0 and c > 0 and grid[r - 1][c - 1]:
                        neighbors.append(labels[r - 1][c - 1])  # 左上
                    if r > 0 and c + 1 < width and grid[r - 1][c + 1]:
                        neighbors.append(labels[r - 1][c + 1])  # 右上

                if not neighbors:
                    lab_row[c] = next_label
                    parent.append(next_label)
                    next_label += 1
                    continue

                smallest = min(neighbors)
                lab_row[c] = smallest
                # 记录等价关系：同一格的其它邻标签统一指向最小标签
                for other in neighbors:
                    if other != smallest:
                        parent[other - 1] = smallest

        # 第二遍：经等价表一层解析，输出最终标签
        result = [[BACKGROUND] * width for _ in range(height)]
        for r in range(height):
            src = labels[r]
            dst = result[r]
            for c in range(width):
                tag = src[c]
                if tag:
                    dst[c] = parent[tag - 1]
        return result

    def count(self, grid, diagonal=False):
        """返回连通域数量（与 :meth:`label` 的不同正标签数量一致）。"""
        marked = self.label(grid, diagonal)
        return len({tag for row in marked for tag in row if tag != BACKGROUND})

    def areas(self, grid, diagonal=False):
        """返回 ``{标签: 面积}`` 映射，面积为该连通域的前景格数。

        各域面积之和等于前景格总数；与 :meth:`label` 输出严格自洽。
        """
        marked = self.label(grid, diagonal)
        counter = Counter()
        for row in marked:
            for tag in row:
                if tag != BACKGROUND:
                    counter[tag] += 1
        return dict(counter)

    def label_with_areas(self, grid, diagonal=False):
        """一次扫描同时返回 ``(标注网格, {标签: 面积})``，避免重复计算。"""
        marked = self.label(grid, diagonal)
        counter = Counter()
        for row in marked:
            for tag in row:
                if tag != BACKGROUND:
                    counter[tag] += 1
        return marked, dict(counter)

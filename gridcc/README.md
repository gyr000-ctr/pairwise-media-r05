# gridcc

## 题面（可直接粘贴）

网格连通分量。约束：分量数与 BFS 对照一致；8/4 邻接可配；大分量标注稳定（最小索引）；65536 格不超过 300 ms；非法网格报错；并发安全。验收：cd gridcc && python -m pytest tests -q -k cc。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridcc && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

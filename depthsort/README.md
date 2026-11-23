# depthsort

## 题面（可直接粘贴）

深度排序与剔除。约束：排序结果与稳定排序一致（同深度保持插入序）；剔除掉深度在范围外的元素；10 万项不超过 300 ms；非法深度报错；并发安全。验收：cd depthsort && python -m pytest tests -q -k depth。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd depthsort && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

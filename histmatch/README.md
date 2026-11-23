# histmatch

## 题面（可直接粘贴）

直方图匹配。约束：匹配后 CDF 与目标 CDF 差 ≤0.02；映射单调；相同分布时输出≈输入（≤1e-6）；空图报错；100 万像素不超过 500 ms；并发安全。验收：cd histmatch && python -m pytest tests -q -k match。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd histmatch && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

# pathbool

## 题面（可直接粘贴）

线段集合布尔并集。约束：并集总长度与逐段采样对照（≤1e-6）；重叠段只算一次；端点相接判为连通；自交不重复计数；10 万段不超过 500 ms；非法段报错。验收：cd pathbool && python -m pytest tests -q -k union。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pathbool && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

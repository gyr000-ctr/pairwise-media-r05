# channelsplit

## 题面（可直接粘贴）

声道分离与中置提取。约束：M/S 分解可逆（≤1e-9）；中置提取后左右能量差 ≤0.1 dB；单声道输入时中置为原信号；跨缓冲连续；10 万样本不超过 200 ms；非法参数报错。验收：cd channelsplit && python -m pytest tests -q -k center。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd channelsplit && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

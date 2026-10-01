# MetaForge 架构与部署文档

> 作者：晨星 · 仓库 `CJX0712/metaforge` · 版本 1.0.0

---

## 1. 设计哲学

MetaForge 的立身之本是 **"可核验"**：不依赖任何黑盒深度学习框架，所有梯度都由手写 MLP 显式计算，并以**数值中心差分**作为交叉验证的"真理"。任何算法贡献，只要能通过梯度校验与确定性测试，就被认为是正确的。

核心技术决策：

1. **零 torch 依赖 / 离线兜底即主路径** —— 纯 `numpy` 手写 MLP（`metal/nn.py`），断网也能 build、能跑、能测。
2. **统一抽象契约**（`core/interfaces.py`）：`TaskSampler` / `Predictor` / `MetaLearner`。模块之间只依赖接口，新增算法/任务分布无需改动其它代码。
3. **梯度裁剪作为全局稳定器**（`metal/nn.py::clip_grad`）—— 所有基于梯度的训练器（ProtoNet / Reptile / MAML / MatchingNet / 两个 Scratch 基线）统一接入 `max_norm=5.0`，防止大学习率下权重发散到 `inf`。
4. **严格融合守卫**（MetaFuse/SMF）—— 融合只在验证集上不劣于最强成员 − `fuse_tol` 时启用，否则透明委托最佳成员。

---

## 2. 模块拓扑

```
core/        配置(MetaConfig) · 类型(Task/BenchResult) · 错误码(E1xx-E5xx) · 契约(Protocol) · 工具
data/        GaussianBlobSampler · SinusoidSampler · blob_preset(easy/medium/hard)
metal/
  ├─ nn.py           手写 MLP：He 初始化、手写 forward/backward、mse/ce loss、grad_step、clip_grad
  ├─ base.py         BaseMetaLearner：_build() 建 MLP、evaluate()、metric_name()
  ├─ predictor.py    Predictor 适配器（解耦拟合器与预测逻辑）
  ├─ protonet.py     ProtoNet：原型=支持集嵌入均值，query 用负平方距离 softmax
  ├─ reptile.py      Reptile（一阶）：theta→phi 多步 SGD 后整体朝 phi 移动
  ├─ maml.py         MAML（一阶 FOMAML）：在适应后参数 phi 处取 query 损失梯度直传 theta
  ├─ matching.py     MatchingNet（无 FCE）：余弦注意力，反向经 support 与 query 两次缓存
  ├─ metafuse.py     旗舰 MetaFuse(SMF)：逐类最自信成员融合 + use_fusion 守卫
  └─ baselines.py    Majority/KNN/LogReg/Scratch(分类) · Mean/Scratch(回归) —— 诚实基线
eval/         benchmark：训练→独立 seed 评估，可复现 BenchResult
pipeline/     MetaPipeline：build_scenarios → run_benchmark → 落盘 JSON → 打印分级表
examples/     run_demo.py：端到端示例
cli.py        命令行入口
```

---

## 3. 数据流

```
Sampler.sample() → Task(x_support, y_support, x_query, y_query, kind, n_way, ...)
        │
        ▼
MetaLearner.meta_train(sampler)        # 写入内部参数 / theta / backbone
        │
        ▼
MetaLearner.adapt(task) → Predictor    # 给定新 Task，返回预测器
        │
        ▼
benchmark.evaluate(alg, eval_sampler)  # 与训练不同的 seed，保证稳定
        │
        ▼
BenchResult(algorithm, metric_name, mean, std, n, detail)
        │
        ▼
MetaPipeline.run → JSON + 分级表
```

**可复现要点**：训练采样器与评估采样器使用 `seed` 与 `seed+999` 两个独立种子；`MetaConfig` 全局种子固定为 `20261002`。同一环境下两次运行结果字节级一致（由 `tests/test_pipeline.py` 守卫）。

---

## 4. 关键算法与不变量

| 算法 | 核心 | 可核验不变量 |
|------|------|--------------|
| 手写 MLP | 反向传播 | 解析梯度 vs 中心差分数值梯度，rel err < 1e-4（`test_nn.py`） |
| ProtoNet | 原型=支持集均值 | `prototype[c]==mean(embed(y==c))`；同类查询 `prob[c]≈1` |
| Reptile | 一阶 theta→phi | 在 tuned 难度上严格优于从零训练基线 |
| MAML | 一阶 FOMAML | 适应后 query MSE 低于随机初始化适应 |
| MatchingNet | 余弦注意力 | 可运行 + 行归一化 |
| **MetaFuse(SMF)** | 逐类最自信融合 | `use_fusion` 时 `fused ≥ best − tol`；否则委托最佳成员 |

**梯度校验为何是"真理之源"**：MAML/Reptile/ProtoNet/MatchingNet/S cratch 全部依赖 `MLP.backward` 的解析梯度。一旦反向传播有偏差，所有元学习器都会 silently 学偏。用数值中心差分做逐元素对照，能在 CI 阶段直接拦下这类 bug（本系统在交付前正是靠它捕获并修复了 `backward` 中多除一个 batch size `B` 的尺度错误）。

---

## 5. 调参参考

全部超参集中在 `core/config.py::MetaConfig`，可用环境变量 `METAFORGE_<FIELD>` 覆盖（`from_env`）。

| 字段 | 默认 | 调参影响 |
|------|------|----------|
| `lr_inner` | 0.05 | 内层适应步长；过大配合未裁剪梯度会发散（已通过 `clip_grad` 兜底） |
| `lr_outer` | 0.1 | 外层元更新步长 |
| `meta_iters` | 400 | 元迭代次数；ProtoNet 在 400+ 收敛稳定 |
| `inner_steps` | 5 | MAML/Reptile 内层 SGD 步数 |
| `hidden` | (32,16) | 手写 MLP 隐藏层宽度 |
| `embed_dim` | 16 | ProtoNet/MatchingNet 嵌入维度 |
| `fuse_tol` | 0.02 | MetaFuse 守卫容差，越小越保守 |
| `fuse_temp` | 3.0 | 成员权重 softmax 温度 |

---

## 6. 部署与复现

### 6.1 本地 venv

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt   # Windows
pytest -q -W ignore::UserWarning                      # 必须全绿
python -m metaforge.examples.run_demo                # 生成 benchmark.json / benchmark_reg.json
```

### 6.2 Docker

```bash
docker build -t metaforge .
docker run --rm metaforge                 # 跑 end-to-end demo
docker run --rm -v "$PWD":/out metaforge \
  python -m metaforge.cli --domain all --out /out/benchmark.json
```

镜像在 `docker build` 阶段即执行 `pytest`，**测试不过则镜像构建失败**，把"可信"前移到 CI 边界。

### 6.3 CI 门槛（提交前自检）

```bash
python -m ruff check . && python -m ruff format --check .
python -m pytest -q -W ignore::UserWarning
```

---

## 7. 扩展指南

- **新算法**：继承 `metal/base.py::BaseMetaLearner`，实现 `meta_train(sampler)` 与 `adapt(task) -> Predictor`，加入 `metal/registry.py::REGISTRY` 与 `default_algorithms`。
- **新任务分布**：实现 `core/interfaces.TaskSampler`，返回满足 `core/types.Task` 契约（支持集须含全部类别）的任务。
- **新融合策略**：在 `metal/metafuse.py` 修改 `_fuse`，保持 `use_fusion` 守卫不变量：融合结果不得劣于最强成员 − `fuse_tol`。

---

© 2026 晨星. MIT License.

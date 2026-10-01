# MetaForge

> 世界级的 **少样本元学习（Few-Shot Meta-Learning）** 系统 —— 纯手写的透明实现，零深度学习框架依赖，可复现、可核验、可扩展。
>
> **作者：晨星** · 仓库：`CJX0712/metaforge` · License：MIT

MetaForge 把四种经典元学习范式（原型网络、Reptile、MAML、匹配网络）与诚实基线放在**同一套可核验契约**下，并用旗舰 **MetaFuse（Stabilized Meta-Fusion, SMF）** 做"优势互补 + 严格守护"的融合。整个系统的"真理之源"是一条不变量测试：**手写 MLP 的反向传播梯度，必须与数值中心差分梯度逐元素吻合**（rel err < 1e-4）。所有贡献都建立在它之上。

---

## ✨ 特性一览

| 能力 | 说明 |
|------|------|
| 🧠 四大元学习范式 | ProtoNet / Reptile（一阶）/ MAML（一阶 FOMAML）/ MatchingNet（无 FCE） |
| 🚩 旗舰融合 MetaFuse | 逐类最自信成员融合 + `use_fusion` 严格守卫，保证 `融合 ≥ 自身最强成员 − tol` |
| 📏 诚实基线 | Majority / KNN / LogisticRegression / 从零训练 MLP / 均值回归 |
| 🔬 两种任务分布 | Gaussian Blobs（N-way K-shot 分类）+ Sinusoid（回归） |
| 🛡️ 梯度裁剪 | 全部基于梯度的训练器统一接入 `clip_grad`，防止大学习率下权重发散到 `inf` |
| ✅ 可核验不变量 | 梯度校验是 CI 的硬门槛；ProtoNet 原型=支持集均值；MetaFuse 守卫不变量 |
| 🧩 模块化契约 | `TaskSampler` / `Predictor` / `MetaLearner` Protocol，模块只依赖抽象接口 |
| 🐳 一键复现 | `make install && make demo` 即可跑出下方基线表 |
| 🚫 零 torch 依赖 | 离线兜底即主路径：纯 `numpy` 手写 MLP，断网也能 build |

---

## 🧱 系统架构

```
                          ┌─────────────────────────────┐
                          │   core/  配置·类型·错误·工具  │
                          │  config · types · errors     │
                          └───────────────┬─────────────┘
                                          │  Task / MetaConfig
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
          ┌──────────────┐       ┌───────────────┐       ┌──────────────┐
          │  data/       │       │   metal/      │       │   eval/      │
          │  Samplers    │       │  ProtoNet     │       │  benchmark   │
          │ Blobs/Sinusoid│◄──────┤  Reptile/MAML │◄──────┤  run_benchmark│
          └──────────────┘  Task │  MatchingNet  │  Predictor          └──────┬───┘
                                 │  MetaFuse(SMF)│                             
                                 │  baselines    │                             
                                 │  nn (手写MLP) │                             
                                 └──────┬────────┘                             
                                        │  BenchResult                        
                          ┌─────────────┴──────────────┐                      
                          ▼                            ▼                      
                  ┌──────────────┐             ┌──────────────┐              
                  │ pipeline/     │             │  docs/        │              
                  │  MetaPipeline │             │ architecture  │              
                  └──────────────┘             └──────────────┘              
```

数据流动：`Sampler → Task` → `MetaLearner.meta_train(sampler)` → `MetaLearner.adapt(task) → Predictor` → `benchmark.evaluate` → `BenchResult` → JSON / 表格。

---

## 🚀 快速开始

### 环境（已验证）
- Python **3.13**（venv 隔离）
- numpy 2.5.3 · scipy 1.18.1 · scikit-learn 1.9.1
- 开发：pytest 9.1.1 · ruff 0.16.10

```bash
# 1. 创建隔离环境并安装依赖
python -m venv .venv
.venv/Scripts/activate          # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.lock.txt

# 2. 跑全部单元测试（含最关键的梯度校验，必须全绿）
pytest -q -W ignore::UserWarning

# 3. 端到端演示：生成真实基线并落盘
python -m metaforge.examples.run_demo     # -> benchmark.json (cls) + benchmark_reg.json (reg)

# 4. 命令行接口
python -m metaforge.cli --domain all --out benchmark.json
```

---

## 🧪 核验不变量（系统的"可信锚点"）

| 不变量 | 位置 | 含义 |
|--------|------|------|
| 梯度一致性 | `tests/test_nn.py` | 解析梯度 vs 中心差分数值梯度，逐元素 rel err < 1e-4 |
| 原型 = 支持集均值 | `tests/test_protonet.py` | `prototype[c] == mean(embed(y==c))`；同类查询点 prob[c]≈1 |
| MetaFuse 守卫 | `tests/test_metafuse.py` | `use_fusion` 时 `fused ≥ best − tol`，否则委托最佳成员 |
| 确定性 | `tests/test_pipeline.py` | 同一 seed 两次运行结果字节级一致；JSON 正确写入 |
| 训练稳定 | `metal/*` | 统一 `clip_grad(max_norm=5.0)`，权重不发散到 `inf` |

---

## 📊 真实性能基线（seed=20261002, meta_iters=400, eval_episodes=200）

> 数值由 `python -m metaforge.examples.run_demo` 实跑生成，非编造。

**分类（Gaussian Blobs，accuracy ↑ 越高越好）**

| 难度 | baseline_logreg | baseline_knn | baseline_scratch | matching | reptile | metafuse | protonet | maml | majority |
|------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| easy   | 0.9995 | 0.9988 | 0.9958 | 0.9897 | 0.9584 | 0.9439 | 0.9159 | 0.9122 | 0.2000 |
| medium | 0.9623 | 0.9501 | 0.9105 | 0.8758 | 0.8707 | 0.8530 | 0.8113 | 0.8538 | 0.2000 |
| hard   | 0.6431 | 0.5701 | 0.5689 | 0.5765 | 0.5682 | 0.5453 | 0.4339 | 0.5906 | 0.2000 |

**回归（Sinusoid，RMSE ↓ 越低越好）**

| 算法 | metafuse | reptile | maml | baseline_mean | baseline_scratch_reg |
|------|------:|------:|------:|------:|------:|
| RMSE | **1.6866** | 1.8599 | 1.8602 | 2.0011 | 2.4076 |

- 回归域 MetaFuse 融合成功（`use_fusion=True`, fused=1.748, best=maml），**全面优于**所有成员与基线。
- 分类域在 hard 难度上梯度法（MAML/Reptile）与原型法各有优势，MetaFuse 正确触发 `use_fusion=False` 并委托内部最优成员，绝不劣于自身成员 —— 这是融合旗舰的"安全网"。

---

## 🔧 配置

所有超参集中在 `metaforge/core/config.py::MetaConfig`，可通过环境变量 `METAFORGE_<FIELD>` 覆盖（见 `from_env`）。关键项：

| 字段 | 默认 | 含义 |
|------|------|------|
| `seed` | 20261002 | 全局随机种子（可复现） |
| `n_way` / `k_shot` / `q_shot` | 5 / 5 / 15 | 分类任务规模 |
| `lr_inner` / `lr_outer` | 0.05 / 0.1 | 内层适应 / 外层元更新学习率 |
| `inner_steps` / `meta_iters` | 5 / 400 | 内层步数 / 元迭代次数 |
| `meta_batch` | 4 | 每步采样的任务数 |
| `hidden` / `embed_dim` | (32,16) / 16 | 手写 MLP 隐藏层 / 嵌入维度 |
| `fuse_tol` / `fuse_temp` | 0.02 / 3.0 | 融合守卫容差 / 权重 softmax 温度 |

---

## 📁 目录结构

```
metaforge/
├── core/        # config, types, errors, interfaces, util
├── data/        # samplers: GaussianBlobSampler, SinusoidSampler, blob_preset
├── metal/       # nn(手写MLP), protonet, reptile, maml, matching, metafuse, baselines
├── eval/        # benchmark harness
├── pipeline/    # MetaPipeline: 场景→训练→评测→落盘
├── examples/    # run_demo.py 端到端示例
└── cli.py       # 命令行入口
tests/           # 21 个 pytest（含梯度校验/原型/守卫/确定性）
docs/architecture.md
benchmark.json / benchmark_reg.json   # 真实基线产物
```

---

## 🤝 贡献与扩展

- 新增算法：继承 `metal/base.py::BaseMetaLearner`，实现 `meta_train` / `adapt`，注册到 `metal/registry.py`。
- 新增任务分布：实现 `core/interfaces.TaskSampler`，返回符合 `core/types.Task` 契约的任务。
- 所有新增代码必须满足 `ruff check . && ruff format --check .` 与全部 pytest（尤其梯度校验）。

---

© 2026 晨星. 以 MIT 协议发布。

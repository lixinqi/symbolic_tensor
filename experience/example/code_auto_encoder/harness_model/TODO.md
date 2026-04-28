# Harness Model 后续工作清单

本文档列出 harness_model 在代码复原任务上的下一步工作项，按优先级与依赖关系排序。

---

## 1. 整理代码复原任务的训练集与评估集

**现状**：当前 `prepare_worktrees` 使用单一批次（`total_batch_size=1`）从 `codebase/` 中随机采样 mask 位置，训练和评估混在一起，无法衡量泛化能力。

**目标**：建立结构化的 train / eval split，使 loss 收敛可以被量化验证。

**具体工作**：
- 从 `experience/example/code_auto_encoder/codebase/` 中按文件粒度划分训练集（~80%）和评估集（~20%），确保同一源文件不同时出现在两个集合中。
- 为每个样本生成多种 mask 策略：
  - 短 mask（1~3 行）：测试局部上下文补全。
  - 长 mask（4~8 行）：测试跨行逻辑推理。
  - 结构性 mask（整段函数体 / 整段类方法）：测试长程依赖。
- 将 `prepare_worktrees` 改为接受 `split="train" | "eval"` 参数，从预生成的索引文件中读取样本，而不是运行时随机 mask。
- 建立评估指标：不仅看 `get_edit_distance_ratio` 的均值，还要统计 AST 通过率、Exact Match 率、以及各 mask 长度下的分层 loss。

**验收标准**：
- `test_harness.py --split eval` 能够输出与训练集隔离的、可复现的 loss。
- 同一评估集在模型迭代前后可以被重复跑分。

---

## 2. 打磨 harness_model 的前反向过程，使 loss 收敛

**现状**：当前 forward 能跑通端到端（seed 42 下 loss ~0.3164），但 backward（符号梯度生成 + `StSGD` patch）尚未在 harness_model 上完整验证。Loss 目前主要来自前向的 deterministic 校验器，而非训练后的模型改进。

**目标**：打通 harness_model 的完整训练闭环，使得多轮训练后 loss 在训练集和评估集上均显著下降。

**具体工作**：
- **前向**：
  - 把 `ft_recurrent` 的 `accumulate_output` 从 naive concat 改为带权重衰减的累积（如早期工具结果权重低，后期权重高），验证是否减少噪声累积。
  - 尝试在 context gather 阶段引入 `experience_tensor` 的 top-k 检索，用已有经验指导 bootstrap read 之外的工具选择（当前 top-k 只在代码生成阶段被使用）。
- **反向**：
  - 在 `HarnessModel` 中显式构造 `experience_tensor`（当前 `topk` 参数已传入但未见完整 experience 流水线）。
  - 验证 `ft_recurrent` 的 backward 是否正确传播符号梯度：外层 recurrent（context gather）的每次迭代、内层 recurrent（tool retry）的每次失败，都应该产生可 patch 的 diff。
  - 调优 `StSGD` 的 learning rate：符号梯度的 patch 粒度（整行 vs 字符级）对 loss 收敛速度影响巨大。
- **训练脚本**：
  - 写一个 `train_harness.py`，支持多 epoch、经验回放（把上一轮的 experience 作为下一轮的先验）。

**验收标准**：
- 在训练集上跑 3~5 个 epoch，mean loss 从 ~0.3 降至 <0.15。
- 评估集 loss 同步下降（无过拟合）。

---

## 3. 重构 grep / 工具使之具备跨文件检索的能力

**现状**：当前的 `HmGrep` 和 `HmGlob` 仅能在单个文件或当前目录下工作。当 mask 区域涉及跨文件引用（如调用其他模块的函数、使用全局变量）时，agent 无法通过一次工具调用定位到定义处，导致 context gather 不完整，generation 出现幻觉。

**目标**：让工具调用能跨越文件边界，使 agent 在收集上下文时具备类似 "跳转到定义" 的能力。

**具体工作**：
- **HmGrep 增强**：
  - 支持 `grep --include="*.py" "pattern" codebase/` 的递归搜索。
  - 返回结果中附带文件名 + 行号，便于后续 `read` 精确读取。
- **HmGlob 增强**：
  - 支持基于文件内容特征的过滤（如 "只返回包含 `class Foo` 的文件"），减少 glob 噪声。
- **新增 HmLocateDef（或扩展 HmGrep）**：
  - 给定一个符号名（如函数名、类名），在整个 codebase 中定位其定义位置。
  - 内部可先用 `grep` 做粗筛，再用简单的 `ast` 解析确认定义行号。
- **HarnessModel 集成**：
  - 在 `code_context_gather` 的 prompt 中，明确告诉 LLM "如果 grep 结果涉及其他文件，你可以继续 read 那个文件"。
  - 这相当于把 "跨文件跳转" 编码为外层 recurrent 的额外迭代（`c` 增加一次）。

**验收标准**：
- 在包含跨文件引用的 mask 样本上，context gather 能收集到定义所在文件的内容。
- 该类样本的 loss 从当前 >0.5 降至 <0.3。

---

## 4. 分析所沉淀的文本经验，明确其有效性

**现状**：框架通过 `StSGD` 将 LLM 生成的 diff 作为符号梯度 patch 到 `experience_tensor` 中。但目前不清楚这些沉淀下来的 (query, key, value) 三元组在后续检索中是否真的被命中、是否真的提升了翻译质量。

**目标**：建立对 experience tensor 的量化分析机制，区分"有效经验"和"无效噪声"。

**具体工作**：
- **命中率统计**：
  - 在每次 forward 的 `ft_moe` 阶段，记录 query 与 experience 中 top-k 的 Jaccard 相似度分布。
  - 统计：有多少次 top-k 结果是前向生成的、有多少次是 cold-start 随机选取的。
- **消融实验**：
  - 跑两组对比实验：
    - A 组：`topk=2`，正常检索 experience。
    - B 组：`topk=0`，关闭 experience 检索（纯 zero-shot）。
  - 对比两组的 loss，量化 experience 带来的绝对收益。
- **经验清洗**：
  - 设计一个阈值机制：如果某条 experience 的 value（翻译结果）在后续多次检索中从未被采纳，或其采纳后对应的 backward diff 始终为 0，则将其系数置为低权或删除。
- **可视化**：
  - 把 experience tensor 中高频 query 和对应 value 抽样打印出来，人工审查是否存在模式（如特定工具调用序列、特定错误类型）。

**验收标准**：
- 输出一份 report，列明 experience 的命中率、top-k 相似度分布、以及 A/B 实验的 loss 差异。
- 能给出至少 3 条被验证为"有效"的经验模式示例。

---

## 5. 整理数值梯度，确保它符合预期

**现状**：`SymbolicTensor` 拥有双通道梯度：符号通道（diff 文本）和数值通道（bfloat16 系数）。当前 harness_model 的核心逻辑（context sufficiency、tool result validation、AST pass/fail）都依赖 Status 编码到系数中，但尚未系统验证：
- 这些系数是否确实参与了 autograd 图？
- `StSGD` 更新后，系数的变化是否与预期方向一致？

**目标**：建立数值梯度的单元测试和可视化，确保 `data`（系数）这条通道不是"幽灵通道"。

**具体工作**：
- **系数追踪**：
  - 在 `ft_recurrent` 的每次迭代中，打印当前元素的系数值（`status.to_float()`），确认 confidence / scbf 的符号和大小符合设计。
  - 验证 `Status` 到 float 的映射没有溢出或精度问题（bfloat16 的数值范围有限）。
- **Autograd 连通性测试**：
  - 构造一个最小可复现的图：`input` → `ft_recurrent` → `loss = get_edit_distance_ratio`。
  - 调用 `loss.backward()`，检查 `input.grad` 是否非 None、形状是否正确。
  - 检查 `StSGD.step()` 之后，`experience_tensor.data`（系数部分）是否发生了数值更新。
- **数值 vs 符号解耦验证**：
  - 设计一个对照：仅更新数值系数（标准 SGD），不执行符号 patch，观察 loss 变化；再仅执行符号 patch，冻结数值系数，观察 loss 变化。
  - 明确两种更新各自对 loss 的贡献比例。
- **异常检测**：
  - 监控 `kConfidenceNotBounded`（-2.0）和 `kContextOverflow`（-3.0）的触发频率。如果大量出现，说明系数空间被特殊状态占满，需要调整 Status 的编码策略。

**验收标准**：
- 通过一组单元测试，证明数值梯度在 `ft_recurrent` → `loss` → `backward` → `StSGD.step()` 全链路中可流动、可更新。
- 输出数值系数在训练前后的分布变化 histogram。

---

## 优先级与依赖关系

```
1. 数据集拆分 ───────────────────────────────────────┐
     │                                              │
     ▼                                              ▼
2. 前反向打磨 ◄─────────────────────────────────── 3. 跨文件检索
     │                                              │
     ▼                                              │
4. 经验有效性分析 ◄──────────────────────────────────┘
     │
     ▼
5. 数值梯度整理
```

- **1 是基础**：没有稳定的 train/eval split，无法判断 2~5 的改进是否真实有效。
- **2 和 3 可并行**：前反向打磨侧重训练机制，跨文件检索侧重工具能力，两者独立。
- **4 依赖 2**：只有在训练能跑起来、experience 能沉淀之后，才能分析其有效性。
- **5 是兜底**：在所有功能稳定后，确保底层的数值通道不是虚设的。

---

*最后更新：2026-04-28*

# TODO #1：整理训练集与评估集

## 按文件粒度划分 train / eval split

**实现**：

从 `codebase/` 目录按文件粒度 80/20 划分，确保同一文件不同时出现在两个集合中。

- 共 65 个源文件，随机打乱后取前 52 个为训练集，后 13 个为评估集。
- 划分结果持久化到 `dataset_index/train/samples.json` 和 `dataset_index/eval/samples.json`。
- 每个文件生成三类 mask 样本：
  - **short**（1-3 行）：测试局部上下文补全
  - **long**（4-8 行）：测试跨行逻辑推理
  - **structural**（整段函数体）：测试长程依赖
- 最终生成 332 条 train 样本 / 82 条 eval 样本。

**关键代码**：`prepare_worktrees.py` 新增 `split` 参数，`"train" | "eval"` 时从索引文件读取，`None` 时走原有随机 mask 路径。

**验证**：

```bash
python -m experience.example.code_auto_encoder.harness_model.test_harness \
    --split eval --total-batch-size 5 --seed 42 --llm-method raw_llm_api
```

同一 seed 下结果完全可复现，与训练集无交叉。

**提交**：`0ce53bd`

---

## 新增评估指标

在 `test_harness.py` 的 `_compute_extra_metrics` 中增加：

- **AST 通过率**：用 `ast.parse(textwrap.dedent(actual))` 检查语法合法性（需 dedent，否则顶层缩进代码会误报 SyntaxError）。
- **Exact Match 率**：逐字符比对输出与 ground truth。
- **按 mask 类型分层的 loss**：分别统计 short / long / structural 三类样本的均值 loss。

**验收状态**：`test_harness.py --split eval` 能输出与训练集隔离、可复现的完整指标报告。

# TODO #2：打磨 harness_model 的前反向过程，使 loss 收敛

## 前向：weighted accumulate_output

**背景**：

原 `_concat_context` 将所有 `read()` 结果等权拼接。对大文件（>200 行）且 mask 在 200 行之后的样本，bootstrap read（前 200 行）的大量无关内容会压缩后续有针对性 read 的占比，导致 context 中靠近 mask 区域的行比例偏低。

**实现**：

- 提取 `_extract_clean_read(cur)` 辅助函数，过滤非 read 成功结果；`_concat_context` 和新函数复用同一逻辑。
- 新增 `_concat_context_weighted`：给每步 read 加 `[Step N]` 标签，超出字符预算（`_CONTEXT_BUDGET_CHARS=12000`）时进行裁剪——保留**首块**（bootstrap，含类结构）和**末块**（最新精准读）完整，将中间块截断至 `_MIDDLE_PIECE_KEEP_CHARS=500` 字符。
- `HarnessModel.__init__` 新增 `accumulate_mode: str = "naive"` 参数，`forward` 中按参数选择累积函数；新增 `last_context_tensor` 属性供后续分析使用。
- `test_harness.py` 新增 `--accumulate-mode {naive,weighted}` CLI 参数；新增 `_compute_noise_metric`（解析 context 中的行号，计算 mask ±30 行窗口内的相关性比率）；在 `_compute_extra_metrics` 中打印 `mean context_len` 和 `relevance ratio`。

**验证方式**：

选取 `st_moe_backward.py`（808 行）中 `mask_start ∈ {218, 324, 355, 458}` 的 4 个深层样本，这类样本 bootstrap read（offset=0, limit=200）无法覆盖 mask 区域，必须触发多步 read，是 weighted 和 naive 产生差异的典型场景。

在真实 LLM（GLM-5-Turbo）下跑两轮对比：

| 指标 | naive | weighted |
|------|-------|----------|
| mean loss | 0.5571 | **0.4324**（↓22.4%）|
| mean relevance | 6.15% | **8.42%** |

代表性案例（`mask_start=458`）：naive 因 LLM 工具调用格式错误导致 context 完全缺失相关内容（relevance=0%，loss=0.7734）；weighted 借助 `[Step N]` 标签引导 LLM 正确补读，relevance 升至 9.09%，loss 降至 0.3750。

**注意**：

- 当前 eval 集（82条，`mask_start ≤ 181`，均在 bootstrap 覆盖范围内）两种模式结果相同，差异仅在大文件深层 mask 样本上体现。
- 顺带修复了三处 bug（`75508ff`）：`raw_llm_query` 429 指数退避重试、`code_gen` 缩进归一化、`test_harness` AST 通过率使用 `textwrap.dedent` 后再解析。

**提交**：`8be49aa`（weighted 实现）、`75508ff`（三处 bug fix）

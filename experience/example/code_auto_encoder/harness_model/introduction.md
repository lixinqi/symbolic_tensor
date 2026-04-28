# Experience 框架导论

本文档面向希望理解 `harness_model` 的开发者，从底层的 `symbolic_tensor` 一直讲到上层的 `harness_model`，说明整个框架的"为什么"和"怎么做"。

---

## 一、SymbolicTensor：把文件系统当成张量存储

### 1.1 核心思想

`symbolic_tensor` 是 PyTorch 的一个扩展。普通 PyTorch 张量的元素是浮点数；而 SymbolicTensor 的每个元素是**磁盘上的一个文本文件**。这意味着：

- 张量元素可以是任意文本：Python 代码、自然语言、Viba DSL、diff 补丁……
- 张量仍然保留一套**数值系数**（bfloat16），和标准 autograd 兼容。
- 张量操作（slice、stack、merge 等）操作的是**文件路径和符号链接**，而不是内存中的数值。

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # flat index 0 的内容
│   ├── 1/data               # flat index 1 的内容
│   └── 1/1/data             # flat index 11 的内容
```

### 1.2 SymbolicTensor 的定义与关键字段

SymbolicTensor 本质上就是一个普通的 `torch.Tensor`，但通过 monkey-patch 附加了自定义属性。关键字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `st_relative_to` | `str` | 存储根目录。所有张量文件都存放在这个目录下。 |
| `st_tensor_uid` | `str` | 张量的唯一标识符（UUID）。`{st_relative_to}/{st_tensor_uid}/` 就是该张量的完整存储路径。 |
| `data` | `torch.Tensor` | 数值系数（bfloat16），和标准 PyTorch 完全兼容，可参与 autograd。 |
| `shape` | `torch.Size` | 张量形状。每个元素对应磁盘上的一个文件。 |

每个元素在磁盘上的存储路径规则：

```
{st_relative_to}/{st_tensor_uid}/storage/{flat_index}/data
```

其中 `flat_index` 的十进制数字被拆成单级目录。例如 flat index 11 的路径是 `storage/1/1/data`，flat index 123 的路径是 `storage/1/2/3/data`。

框架通过 `register_tensor_ops.py` 给 `torch.Tensor` 注册了一批符号操作方法：

| 方法 | 作用 |
|---|---|
| `st_assign(rvalue)` | 把 rvalue 的符号内容拷贝到本张量 |
| `st_assign_view(rvalue)` | 通过符号链接创建视图（共享存储） |
| `st_get_diff(rvalue)` | 计算本张量与 rvalue 的 unified diff |
| `st_patch(rvalue)` | 把 rvalue 中的 diff 应用到本张量 |
| `st_view_slicer[...]` | Pythonic 切片，返回 symlink 视图 |
| `st_value_slicer[...]` | Pythonic 切片，返回独立拷贝 |

### 1.3 双通道梯度系统

SymbolicTensor 的梯度传播有两个通道：

| 通道 | 携带内容 | 计算方式 |
|---|---|---|
| **数值通道** (coefficient) | 浮点值 (bfloat16) | 标准 autograd / SGD 算术 |
| **符号通道** (symbolic) | Unified diff 文本 | LLM 计算 `diff -u`，存在文件中 |

前向时，LLM 读取输入文件，写入输出文件；反向时，LLM 计算"如何把输出改得更好"，生成 diff；优化器 `StSGD` 用 `patch` 命令把 diff 应用到参数（即 Experience）上。

### 1.4 ExperienceTensor

`ExperienceTensor` 是一个形状为 `[N, 3]` 的 SymbolicTensor，每行是一个 `(query, key, value)` 三元组：

- **Query** (位置 0)：语义关键词，用于 Jaccard 相似度检索
- **Key** (位置 1)：源域内容（如 Python 代码）
- **Value** (位置 2)：目标域内容（如 Viba 代码）

它是模型的**可学习权重**——训练开始时是空的，每一轮通过 patch 增量更新。模型不是"记住"训练数据，而是**在运行时积累 experience**。

### 1.5 Autograd Function

SymbolicTensor 的核心操作都封装在 `torch.autograd.Function` 中：

| Function | 作用 |
|---|---|
| `StMoe` | 混合专家：query 生成 → 检索 top-k → LLM 翻译 → 回写 |
| `st_attention` | 注意力操作（slice_attention + merge） |
| `st_stack` | 沿新轴堆叠符号张量 |
| `slice_view` | 通过符号链接创建视图（共享存储） |
| `slice_tensor` | 创建独立拷贝 |
| `GetEditDistanceRatio` | 损失函数：Levenshtein 编辑距离比 |

---

## 二、FutureTensor：前向异步 + 循环，反向异步 + 并发

### 2.1 为什么需要 FutureTensor？

SymbolicTensor 的前向是**同步阻塞**的：每个 autograd Function 内部直接调用 LLM，等结果返回才继续。这在训练循环里没问题，但在**agent 编排**场景下不够灵活：

- Agent 需要**多轮 retry**（生成 → 验证 → 失败 → 重试）。
- Agent 的每一步应该**异步执行**，且循环次数应该**编码到张量维度**里。
- 反向传播时，需要**并发地**让 LLM 反思所有失败的尝试。

FutureTensor 解决了这些问题。

### 2.2 FutureTensor 的定义

```python
FutureTensor[shape list[int]] :=
    SymbolicTensor[shape]
    * ft_forwarded bool          # 是否已经物化
    * ft_forward(prompt_tensor)  # 物化函数
    * ft_async_get(coordinates, prompt) -> (str, Status)  # 异步元素生成器
```

- `ft_async_get` 是**纯异步回调**：给定坐标和 prompt，返回 `(内容, 状态)`。
- `ft_forward` 是**物化触发器**：传入一个同形状的 prompt_tensor，对所有元素并发调用 `ft_async_get`，把结果写回 SymbolicTensor 的存储文件。
- 一旦 `ft_forwarded=True`，再次调用 `ft_forward` 会短路返回（幂等）。

### 2.3 前向：异步 + 循环

FutureTensor 的前向有两个核心特征：

**1) 异步并发**

`ft_forward` 内部用 `asyncio.gather` 并发执行所有 `ft_async_get`：

```python
async def _gather():
    tasks = [self.ft_async_get(coords, prompt) for coords, prompt in ...]
    return await asyncio.gather(*tasks)
```

这意味着一个 `[batch=8, retries=5]` 的张量，40 个元素可以同时向 LLM 发请求。

**2) 循环编码到维度——因为 LLM 是概率计算模型**

`ft_recurrent` 的设计出发点是：**LLM API 不是确定性的矩阵乘法，而是一个概率计算模型**。每次调用都有可能产生错误、幻觉或格式不合法的输出。因此必须有一个机制来：

- **反复确认**：生成结果 → 用 validator 检查 → 失败则重试。
- **筛选过滤**：在多次尝试中选出置信度最高的成功结果，或退而求其次选"最接近成功"的失败结果。
- **累积上下文**：把之前失败的 trace 保留下来，作为下一轮 retry 的 prompt，让 LLM 知道"上次错在哪"。

`ft_recurrent` 把这套"生成-验证-重试"的**for 循环编码到张量的最后一个维度**。输入形状是 `(*prefix_dims, recurrent_dim)`，输出形状是 `(*prefix_dims)`。

对每个前缀坐标，它迭代 `i in range(recurrent_dim)`：
- 调用 `input.ft_async_get([*prefix, i], prompt_i)`
- 拿到 `(output_i, status_i)`
- 如果 `status.is_confidence`：立即返回（成功退出）
- 如果失败：把 `output_i` 追加到 prompt，进入下一轮

```
input  [batch, retries]  --ft_recurrent-->  output [batch]
                              prompt_tensor [batch, retries]
```

`accumulate_output` 参数允许跨迭代累加输出（如把每轮的 read 结果拼接成完整上下文）。

### 2.4 反向：异步 + 并发

`ft_recurrent` 是一个 `torch.autograd.Function`。反向时：

- **输入已经是物化的 SymbolicTensor**（forward 结束后 `ft_forward` 已经跑过）。
- backward 不再用 FutureTensor 的惰性语义，而是直接操作 SymbolicTensor 的存储文件。
- 对所有需要反思的元素，**并发构造 AgentTask**，通过 `TaskHandler()` 一次性提交给 LLM。
- LLM 对每个元素写出改进后的内容，框架计算 `diff`，存为梯度。

这就是"**前向是 lazy sequential retry，反向是 eager concurrent reflection**"的精髓。

### 2.5 Status：把控制流编码到张量系数

FutureTensor 用 `Status` 把"成功/失败/重试"的控制流编码到张量的数值系数中：

| Status | 含义 | 存为 float |
|---|---|---|
| `confidence(v)` | 成功，置信度 v | `+v` |
| `self_confidence_but_failed(v)` | 失败，自评置信度 v | `-v` |
| `kConfidenceNotBounded` | 置信度越界 | `-2.0` |
| `kContextOverflow` | 上下文太长 | `-3.0` |

`ft_recurrent` 根据系数的正负判断是成功还是失败，从而决定是否提前退出循环。

---

## 三、Future Op：循环和 trace 是如何编码的

### 3.1 ft_recurrent：for 循环 = 最后一个维度

```python
# 输入形状 [batch, retries]
# 含义：batch 个样本，每个样本最多 retries 次尝试
ft_input = FutureTensor([batch_size, max_retries], tmpdir, ft_async_get)
output, prompt_tensor = ft_recurrent(ft_input, accumulate_output=concat_fn)
# output 形状 [batch_size]
# prompt_tensor 形状 [batch_size, max_retries]
```

循环完全编码在维度里：
- **循环变量 `i`** = 最后一个维度的坐标。
- **循环次数** = 最后一个维度的大小。
- **trace（prompt 累积历史）** = `prompt_tensor` 中每个 `[*prefix, i]` 位置的文件内容。

`ft_async_get` 的签名 `(coordinates, prompt)` 就是 trace 的载体：
- `coordinates` 告诉你当前是第几轮 retry。
- `prompt` 是前面所有轮次的输出拼接而成的累积上下文。

**镂空支持变长循环**：如果某轮 `status=confidence`，`ft_recurrent` 提前退出，后面的维度位置不会被访问（系数保持为 0，内容保持为 TODO）。

### 3.2 ft_moe：把 MoE 编码到元素级异步调用

```python
ft_input = FutureTensor([batch_size], tmpdir, ft_async_get)
output, prompt_tensor, indexes = ft_moe(
    ft_input, experience_tensor, topk=16
)
# output 形状 [batch_size]
```

每个元素的 `ft_async_get` 内部执行完整的 MoE 流程：
1. 用输入内容生成 query
2. 从 experience 中检索 top-k
3. 构造 workspace，调用 LLM
4. 返回翻译结果

这里 **batch 维度就是 for 循环**：8 个 batch 元素 = 8 次独立的 MoE 调用，它们在 `ft_forward` 时并发执行。

### 3.3 ft_unary：纯函数映射

```python
def ft_unary(input_ft, fn) -> FutureTensor:
    async def wrapped(coords, prompt):
        output, status = await input_ft.ft_async_get(coords, prompt)
        return fn(coords, prompt, output, status)
    return FutureTensor(input_ft.shape, input_ft.st_relative_to, wrapped)
```

`ft_unary` 不改变形状，只改变每个元素的 `(output, status)`。它是 FutureTensor 的**函数式组合**基础构件。

### 3.4 维度即控制流的总结

| 概念 | 传统代码 | FutureTensor 编码 |
|---|---|---|
| for 循环 | `for i in range(R)` | 最后一个维度 `R` |
| 嵌套循环 | 外层 for + 内层 for | 形状 `[..., C, R]` |
| 循环变量 `i` | 局部变量 | `coordinates[-1]` |
| trace / 历史 | 字符串拼接 | `prompt_tensor` 的存储文件 |
| 提前退出 | `break` | `status=confidence` |
| 失败重试 | `continue` | `status=scbf` + prompt 累积 |

---

## 四、多层 Future Op 的列式执行：与 Symbolic Op 的本质差异

### 4.1 Symbolic Op 的前向是"行式"的

SymbolicTensor 的 autograd Function（如 `StMoe`）的前向是**同步、行式（row-wise）**的：

```python
# st_moe_forward 内部
for each_batch_element:
    query = llm_generate_query(input[b])
    topk = select_experience(query)
    workspace = build_workspace(topk)
    output[b] = llm_translate(workspace)   # 阻塞等待 LLM
```

一个 Function 调用内部，用 Python 的 `for` 循环遍历 batch 维度。每个元素的处理是**串行**的（虽然可以用 `TaskHandler` 把 batch 内的请求并发化，但从控制流角度看，Function 的语义是"一次性把整张张量算完"）。

行式执行的特点：
- **同一层算子**先把所有 batch 元素处理完，才进入下一层算子。
- 无法表达"本元素先走完 A→B→C，再处理下一个元素"的**跨层列式流水线**。
- retry 逻辑必须写在 Function 内部，无法用张量维度表达。

### 4.2 多层 Future Op 的前向是"列式"的

FutureTensor 的 `ft_forward` 物化机制，在多层 future_op 叠加时，会产生一种**列式（column-wise）**执行序列。

考虑 harness_model 中的 Stage 1 流水线：

```
ft_raw [batch, C, R]
  → ft_unary(validate)          [batch, C, R]
  → ft_recurrent(inner)         [batch, C]     # 内层：重试工具调用
  → ft_unary(sufficiency)       [batch, C]
  → ft_recurrent(outer)         [batch]        # 外层：累加上下文
```

当调用 `context_ft.ft_forward(prompt_tensor)` 时，`context_ft` 的形状是 `[batch]`，`ft_forward` 需要为每个 `b` 获取结果。但对每个 `b`，它触发的不是单次 LLM 调用，而是**一整条嵌套的 future_op 链**。

实际执行时序如下（以 batch 中第 0 个元素为例）：

```
# 外层 recurrent 处理 b=0，开始迭代 c=0,1,2...
outer[0]:
  c=0:
    # 内层 recurrent 处理 (b=0,c=0)，开始迭代 r=0,1,2...
    inner[0,0]:
      r=0: ft_unary(validate) -> ft_raw.ft_async_get([0,0,0]) -> LLM(bootstrap read)
             ↓ 返回 (trace, scbf)
      r=1: ft_unary(validate) -> ft_raw.ft_async_get([0,0,1]) -> LLM(grep)
             ↓ 返回 (trace, scbf)
      ...
      r=k: 内层某个 r 返回 confidence -> inner 退出，输出 accumulator
    ↓
    ft_unary(sufficiency) 检查 (prompt + accumulator)
    -> 返回 confidence（上下文已足够）
  ↓
  outer 收到 confidence，把结果写入 accumulate_output
  -> outer 退出，b=0 完成

# 接着处理 b=1
outer[1]:
  c=0:
    inner[1,0]:
      r=0: ...
      ...
```

可以看到，执行顺序不是：
```
❌ 先算完 ft_raw 所有元素 → 再算 ft_unary 所有元素 → 再算 ft_recurrent 所有元素
```

而是：
```
✅ (ft_raw[0,0,0] → ft_unary[0,0,0] → ... → inner[0,0] → sufficiency[0,0] → outer[0])
   完成后，再处理下一个 batch 元素。
```

**这就是列式执行**：对每一个"列"（即每一个外层坐标 `b`），完整执行 `fop0[0] → fop1[0] → ... → fopn[0]` 整条链，拿到最终结果后，再进入下一列 `fop0[1] → fop1[1] → ...`。

### 4.3 为什么列式执行对 Agent 编排至关重要

Symbolic Op 的行式执行假设每一层都是"无状态矩阵运算"——同一层的所有元素可以独立并行，层与层之间只有数据依赖。但 **Agent 编排不是矩阵运算**，它有以下几个特性：

**1) Agent 有内部状态（trace / prompt 累积）**

Agent 的每一步都依赖于之前所有步骤的 trace。在 harness_model 中：
- 内层 recurrent 的 `prompt_tensor` 记录了 `r=0` 的失败原因，作为 `r=1` 的输入。
- 外层 recurrent 的 `accumulate_output` 记录了 `c=0` 的 read 结果，作为 `c=1` 的上下文基础。

如果像 Symbolic Op 那样行式执行，这些 trace 必须显式地在层与层之间传递，而且 retry 循环只能硬编码在 Python 里。

**2) Subagent 必须在 Agent 内部完成**

 harness_model 的"收集上下文"是一个 Agent 任务，而"调用 read/grep 工具"是这个 Agent 的 Subagent 任务。Subagent 可能失败（LLM 输出了无效工具名），需要 retry。在列式执行中：

```
Agent (outer recurrent) 决定"下一步收集什么上下文"
  -> 启动 Subagent (inner recurrent) 去执行工具调用
     -> Subagent 内部 retry 直到拿到有效结果
     -> Subagent 返回结果给 Agent
  -> Agent 用 validator 检查上下文是否足够
  -> 不够？Agent 进入下一轮 c+1，再启动一次 Subagent
```

这种"Agent 启动 Subagent，Subagent 完成后再回到 Agent"的嵌套结构，在列式执行中**天然被维度编码**：
- Agent 的 step = 外层 recurrent 的坐标 `c`
- Subagent 的 retry = 内层 recurrent 的坐标 `r`
- Subagent 的 trace = 内层 `prompt_tensor` 的存储文件
- Agent 的 trace = 外层 `accumulate_output` 的累积结果

**3) ft_forward 提供统一的物化调度**

当 `context_ft.ft_forward(prompts)` 被调用时，框架不需要知道"这条链上有几层 recurrent、几个 unary"。它只需要：
- 遍历 `context_ft` 的所有坐标（外层维度 `b`）
- 对每个坐标，调用 `ft_async_get`
- 被调用的 `ft_async_get` 是 `ft_recurrent(outer)` 的，它内部会调用 `ft_unary(sufficiency)` 的
- `ft_unary` 内部又会调用 `ft_recurrent(inner)` 的
- 层层嵌套，直到最底层的 `ft_raw` 真正调用 LLM

整个调度是**递归惰性的**，由 `ft_forward` 的并发遍历 + `ft_recurrent` 的循环控制 + `ft_unary` 的纯函数映射共同完成。

### 4.4 对比总结

| 特性 | Symbolic Op（行式） | 多层 Future Op（列式） |
|---|---|---|
| 执行粒度 | 一层算子做完所有 batch | 一个 batch 元素走完所有层 |
| 循环表达 | Python `for` / `while` | 张量维度 `recurrent_dim` |
| trace 传递 | 显式传参或全局变量 | `prompt_tensor` 和 `accumulate_output` 隐式存储 |
| retry 机制 | 硬编码在 Function 内 | `ft_recurrent` 自动处理 |
| Agent/Subagent | 难以结构化表达 | 外层/内层 recurrent 天然对应 |
| 调度方式 | 同步阻塞 | `ft_forward` 异步并发调度 |

---

## 五、HarnessModel：用 FutureTensor 模拟 Claude Code

### 5.1 目标

`harness_model` 是一个**纯框架内实现的 Claude Code 模拟器**：
- 不调用 Claude Code CLI。
- LLM 只通过显式的工具调用（read/grep/glob）获取代码库上下文。
- 所有执行 trace 都存入 experience tensor，可供后续学习。

### 5.2 两阶段架构

```
worktree_tensor [batch]
    │
    ▼
Stage 1: code_context_gather
    ft_raw [batch, C, R]
        → ft_unary(validate_tool_result)
        → ft_recurrent(inner)  [batch, C]      # 内层：重试直到拿到有效工具结果
        → ft_unary(check_context_sufficiency)
        → ft_recurrent(outer, accumulate_output) [batch]  # 外层：累加干净上下文
    │
    ▼
context_tensor [batch]
    │
    ▼
Stage 2: code_gen
    ft_gen [batch, L]
        → ft_recurrent         [batch]          # 生成代码 → 语法验证 → 重试
    │
    ▼
output_tensor [batch]
```

- **C = `max_context_collects`**：最多收集几轮上下文。
- **R = `max_tool_call_retries`**：每轮工具调用最多重试几次（LLM 可能输出无效工具名）。
- **L = `max_codegen_steps`**：代码生成最多重试几次（语法验证失败时重试）。

### 5.3 关键设计

**1) Bootstrap read**

第一轮 (`c=0, r=0`) 不经过 LLM，直接强制读取目标文件 `offset=0, limit=200`。这是最大的性能优化——没有 bootstrap 时，LLM 会浪费 retry 在 tiny 的 read window 上，loss 会从 ~0.2 暴涨到 ~0.7+。

**2) Clean accumulator**

`accumulate_output=_concat_context` 只保留 `[read(...)]` 的结果，丢弃 `[grep(...)]` 和 `[glob(...)]` 的噪声。执行 trace 存入 experience；干净的 accumulator 喂给 generation。

**3) 语法验证 + 缩进搜索**

代码生成后，把候选代码插入到实际 masked 文件的位置，运行 `ast.parse()`。如果失败，尝试 0/2/4/6/8/10/12 空格的缩进，再解析。这解决了"snippet 本身合法，但相对缩进不对"的问题。

**4) 上下文验证器**

不是让 LLM 自己说"够了"，而是用一个**确定性验证器**检查：
- 目标文件是否已经被 read 过？
- read 结果是否包含 mask 区域前后至少 2 行代码？

满足才发 `confidence`，否则继续收集。

### 5.4 Pipeline 风格的前向

```python
def forward(self, worktree_tensor):
    # Stage 1
    ft_raw = FutureTensor([batch, C, R], tmpdir, self._make_tool_use(worktree_tensor))
    ft_validated = ft_unary(ft_raw, self._validate_tool_result)
    ft_tool_result, _ = ft_recurrent(ft_validated)           # 内层 recurrent
    ft_checked = ft_unary(ft_tool_result, self._check_context_sufficiency)
    context_ft, _ = ft_recurrent(ft_checked, accumulate_output=_concat_context)  # 外层 recurrent

    # Materialize
    context_ft.ft_forward(make_tensor(["gather"] * batch, tmpdir))
    context_tensor = context_ft._tensor

    # Stage 2
    ft_gen = FutureTensor([batch, L], tmpdir, self._make_code_gen(context_tensor, worktree_tensor))
    output_ft, _ = ft_recurrent(ft_gen)
    output_ft.ft_forward(make_tensor(["generate"] * batch, tmpdir))
    return output_ft._tensor
```

整个 forward 是一个**声明式数据流**：FutureTensor → ft_unary → ft_recurrent → ... → SymbolicTensor。

### 5.5 验证结果

在 `symbolic_tensor/tensor_util/assign_tensor.py:13-18` 的 mask 恢复任务上：
- Loss: **0.3164**（seed 42，deepseek-chat）
- Stage 1: bootstrap read → grep 相关文件 → context sufficient
- Stage 2: 一次生成通过 AST 语法检查

---

## 六、总结

| 层次 | 核心抽象 | 关键特征 |
|---|---|---|
| **SymbolicTensor** | 文件系统即张量存储 | 双通道梯度、patch-based 优化、ExperienceTensor |
| **FutureTensor** | 惰性异步张量 | 前向 lazy sequential、反向 eager concurrent、Status 控制流 |
| **Future Op** | 维度即控制流 | for 循环 = 最后一个维度、trace = prompt_tensor、嵌套 recurrent |
| **HarnessModel** | Agent 编排 | 两阶段 pipeline、bootstrap read、clean accumulator、语法验证 |

整个框架的哲学是：**把 agent 的循环、分支、trace 全部编码到张量的形状和存储中**，而不是隐藏在 Python 的控制流里。这让 agent 的行为可以被框架调度、被 autograd 追踪、被 optimizer 学习。

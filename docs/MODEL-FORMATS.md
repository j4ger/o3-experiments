# 模型格式与运行路径

本文合并了"ONNX 之外有没有更合适的模型格式"和"跑任意架构模型有哪些落地路径"两份分析。

---

## 第一部分：格式选型

## 先重新框定问题

"选模型格式"实际上是在选 **"谁来生成 NPU 能执行的图"**。这颗 NPU 只认：

```
mida graph / subgraph  ← xnnl 算子（matmul / RMSNorm / quantize）
                       ← 或 Acuity vsi_nn_* / NBG
```

所以真正的问题是：**从"一个模型"到"一张 mida 图"，哪条路最短。**
格式本身只是外壳。

## 1. 候选对比

| 方案 | 前端格式 | 谁生成 NPU 图 | 任意架构 | 工作量 | 与这颗 NPU 的适配 |
|---|---|---|---|---|---|
| **ONNX + ORT EP** | ONNX（protobuf，~200 算子） | 我们写 ExecutionProvider | ✅ 最强 | 大 | 通用，但大部分算子用不上 NPU |
| **GGUF + ggml backend** | GGUF（单文件，量化内建） | 我们写 ggml backend | LLM 架构（llama.cpp 已内置多种） | **小** | **最贴合：NPU 强项就是 matmul** |
| TFLite | FlatBuffers | 无可用 NPU delegate（NNAPI 无驱动） | 有限 | 中 | 差 |
| MNN | ONNX/MNN | MNN backend | 较好 | 中 | 中（多一个大型依赖） |
| 厂商格式 `xnn.plan` | YMCA FlatBuffers | 厂商编译器（设备上没有） | ❌ | 无解 | 原生但不可控 |
| 自建 mini-IR | 自定义 JSON/二进制 | 我们的 o3npu 驱动 | 仅限我们实现的块 | 中 | 完全可控 |

## 2. 为什么 GGUF 比 ONNX 更合适

**（a）NPU 的强项决定了模型类型**

`xnnl` 目前确认的算子：`MatMul` / `RMSNormalization` / `Quantize`。
这是 **LLM 的形状**，不是 ResNet 的形状。既然如此，选一个 LLM 原生格式，
比选一个通用图格式再从中挑出 LLM 部分更直接。

**（b）llama.cpp 的后端接口天生支持混合执行**

```c
// ggml/include/ggml-backend.h
bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor *op);
bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor *op);

ggml_backend_t backends[] = { backend_npu, backend_cpu };
ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, ...);
ggml_backend_sched_graph_compute(sched, graph);
```

- 我们只实现 `supports_op` 返回 true 的算子（先只认 `GGML_OP_MUL_MAT`）；
- 其余算子由 scheduler 自动交给 CPU，**不需要实现全部算子**；
- 和 ORT EP 是同一个设计模型，但 ggml 的算子集小得多、纯 C、无 protobuf。

**（c）量化格式已经对齐**

GGUF 内建 `Q4_0`/`Q4_K` 等 int4 量化，和我们已解出的
"行主序 int4 + per-channel scale × 6.81" 是同一类东西，转换成本低。

**（d）交叉编译简单**

llama.cpp 是 CMake + 纯 C/C++，直接 `-DCMAKE_TOOLCHAIN_FILE=$NDK/...` 就能出
arm64-v8a 二进制，不需要 Python、不需要 protobuf 工具链。

**（e）"任意架构"的边界清楚**

GGUF 支持的是"llama.cpp 已实现的架构"（Llama/Qwen/Gemma/Phi…）。
这比"任意 ONNX"窄，但比"厂商支持的那几个模型"宽得多，
而且**新增架构只需要在 llama.cpp 里加模型定义**，不用碰 NPU 侧。

## 3. 什么时候仍然该用 ONNX

- 目标不是 LLM（检测、分割、语音、embedding 之外的自定义图）；
- 需要跨框架（PyTorch/JAX/TF）导出的通用图；
- 想把 NPU 当"ORT 的一个加速后端"来评估，而不是专攻 LLM。

结论：**两条路不冲突**——GGUF 是 LLM 主力，ONNX 是通用兜底。

## 4. 推荐

1. **主力：GGUF + ggml 的 NPU backend**
   - 先实现 `GGML_OP_MUL_MAT`（Transformer 的主要算力）；
   - 用 llama.cpp 的 CPU backend 当数值基准；
   - 逐步加 `RMS_NORM` / `ADD` / `MUL` / `ROPE`。
2. **兜底：ONNX + ORT EP**（当需要非 LLM 架构时）
3. **对拍：官方 `xnn.plan` / `.bell`**
   - 用设备上已有的模型当"金标准"，验证我们的 backend 算得对。

## 5. 参考

- `ggml/include/ggml-backend.h`（`supports_op` / `ggml_backend_sched_*`）
- 我们的驱动：`tools/npu/o3npu.{h,c}`
- NPU 侧算子：`docs/NPU-ACCESS.md`、`docs/generated/xnnl_api.h`
- 容器格式：`docs/FORMAT-xnlm.md`


---

## 第二部分：ONNX 与各条运行路径

结论先给：**ONNX 本身没问题，问题在于"谁来把 ONNX 图翻译成 NPU 能执行的图"。**
这台设备上不存在 ONNX 导入器，但存在两条可用的翻译目标，其中一条可以完全由我们掌控。

## 1. 设备上实际存在的三条 NPU 执行路径

| 路径 | 格式 / API | 设备侧能编译吗 | 可访问性 |
|---|---|---|---|
| **Xiaomi Xnrt** | `xnn.plan`（FlatBuffers，魔数 `YMCA`），运行时 `libxnn_runtime.so` 的 `Xnn_*` API；封装 `libxnnapi.so` 的 `XRingXnnApi`（`initVppXnn` / `SetTensorAddress` / `Run`） | ❌ 编译器不在设备上 | 库可读 |
| **Vivante Acuity / NBG** | `vsi_nn_*` 图 API（`libovxlib.so`，5.7 MB）+ `nbg_*`（`libNBGlinker.so`）+ VXC 二进制 | ✅ `vsi_nn_GenerateNBG` 在设备上 | `libovxlib`/`libNBGlinker` 是 `same_process_hal_file`（可读）；但公共入口 `libOpenVXU.so` **连 root 都读不了** |
| **我们自己的 mida 直驱** | `o3npu`（`libxnumd` → `/dev/hpc-cdev`） | — | ✅ 已验证可用 |

证据：
- `libxnn_runtime.so` 里有 `Failed to read xnn.plan`、"Topo order of subgraphs retrieved from xnn.plan"、`Xnn_ContextEnqueue/Execute/GetTensorAddress/SetInputShape`。
- `libxnnapi.so`（35 KB）只 dlopen `libxnn_runtime.so`，是"加载 plan → 设张量地址 → Run"的薄封装。
- `libovxlib.so` 导出完整 `vsi_nn_*`（`AddNode`/`CreateTensor`/`SetupGraph`/`VerifyGraph`/`RunGraph`/`GenerateNBG`）。
- `ls -lZ`：`libovxlib.so`、`libNBGlinker.so` 是 `u:object_r:same_process_hal_file:s0`；
  `libOpenVXU.so` 对 shell **和 root 都 Permission denied**（受保护类型）。

## 2. ONNX 的四条落地方式

### A. ONNX Runtime on CPU（最简洁，任意架构，但不碰 NPU）

- 直接把 ORT 编译成 Android arm64 静态/动态库，跑任意 ONNX。
- 优点：**任何架构都能跑**、成熟、零逆向、可立即验证正确性。
- 缺点：只用 CPU；NPU 闲置。
- 适用：先把模型跑通、做数值基准、给 NPU 路径当参考实现。

### B. ONNX Runtime + 自定义 EP（推荐：既任意架构又用 NPU）

ORT 的执行提供者（EP）机制天生就是"能加速的算子交给 EP，其余自动回退 CPU"：

```
ORT 会话
 ├─ 我们的 O3 EP   → 只认领 MatMul / Gemm / Add / Mul / RMSNorm ...
 │                    → 翻译成 xnnl 图 → mida → NPU
 └─ CPU EP        → 其余所有算子（自动 fallback）
```

- **任意架构**成立：图由 ORT 解析和调度，我们只实现"能上 NPU 的那部分"。
- NPU 利用率随算子覆盖率线性增长，不必等"全图可编译"。
- 实现要点：
  1. ORT 的 `IExecutionProvider`（C++）注册到 `SessionOptions`；
  2. `GetCapability` 里按 op_type + dtype + shape 认领节点；
  3. 编译期把权重上传到 NPU 内存（`o3npu_alloc(cpu_visible=1)`），
     运行期只做输入输出拷贝 + `midaStreamLaunchSubgraph`；
  4. I/O 用 CPU 可映射内存，避免额外拷贝。
- 工作量大头：EP 骨架（一次性）+ 每个算子的 lowering（可增量）。
- 这是"成体系 + 成熟库 + 任意架构"三者兼得的唯一现实方案。

### C. ONNX → Acuity（vsi_nn）→ NBG

- 设备上有 `vsi_nn_GenerateNBG`，理论上可以在设备上把图编译成 NBG 再执行。
- 但公共初始化/加载入口（`vsi_nn_Init`、LoadNetwork）不在 `libovxlib.so` 里，
  且配套的 `libOpenVXU.so` 被 SELinux 保护，**开箱不可用**。
- 结论：需要大量逆向，收益不比 B 大。

### D. ONNX → `xnn.plan`

- 需要小米的编译器（PC/云端工具），设备上没有；格式只解了容器层，计算图编码未解。
- 不现实。

## 3. 推荐路线

**第一步（立刻可做）**：ORT CPU 跑通任意 ONNX，建立数值基准。
**第二步**：写 O3 EP，先只认领 `MatMul`（Transformer 的主要算力），
用 `xnnlMatmulCreateGraph` + int4 groupwise（我们已解出量化参数）实现，
其余算子回退 CPU。
**第三步**：按 profile 逐步加算子（Add/Mul/RMSNorm/Softmax）。

这样"任意架构"由 ORT 保证，"NPU 加速"由 EP 的算子覆盖率决定，两者解耦。

## 4. 诚实的边界

任何 NPU 都只认自己编译器支持的算子集。"跑任意架构"在 NPU 上从来不是"全图可编译"，
而是"图上能认领多少算子"。ORT + EP 的混合执行正是工业界的标准答案
（QNN EP、NNAPI EP、CoreML EP 都是这个模型）。

## 5. 参考

- `libxnn_runtime.so`：`Xnn_Context*` 系列（模型级，吃 `xnn.plan`）
- `libxnnapi.so`：`xvpuxnnapi::XRingXnnApi`
- `libovxlib.so`：`vsi_nn_*`（Acuity 图 API）
- `libNBGlinker.so`：`nbg_*`
- 我们自己的驱动：`tools/npu/o3npu.{h,c}`、`docs/NPU-ACCESS.md`


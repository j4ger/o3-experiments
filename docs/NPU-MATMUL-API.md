# xnnl matmul：从二进制读出来的接口事实

来源：Ghidra 反编译 `libxnnl.so`（sha256 `474fb422…`，镜像基址 `0x100000`）。

## 1. `xnnlMatmulCreateGraph` 是调度器，不是实现

伪代码（Ghidra，`bellv100/src/layers/matmul.cpp`）：

```c
if (ParamCheck(obj) != 0) { log("xnnl error! xnnl status: %d"); return ParamCheck(obj); }
if (PolicyFunc(obj) != 0) { log("xnnl error! xnnl status: %d"); return PolicyFunc(obj); }

log("input.shape[%d]=%d" ...);      // 逐维打印
log("weight.shape[%d]=%d" ...);
log("ewData.shape[%d]=%d" ...);     // 可选，允许为空
log("output.shape[%d]=%d" ...);
log("epilogueSize_=%d");
log("groupwiseScaleSize_=%ld");
log("M=%d, K=%d, N=%d");
log("block M=%d, block K=%d, block N=%d");
log("split K: [%d]");
log("block sdp count=%d");
log("transposedW_=%d");
log("kTmaGlobalStridesMNK=%d", 0x8000);
log("policyMode_=%d");
log("kernel name: %s");
log("thread block dim XYZ: {%d, %d, %d}");
log("grid dim XYZ: {%d, %d, %d}");
log("shared memory byte: %u");

switch (policyMode_) {
  case 1:  return createGraphSpecializedN(obj);
  case 2:  log("split K not supported."); return 2;
  default: return createGraphSplitMN(obj);
}
```

因此：

- **入口只做校验 + 选策略 + 建图**，真正的实现分两条：
  `createGraphSpecializedN`（`policyMode_ == 1`）与 `createGraphSplitMN`（其余）。
- **`policyMode_ == 2`（split K）直接被拒绝**，返回状态 2。这是我们能提前避开的坑：
  长 K 的 matmul 不会被自动切开，必须调用方自己处理。
- 两个 `createGraph*` 都接收同一个内部对象（`xnnlMatmulOpImpl`），说明 15 个参数
  最终都填进这个对象，而不是逐个传给 mida。

## 2. 对象里已知的字段（来自日志字符串）

| 字段 | 含义 |
|---|---|
| `input.shape[]` / `weight.shape[]` / `ewData.shape[]` / `output.shape[]` | 四个张量的形状（`ewData` 可为空） |
| `M`、`K`、`N` | matmul 三维 |
| `block M/K/N` | 分块大小（策略算出来的） |
| `split K` | 是否沿 K 切分 |
| `block sdp count` | SDP 数量 |
| `transposedW_` | 权重是否转置 |
| `epilogueSize_` | epilogue 数据大小 |
| `groupwiseScaleSize_` | groupwise 量化 scale 的大小 |
| `policyMode_` | 选路开关（1=SpecializedN，2=SplitK 拒绝，其他=SplitMN） |
| `kernel name` / `thread block dim` / `grid dim` / `shared memory byte` | 选出来的 NPU kernel 及其启动配置 |

`kTmaGlobalStridesMNK = 0x8000` 是常量，说明 TMA 描述符按 M/N/K 固定 stride 布局。

## 3. 对接我们后端时的直接结论

1. **参数顺序不必猜**：四个张量（input / weight / ewData / output）各自
   描述符 + 数据指针，加上 epilogue 描述符、`transposedW` 布尔、一个 uint32。
2. **`ewData` 可以为空** —— 对应"无 bias"的 matmul。
3. **split K 不支持**：K 很大时要自己切，否则 `policyMode_` 落到 2 直接失败。
4. **形状不合法会在 `ParamCheck` 阶段返回非 0**，可以据此快速试错。

## 4. `ParamCheck` 的约束（反编译所得）

每一条都是拒绝消息原文，说明输入必须满足：

**布局与秩**

- `inputTensorDesc_->layout != XNNL_TENSOR_NHWC` → 输入必须是 **NHWC**
- `inputTensorDesc_->layout != weightTensorDesc_->layout`
- `inputTensorDesc_->layout != outputTensorDesc_->layout` → 三者布局必须一致
- `inputTensorDesc_->rank != weightTensorDesc_->rank` / `... != outputTensorDesc_->rank`
  / `... != ewDataDesc_->rank` → **rank 必须全部相等**

**形状**

- `inputTensorDesc_->shape[0] != 1`、`shape[1] != 1` → 输入前两维必须为 1
- `shape[1]` 在 input/weight/output 之间必须一致
- `shape[2]` 在 input/weight/output 之间必须一致
- `inputTensorDesc_->shape[3] != wK` → 输入末维 = K
- `outputTensorDesc_->shape[3] != wN` → 输出末维 = N
- `outputTensorDesc_->shape[0] != maxBatch`
- `ewDataDesc_->shape[2] != outputTensorDesc_->shape[2]`

**尺寸上限**

- `M_ > kTmaGlobalStridesMNK`、`K_ > kTmaGlobalStridesMNK` → **M、K 均 ≤ 0x8000（32768）**

**batch**

- `minBatch < 1 || maxBatch < 1`
- `maxBatch % minBatch != 0` → maxBatch 必须是 minBatch 的整数倍

**可选 epilogue / ewData**

- `ewDataDesc_ != nullptr` / `ewData_ != nullptr` / `ewData_ == nullptr` → 两者要么都给，要么都不给
- 无 epilogue 时必须满足 `act1Op == XNNL_ACT1_TYPE_DISABLE`、
  `act2Op == XNNL_ACT2_TYPE_DISABLE`、`actScaleMode == XNNL_EPL_DATA_MODE_DISABLE`、
  `ewOp` 在 `[XNNL_EW_OP_DISABLE, XNNL_EW_OP_SUB]` 区间内

**指针**

- `handle_ == nullptr`、`graph_ == nullptr`、`input_ == nullptr`、`weight_ == nullptr`、
  `output_ == nullptr`、`inputTensorDesc_ == nullptr`、`weightTensorDesc_ == nullptr`、
  `outputTensorDesc_ == nullptr` → 全部非空检查

**对齐**

- `"N=%d should transposed weight, out N bytes:%d is not divisible by %d"`
  → 当权重不转置（或需要转置）时，N 的字节数必须被某个对齐值整除

## 5. `PolicyFunc`：怎么选 kernel（反编译所得）

决策相关的日志字符串就是文档：

| 消息 | 含义 |
|---|---|
| `"specialized N not support sequence length > 64, current sequence len=%d"` | **specialized N 只支持 N ≤ 64** |
| `"need to split K, not support yet"` | 需要切 K 时直接拒绝 |
| `"split K not supported."` | 同上，`policyMode_ == 2` 的返回点 |
| `"blkN_=%d N_=%d"` | 打印选出来的 block N |

对应代码：

```c
if (*(int *)(this + 0x174) == 1 && 0x40 < *(uint *)(this + 0x148)) {
    log("specialized N not support sequence length > 64, current sequence len=%d", ...);
    ...
}
if (*(int *)(this + 0x174) == 2) {  /* split K */ ... }
```

即：

- **N ≤ 64** → `policyMode_ = 1` → `createGraphSpecializedN`
- **N > 64** → `createGraphSplitMN`
- **K 超限需要切分** → 报错返回，不静默降级

这解释了一个实际约束：**token 数（N）不超过 64 时走专用 kernel，超过则走 split-MN 路径**；
两条路径的 kernel 与分块都不同。

## 6. 还没拿到的

- `ParamCheck` 的具体约束（哪些形状/对齐/类型被拒）。
- `PolicyFunc` 如何决定 `policyMode_`（阈值、对齐要求）。
- `createGraphSplitMN` 内部如何把对象翻译成 mida 节点（真正的图构建）。

`ParamCheck` 与 `PolicyFunc` 已完成；`createGraph*` 未完成。

这三项都可以用同一套离线流程继续：Ghidra 快照 + 按地址反编译。

| 函数 | Ghidra 地址 |
|---|---|
| `xnnlMatmulOpImpl::ParamCheck` | `0x6cf8c0` |
| `xnnlMatmulOpImpl::PolicyFunc` | `0x6d23e4` |
| `xnnlMatmulOpImpl::createGraphSpecializedN` | `0x6d53dc` |
| `xnnlMatmulOpImpl::createGraphSplitMN` | `0x6d871c`（尝试过，函数过大，反编译器一小时内未完成） |

## 7. 复现命令

```bash
G=/nix/store/…-ghidra-12.1.4/lib/ghidra
GHIDRA_INSTALL_DIR=$G npx -y rea-agents@6.7.0 function \
  artifacts/vendor/libxnnl.so 0x6d386c --snapshot /tmp/xnnl-snap.json --format md
```

注意 Ghidra 把镜像基址放在 `0x100000`，所以 ELF VMA `0x5d386c` 对应
Ghidra 地址 `0x6d386c`；直接用 VMA 会报 "No procedure exists"。

---

## 8. 设备回来了怎么测（一次跑完）

探针：`tools/probe/probe_npu_matmul.cpp`，已交叉编译为 arm64。

```bash
adb push probe_npu_matmul /data/local/tmp/
adb shell chmod 755 /data/local/tmp/probe_npu_matmul

# 两条 kernel 路径各跑一次
adb shell "/data/local/tmp/o3p_chat.bin --shell '/data/local/tmp/probe_npu_matmul 64 64 8'"
adb shell "/data/local/tmp/o3p_chat.bin --shell '/data/local/tmp/probe_npu_matmul 64 64 128'"
```

输出会逐段告诉你卡在哪一层：

| 输出 | 含义 |
|---|---|
| `midaDeviceGet -> 0` | 设备可用（已知会通过） |
| `xnnlCreate -> 0` | 算子库可用（已知会通过） |
| `make_desc ... -> 非 0` | 描述符创建/设置被拒，形状或布局不对 |
| `xnnlMatmulCreateGraph -> 0` | **ParamCheck + PolicyFunc 都通过**，形状契约正确 |
| `xnnlMatmulCreateGraph -> 1` | BAD_PARAM：形状违反 ParamCheck |
| `xnnlMatmulCreateGraph -> 2` | NOT_SUPPORTED：走了 split K，或该路径不支持 |
| `midaSubgraphInstantiate(l2b=.. va=..) -> 0` | 执行参数组合被接受（探针扫 9 种） |
| `max|NPU-CPU| = ... MATCH` | **数值正确** |

### 形状的已知不确定点

`ParamCheck` 的约束是硬事实，但**四维形状的语义编码没有完全确定**。探针用的是最贴合
约束的一种读法：

```
input  [1, a, b, K]
weight [1, a, b, ?]
output [maxBatch, a, b, N]
```

若 `xnnlMatmulCreateGraph` 返回非 0，说明这个读法不对；按返回值逐个维度试比重新反编译快。

### 判定顺序

1. 描述符能建 → 形状基本合法
2. 建图返回 0 → **契约读对了**
3. instantiate 返回 0 → 执行参数对了
4. 数值 MATCH → 整条 NPU matmul 打通

任何一步失败都是**局部**问题，不会白跑。

---

## 9. 实机结果（2026-10-29）

探针在 `lhasa` 上跑通了三层，卡在第四层：

```
midaDeviceGet -> 0
xnnlCreate -> 0
descriptors in=.. w=.. out=..          （fp16, NHWC）
xnnlMatmulCreateGraph -> 0            ← 契约读对了
midaGraphCreate/AddSubgraphNode/Instantiate -> 0
midaStreamLaunchGraph -> 14 [midaErrorLaunchFailure: launch task error]
```

### 已确认的调用契约（实机验证）

| 项 | 结论 |
|---|---|
| dtype | **必须是 HALF**；FLOAT 被 `ParamCheck` 拒绝 |
| 布局 | NHWC（layout=2），rank 4 |
| input 形状 | `[1, 1, 1, K]` |
| weight 形状 | **`[1, 1, K, N]`**（不是 `[1,1,1,N]`） |
| output 形状 | `[1, 1, 1, N]` |
| epilogue | 必须传描述符；全 DISABLE 可通过 |
| 数据指针 | fp16 主机缓冲即可（库内部处理） |
| 图执行 | `midaGraphAddSubgraphNode` → `midaGraphInstantiate` → `midaStreamLaunchGraph`，前两步返回 0 |

### 卡点

`midaStreamLaunchGraph` 返回 **14 = `midaErrorLaunchFailure`（launch task error）**，
与 `l2bSize`/`vaSize`/`stack` 的组合无关（14 / 10 / 12 都出现过，均为 launch 失败族）。
说明问题不在参数格式，而在**任务本身没被 NPU 固件接受**。

下一个怀疑方向（按优先级）：

1. **kernel 二进制没有加载**：`PolicyFunc` 会打印 `kernel name`，但 kernel 可能
   需要用某个 `mida...LoadKernel`/module 接口预先注册到 device；
2. `midaLaunchConfig.stripe_names[32]` 需要填，而不是留空；
3. 需要 `midaSubgraphSetDMASplitMode` 之类的子图设置；
4. 该 shape 组合没有可用的 kernel，需要换更大的 M/N。

已确认**不是**的原因：形状、dtype、布局、epilogue、graph 组装、l2b/va/stack 参数。

## 10. 第二轮实机（重启后）

重启清掉了卡死的固件状态，`midaDeviceGet` 恢复正常。这轮拿到两个新事实：

### 10.1 启动要用异步 + stream type 1

```
launch(type=0)      -> 10 [midaErrorFWTimeout]
launch_async(type=0) -> 6 [midaErrorNotSupported]
launch(type=1)      -> 6 [midaErrorNotSupported]
launch_async(type=1) -> 0        ← 唯一被接受的组合
sync                 -> 10 [midaErrorFWTimeout]
```

即：**同步启动不被支持，必须 `midaStreamLaunchGraphAsync` + `midaStreamConfig.type = 1`**。
但随后 `midaStreamSynchronize` 超时，输出仍是垃圾值 → 任务没有真正完成。

### 10.2 mida 堆内存不暴露设备 VA

| flag | CPU 指针 | 设备 VA | fd |
|---|---|---|---|
| 0 | 0 | 0 | 有效 |
| 1 | 有效 | **0** | 有效 |
| 2 | 0 | 0 | 有效 |

所以 `midaMemoryAlloc` 拿不到可用于建图的设备地址；只有 dma-buf fd。
把 VA=0 传给 `xnnlMatmulCreateGraph` 会被 ParamCheck 拒绝（返回 1）。

### 10.3 当前状态与剩余未知

- 传 **host 指针**：builder 返回 0，但 async launch 后 sync 超时、结果错。
- 传 **mida VA**：builder 返回 1（VA 恒为 0）。

结论：**卡点是内存语义**——xnnl 需要的不是裸 host 指针，也不是
`midaMemoryGetDeviceVa`（拿不到），而是某种由它自己管理的缓冲。最可能藏在
`createGraphSplitMN` 里（该函数太大，Ghidra 一小时内没出结果）。

下一步建议：
1. 用更长的 Ghidra 预算（或只反编译 `createGraphSplitMN` 的前 200 条指令）看它如何
   处理 `input_`/`weight_`/`output_`（对象偏移 0x28/0x50/0x58）；
2. 或看 `libxnn_runtime.so` 的 `Xnn_ContextExecute` 如何准备内存并调用 mida。

## 11. 设备内存路径打通（但仍不执行）

### 11.1 唯一能拿到设备 VA 的方式

堆内存（`midaMemoryAlloc` flag 0/1/2）的 `midaMemoryGetDeviceVa` **永远是 0**。
可行的是"预留地址 + 把 dma-buf 绑上去"：

```c
midaMemoryAlloc(&mem, dev, size, 1, 0, name);      // CPU 指针 + fd
midaMemoryGetPtr(mem);                              // 用于填数据
fd = midaMemoryGetFd(mem);
midaAddressReserve(&addr, dev, size, 2, 0);        // type=2 才成功
va = midaAddressGetDeviceVa(addr);                  // 非零，例如 0x456000
midaMemoryMapWithAddr(&mapped, fd, size, 0, addr);  // 同一 fd 绑定到该 VA
```

`midaAddressReserve` 的第一次参数是 type：**只有 `(2, 0)` 成功**，
其他组合（0/1/4096 等）都返回 1。地址空间按预留递增（0x1000 → 0x101000 → 0x501000）。
`midaMemoryAllocWithAddr` 在这个设备上**必然 SIGSEGV**，不要用。

### 11.2 进展

```
reserve → va / map → 都成功
xnnlMatmulCreateGraph(in=0x456000 w=0x458000 out=0x459000) -> 0
midaGraphAddSubgraphNode -> 0, midaGraphInstantiate -> 0
midaStreamLaunchGraphAsync(type=1) -> 0
midaStreamSynchronize -> 10 [midaErrorFWTimeout]
midaStreamLaunchGraphAsyncWithCallback -> 0, 回调从未触发
max|NPU-CPU| = 8.24  （输出仍是全零，任务从未执行）
```

### 11.3 结论

建图与提交都成功了，**固件没有执行任务**。剩下的怀疑：

1. `midaLaunchConfig.stripe_names[32]`（每项 5 字节）留空——厂商流程里这些名字应当填上；
2. `midaSubgraphExecParams` 的 `l2bSize`/`vaSize` 仍为 0（我们扫过 0/1MB/2MB，但也许要匹配真实需求）；
3. kernel 需要额外加载步骤（xnnl 内部或 runtime 的 plan 流程）。

判定方法已经具备：只要 `sync` 返回 0 且 `max|NPU-CPU|` 变小，就说明任务真的跑了。

## 12. 从设备拉下来的配置（2026-10-30，关机前）

### 12.1 `/vendor/etc/npu/` 的结构

```
/vendor/etc/npu/
├── compiler/VIP9000NANOQI_MP6_PID0X1000002B.config   (140 B)
├── config/aon/config.json                            (567 B)
└── models/aon/*.nb                                    (106 MB，7 个模型)
```

`config.json`：

```json
{ "modelPath":   { "1": ".../FD_cipher.nb", "2": ".../Gaze_cipher.nb" },
  "modelPath_o3": { "0": ".../FD_cipher.nb", ... "6": ".../QR_cipher.nb" } }
```

`compiler/*.config`：

```
chipModel    = 0x00009000;   chipRevision = 0x00009200;
productID    = 0x0509000b;   ecoID        = 0x60000000;
customerID   = 0x1000002b;
```

### 12.2 这解释了什么

**厂商在这台设备上跑的是预编译的 `.nb`（NBG）模型**，不是运行时现搭的图。
`models/aon/*.nb` 是 106 MB 的二进制网络，由 VeriSilicon 的编译器离线生成。

由此可以解释我们卡住的原因：

- `xnnlMatmulCreateGraph` 能建图、能实例化，但**固件可能期待一个随 NBG 一起下发的
  kernel/program**，而运行时现搭的 matmul 没有这个 program，于是任务提交后
  固件不执行（`midaErrorFWTimeout`），输出保持全零。
- `libNBGlinker.so`、`vsi_nn_GenerateNBG` 这些符号正是 NBG 流程的一部分。

### 12.3 对路线的直接影响

| 路线 | 可行性 |
|---|---|
| 运行时用 xnnl 现搭 matmul 图 | **受阻**：缺 kernel program，固件不执行 |
| 用厂商 NBG 模型（`.nb`） | 需要 VeriSilicon 编译器（离线工具，未获得） |
| 用 `libxnn_runtime` 加载 `xnn.plan` | 需要 plan 文件（同样是编译产物） |
| CPU（ggml） | **已可用**，184 t/s |

也就是说：**"让 NPU 跑任意我们自己的模型"目前卡在"缺厂商编译器"这一层**，
而不是卡在 API 细节上。继续深挖 xnnl 也许能绕过（例如手工构造 kernel program），
但那是另一个量级的工程量。

## 13. 决定性结论：NBG 必须由编译器生成

### 13.1 `vxGenerateNBG` 做了什么（反编译 `libOpenVX.so`）

```c
if (nodeCount == 0) return -1;
/* 图里若已含 binary 节点则拒绝 */
if ((node->type | 2) == 0x75a003) { print("has binary in graph, can't genereate NBG"); }

/* 关键：先 verify，再 process —— 这是编译器流水线 */
setCompileMode(1);
vxVerifyGraph(graph);
setFlag(2);
vxProcessGraph(graph);
size = *(graph + 0x7a188);
print("Generate NBG in memory Actual NBG size : %lld bytes", size);
```

也就是说 **NBG 不是"序列化现有图"，而是"跑一遍编译器再导出结果"**。
`vxProcessGraph` 会调用 NN/TP 编译器，把网络编译成 NPU 的 command stream 与权重切片，
最后 `vxoBinaryGraph_SaveNBGCacheBufferMemory` 才把这块缓存写成文件。

### 13.2 NBG 容器里有什么

从 `libOpenVX.so` 的符号与字符串可以完整列出：

| 成分 | 证据 |
|---|---|
| 分段：basic / ioinfo / patchdata / table(+v1) / axisram / data_copy | `nbg_fmt_read_basic_section_data`、`nbg_fmt_adjust_ioinfo_section_data`、`nbg_fmt_adjust_patchdata_section_data`、`nbg_fmt_adjust_table_section_v1`、`nbg_fmt_read_axisram`、`nbg_fmt_read_data_copy` |
| NN kernel stream + TP kernel stream | `"fail to get NBG size, NN kernel stream size is 0"` / `"... tp kernel stream size is 0"` |
| weights/biases 切片 | `"... tp weights_biases sclice num is 0"` |
| 设备数与每设备核数 | `"generate NBG, device count=%d, core count per-device"` |
| 内存池 / AXI-SRAM / VIP-SRAM 地址映射 | `"generate NBG, memoryPool startAddress=0x%llx"`、`"map VIP-SRAM startAddress=0x%lx"` |
| 输入输出表（含物理地址回填） | `"update input or output table, oldPhysical: 0x%08X, newPhysical: 0x%08X"` |
| CRC 与结束标记 | `"NBG CRC Sum: 0x%08x"`、`"NBGEND"` |

### 13.3 对"跑我们自己的模型"意味着什么

要让 NPU 执行任意 matmul / 模型，必须提供一份 NBG；而 NBG 里的核心内容是
**编译器生成的 NPU command stream**。手工构造它等价于：

- 自己实现 VeriSilicon VIP9000 的指令编码、寄存器配置、DMA 描述符、SRAM 分配策略；
- 自己完成算子切分与调度（`PolicyFunc` 里的 block M/K/N、split K、sdp count 等就是它的产物）。

这不是"再逆向几个函数"的量级，而是**写一个编译器后端**。

### 13.4 因此，三条路线的最终判定

| 路线 | 判定 |
|---|---|
| 运行时现搭 xnnl 图 | ❌ 固件不执行（缺 kernel program，`midaErrorFWTimeout`） |
| 手工构造 NBG | ❌ 等价于实现编译器后端 |
| 用厂商编译器生成 NBG | ⭕ 唯一可行，但工具未获得 |
| **CPU（ggml）** | ✅ **已可用**（SmolLM2-135M Q4：pp32 323 / tg16 184 t/s） |

**离线能做的逆向到此完成**：不是"没找到方法"，而是**找到了边界**——
在这台设备上，绕过厂商编译器去驱动 NPU 的成本高于收益。

## 14. 更正：编译器就在我们手里（第 13 节的结论作废）

第 13 节说"需要厂商编译器、工具未获得"——**这个结论是错的**，用户当场指出了它。
复查后确认：**NN 编译器就在 `libOpenVX.so` 里**。

### 14.1 证据

`libOpenVX.so` 导出的符号：

```
compiler_core_graph_optimization_hint
compiler_create_tensor
compiler_reorder_reshape_relu_elwt
compiler_blocksize_min / compiler_blocksize_max
cr_compile_storage / cr_compile_storage_deinit
```

这就是图编译器本体。第 13 节看到的 `vxGenerateNBG -> vxProcessGraph` 之所以像
"外部工具"，是因为**编译器是通过公开 API 调用的**，不是命令行程序。

### 14.2 我们手上其实齐了

| 组件 | 库 | 状态 |
|---|---|---|
| 图/算子 API | `libOpenVX.so`、`libovxlib.so` | ✅ 已有 |
| 编译器 | `libOpenVX.so`（`cr_compile_*`、`compiler_*`） | ✅ 已有 |
| NBG 生成/加载 | `vxGenerateNBG`、`nbg_create_network_from_file/memory` | ✅ 已有 |
| 执行/驱动 | `libNBGlinker.so` → `libxnumd.so` → `/dev/hpc-*` | ✅ 已有且已打通 |

### 14.3 正确的一层：OpenVX / vsi_nn，而不是 xnnl

`xnnl` 是**LLM 专用层**，它假设 kernel program 已经由上层准备好；
`vsi_nn` / OpenVX 才是**通用 NN 框架**，编译器在这一层被调用。

关键 API（均已确认导出）：

```
vxCreateContext        vxCreateGraph         vxCreateTensor
vxFullyConnectedLayer  vxVerifyGraph         vxProcessGraph
vxCopyTensorPatch
vsi_nn_CreateGraph     vsi_nn_AddNode        vsi_nn_VerifyGraph
vsi_nn_RunGraph        vsi_nn_AsyncRunGraph
```

`vxFullyConnectedLayer` 的签名从反汇编得出（x0..x6）：

```c
vx_node vxFullyConnectedLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights,
                              vx_tensor biases, vx_enum overflow_policy,
                              vx_enum rounding_policy, vx_tensor outputs);
```

这正是 Khronos NN 扩展的签名 —— 也就是 **matmul 的标准算子**。

### 14.4 下一步（探针已写好）

`tools/probe/probe_openvx_matmul.c`：

```
vxCreateContext -> vxCreateGraph -> 建 4 个 tensor -> vxFullyConnectedLayer
-> vxVerifyGraph（这一步触发编译器）-> vxProcessGraph（NPU 执行）
-> vxCopyTensorPatch 读回 -> 与 CPU 对拍
```

它还会在运行时用 `vxnneGetTypeSize` **自己探测 VX_TYPE_* 的枚举值**，避免猜错类型常量。

判定标准不变：`vxVerifyGraph` 返回 0 说明编译成功，`max|NPU-CPU|` 变小说明真的跑了。

## 15. 实机验证：OpenVX 编译器在这台设备上不可用

第 14 节说"编译器就在 libOpenVX 里、可以直接调"——**这个判断对了一半**：
编译器代码确实在，但它的 HAL 在这台设备上初始化不了。

### 15.1 崩溃点（tombstone 原始记录）

```
signal 11 (SIGSEGV), fault addr 0x0 (read)
#00 gcoHAL_DeviceControl_NoKernel+1368        libGAL.so
#01 _QueryHardwareIdAndOptions+216            libGAL.so
#02 gcoHARDWARE_ConstructEx+3112              libGAL.so
#03 gcoVX_CreateHW+160                        libGAL.so
#04 gcoVX_Construct+460                       libGAL.so
#05 gcoVX_SwitchContext+176                   libGAL.so
#06 vxCreateContext+328                       libOpenVX.so
#07 main                                      probe_openvx_matmul
```

`vxCreateContext` → `gcoVX_SwitchContext` → `gcoVX_CreateHW` → `gcoHARDWARE_ConstructEx`
→ `_QueryHardwareIdAndOptions` → **`gcoHAL_DeviceControl_NoKernel` 空指针**。

### 15.2 原因

`libOpenVX` / `libGAL` 是 **Vivante GC** 平台的构建，HAL 启动时要查询 GC 硬件；
而这台设备是 **Mali GPU**（`/dev/mali0`，没有 `/dev/galcore`）。
HAL 退到 "NoKernel" 路径，又因为没有预留连续视频内存而解引用空指针。

`strings libGAL.so` 里能看到相关开关：`VIV_VX_SKIP_SHADER`、
`VIV_VX_NN_ACTIVE_CORE_COUNT`、`VIV_DEBUG`、`gcoHAL_NoKernel_IsFeatureAvailable`，
以及 `"failed to allocate linear video memory for No Kernel driver"` ——
说明 NoKernel 模式需要一块预留给 GPU 的连续内存，这台机器没有。

### 15.3 那设备上到底怎么跑 NPU？

`libNPURuntime.so` 的符号给出了答案：

```
npu::umd::Network::Create(Device*, void* buffer, uint32_t size, network_config*, Network**)
npu::umd::Task::Create(...)
npu::umd::Stream::Create(...)
```

`Network::Create` 接收**一块 buffer**——那就是 **NBG 镜像**。
也就是说设备侧的流程是：**加载已编译好的 NBG → 建 Network/Task → Stream 执行**，
编译不在这台手机上做。

### 15.4 最终判定（修正版）

| 环节 | 能否在设备上完成 | 证据 |
|---|---|---|
| 加载并执行 NBG | ✅ 可以 | `Network::Create(buffer, size)`；mida 链路已打通 |
| 用 OpenVX 编译图 → NBG | ❌ 不行 | `vxCreateContext` 在 GC HAL 处空指针（tombstone 如上） |
| 编译 NBG | ❌ 需要 PC 端 VeriSilicon 工具链 | `.nb` 模型随固件下发 |

**所以：能在设备上"跑" NBG，但不能在设备上"生成" NBG。**
要让 NPU 跑我们自己的模型，缺的仍然是**离线编译这一步**，
而且现在有 tombstone 证明这不是配置问题，是 HAL 与硬件不匹配。

---

## 16. 更正（决定性）：不需要编译器 —— kernel 是随库预编译下发的

第 13–15 节有一个共同的错误前提：**把 NBG / `xnn.plan` / `.bell` 当成了
NPU 唯一的执行形式**。它不是。NBG 只是**可选的 AOT 交付格式**；`libxnnl.so`
本身就把编译好的 kernel 带在身上。

### 16.1 证据：`libxnnl.so` 内嵌 64 个 ELF kernel 模块

对 `libxnnl.so` 扫描 ELF 魔数并按节头解析，得到 **66 个 ELF 镜像**：
1 个是共享库本身，1 个是普通对象，**剩下 64 个是 kernel 模块**。
每个模块都有：

| 节 | 含义 |
|---|---|
| `.k_text.<Kernel>` | 编译好的 VIP9000 代码 |
| `.k_abi.<Kernel>` | 参数 ABI 描述 |
| `.k_arg.<Kernel>` | 参数布局（部分模块有） |
| `.k_info.<Kernel>` | 启动几何 / 元数据 |

合计 **686 个 kernel**（686 个 `.k_text` 与 686 个 `.k_abi` 一一对应）。

复现：

```bash
node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so
```

输出摘要：

```
kernel modules  64
kernel names    686
  gemm/matmul    80
  bmm/attention  13
  rmsnorm        35
  layernorm      48
  softmax        45
  rope           16
  quant         105
  activation     62
  conv            5
  pool           26
  pad/tile/copy   9
```

matmul 相关的模块：

```
0x0826db   10  llmBmmSplitMNKernel_ViTQkv_DSMEM128, llmBmmSplitMNKernel_DSMEM128, ...
0x0f0ac3   62  llmGemmSplitMNKernel_ViTPostAttn, gemmSplitMNEpilogueFusedKernel, ...
0x13553b   16  OneDispatchGemmOpHighPrecisionKernel, ...
0x416033    3  MatmulOpPerChnSplitKernel, MatmulOpPerChnSplitKernelL1B, MatmulOpKernelSpecializedN
0x41bb83   10  MatmulRopeQuant_F32_S16_General, ...
```

`MatmulOpKernelSpecializedN` 正是第 5 节 `policyMode_ == 1`（N ≤ 64）选中的那个
kernel，它就在 `0x416033` 这个模块里。

### 16.2 加载路径

`libxnnl.so` 不直接依赖 `libxnumd.so`（`DT_NEEDED` 里没有它），而是
`dlopen("libxnumd.so")` + `dlsym`。相关字符串：

```
Created handle with %u discrete ELF modules
Failed to load discrete ELF [%u] '%s': umd status=%d
kernel '%s' not found in any discrete module
handle_->discreteModules.empty()
midaModuleLoadData
midaModuleGetFunction
```

以及源文件路径（安卓构建系统的生成规则）：

```
out/soong/.intermediates/vendor/xring/ai/xnnl/compile_discrete_mida_genrule/
    gen/bellv100/src/core/mida_elf.cpp
```

`libxnnl.so` 还导出专门用来枚举这些模块的 API：

```
xnnlGetDiscreteElfCount / xnnlGetDiscreteElfName / xnnlGetDiscreteElfData
xnnlGetDiscreteElfSize  / xnnlGetUsedModules
```

**结论：编译发生在小米构建时（`compile_discrete_mida_genrule`），
设备上只做 `midaModuleLoadData` 加载。** 设备端不需要任何编译器。

### 16.3 为什么第 15 节的 tombstone 不影响这条路径

tombstone 发生在 `vxCreateContext` → `gcoVX_CreateHW` → `gcoHAL_DeviceControl_NoKernel`，
那是 **`libOpenVX` / `libGAL` 的 Vivante GC 图编译器**，只服务于
"OpenVX 图 → NBG" 这条 AOT 路线。xnnl 完全不经过 `libOpenVX` / `libGAL`：
它依赖 `libc++/libc/libm/libdl/liblog`，运行时只和 `libxnumd.so` 说话。

两条路线互不相干：

| 路线 | 依赖 | 状态 |
|---|---|---|
| OpenVX 图 → NBG | `libOpenVX` → `libGAL` → GC HAL | ❌ 在这台 Mali 设备上初始化失败 |
| xnnl kernel 节点 → mida | `libxnnl`（内嵌 ELF）→ `libxnumd` | ✅ 库可用，kernel 已加载 |

### 16.4 三条执行路径，只有第一条需要"编译器"

| # | 路径 | 是否需要编译器 |
|---|---|---|
| A | AOT：`vxProcessGraph` → NBG，或 `sample.bell` / `xnn.plan` → `Xnn_CreatePlanFromFile` | 需要（厂商已把 `sample.bell` 编好） |
| B | 运行时图：`xnnl*CreateGraph` → `midaGraph*`，或 `Xnn_EngineCreateContext` → `Xnn_ContextExecute` | **不需要** |
| C | 厂商 LLM 引擎：`XNLM_Runtime_createModel(config.json + fp16 .bin)` → `XNLM_Model_run` | **不需要** |

C 的证据：`libxnlm.so` 的符号与字符串里有完整的模型加载链路
（`WeightImpl::parseConfig/parseLayer/parseEmbedding/parseLMHead/loadFp16Weight`、
`XNLM_KVCache_*`、`XNLM_Model_run`），配置键是通用的 transformer 字段
（`hidden_size`、`intermediate_size`、`num_attention_heads`、`num_hidden_layers`、
`vocab_size`、`model_type`）。它按层建 xnnl 子图并直接跑，
**输入是 config + fp16 权重，不是 NBG**。

一个限制：`libxnlm.so` 的 `model_type` 目前只注册了两个实现
（`Qwen2_5_vl`、`Qwen3`/`Qwen3VL`），所以 C 路线只对这两族模型开箱可用。

### 16.5 对 ggml 的直接结论

**不需要编译器。**

需要说清的是当前 `tools/ggml-o3npu/ggml-o3npu.cpp` 的真实状态：
`o3npu_matmul` 是一个**标量 CPU 三重循环**，没有调用任何 NPU。
PPL 1.2729 证明的是 ggml 动态后端的接线、buffer 偏移与算子分派是对的，
**不是 NPU 执行**。所以"ggml 跑通"和"ggml 用上 NPU"是两件事：

| 问题 | 答案 |
|---|---|
| ggml 能跑吗 | 能，CPU 后端，`-t 6`，SmolLM2-135M Q4 323/184 t/s |
| ggml 上 NPU 需要编译器吗 | **不需要**，kernel 已在 `libxnnl.so` 里 |
| 那现在缺什么 | **执行路径没对齐**：探针 `xnnlMatmulCreateGraph=0`、`graph_instantiate=0`、`launch_async=0`，但 `sync=10 (FWTimeout)`，输出全零 |

`FWTimeout` 是 launch 之后固件不执行，属于启动/参数问题，与编译无关。
可疑点（按优先级）：

1. `midaGraphAddSubgraphNode` 的 `params` 结构（探针里是 `uint8_t snp[0x28]` 手工拼的）
   只把 subgraph 指针放在 offset 0，其余字段是猜的；
2. `midaStreamCreate` 的 config（`uint8_t scfg[0x0c]`，type 放在 offset 4）同样是猜的；
3. 走 `npu::umd` 而不是裸 mida：`libxnumd` 里有 `BootDriver::Boot/Init/IsBoot`
   与 `DeviceImpl::Init(unsigned int, midaDeviceConfig*)`，RPMsg 启动日志是
   `---------start boot---------` / `hpc-cdev boot success!!!!!!!`；
4. `midaGraphInstantiate` 之后可能还需要 `midaGraphExecGetSubgraphExec` /
   `midaGraphExecUpdate` 再 launch。

### 16.6 下一步（设备回来后按序验证）

1. 拉全依赖：`libxninfer.so`、`libmisight.so`、`libbase.so`、`libmnne_crypto.so`
   —— 它们出现在 `libxnn_runtime.so` / `libxnn.so` 的 `DT_NEEDED` 里，本地还没有；
2. **先用厂商已编好的 `sample.bell` 打通执行**：
   `Xnn_CreateRuntime` → `Xnn_CreatePlanFromFile("sample.bell")` →
   `Xnn_CreateDevice` → `Xnn_EngineCreateContext` → `Xnn_ContextExecute`。
   这一步跑通就证明"设备 + 驱动 + 我们的调用"整条链没问题；
3. 再确认 `Xnn_EngineCreateContext` 的入参：若它接受 xnnl 构建的图，
   就用它替代手工拼 mida 参数；若不接受，则回到 mida 但按
   `npu::umd::SubgraphExecImpl` 的字段布局补齐 launch 参数；
4. 通了之后，`GGML_OP_MUL_MAT` 才有意义接到 NPU。

### 16.7 一句话

**"需要编译器"这个结论只对 NBG 整网交付成立；对 xnnl/xnlm/ggml 全都不成立。**
kernel 是随 `libxnnl.so` 预编译下发的，686 个，覆盖 matmul / attention / norm /
softmax / rope / quant / activation 全套。之前卡住的是**执行**，不是**编译**。

---

## 17. 执行链路的正确形态（离线静态分析，2026-10-30）

第 11 节把 `xnnlMatmulCreateGraph` 产出的子图又包进 `midaGraphCreate` /
`midaGraphAddSubgraphNode` / `midaGraphInstantiate` 再 launch。反编译显示这层包装
是多余的，而且会丢掉子图自己的执行参数。

### 17.1 `createGraphSpecializedN` 直接构建一个完整子图

反编译（`docs/generated/decomp/createGraphSpecializedN.c`，
源文件 `vendor/xring/ai/xnnl/bellv100/src/layers/matmul_specializedN.cpp`）：

```
xnnlInitTmaLsDesc(... "inTmaLsDesc")
xnnlInitTmaLsDesc(... "weightTmaLsDesc")
xnnlInitTmaLsDesc(... "outTmaLsDesc")
xnnlUmdWrapper::TmaLsArgsInit(... "gwqTmaDesc")
GraphManager::addQWaitNode(graph, subgraph, dep, 0)
xnnlUmdWrapper::AddKernelNode(graph, &node, subgraph, 0, &midaKernelNodeParams)
xnnlUmdWrapper::AddBarrierNode(graph, &node, subgraph, 0, &midaBarrierNodeParams)
GraphManager::addQWaitNode(graph, subgraph, barrier, 1)
xnnlUmdWrapper::AddL2bWriteNode(graph, &node, subgraph, 1, &midaRWL2BNodeParams)
xnnlUmdWrapper::AddBarrierNode(graph, &node, subgraph, 1, &midaBarrierNodeParams)
GraphManager::addFreeNode(graph, subgraph, ..., 1)
```

也就是说 `xnnlMatmulCreateGraph` 的第二个参数（`midaSubgraph_t`）**就是执行单元**：
kernel node、barrier、L2B 读写、释放节点都已经加好了。
调用方只需要：

```
midaSubgraphInstantiate(&exec, subgraph, &execParams)
midaStreamLaunchSubgraphAsync(exec, &launchConfig, stream)
midaStreamSynchronize(stream)
```

### 17.2 结构体语义（来自 `docs/generated/mida_api.h`）

```c
struct midaSubgraphExecParams {   /* 0x20 */
    uint32_t l2bSize;       /* +0x00 */
    uint32_t vaSize;        /* +0x04 */
    uint32_t regFenceFlag;  /* +0x08 */
    uint32_t regNum;        /* +0x0c */
    midaSubgraphStackInfo stackInfo; /* +0x10 { uint64_t stack_size; midaMemory_t stack_mem; } */
};

struct midaSubgraphNodeParams {   /* 0x28 */
    midaSubgraph_t        subgraph; /* +0x00 */
    midaSubgraphExecParams execPara;/* +0x08 */
};
```

`midaSubgraphInstantiate` 的第三个参数**必须非空**（`mida_xnumd_entrypoints.c` 里
`if (param_2 != 0 && param_3 != 0)`，随后把 `param_3` 原样传给
`SubgraphImpl::Instantiate`）。之前探针里的 `uint8_t snp[0x28]` 只在 offset 0 放了
subgraph，`execPara` 全零；graph 路线又把子图包了一层，参数根本落不到子图上。

### 17.3 对探针与后端的要求

1. 直接对 `xnnlMatmulCreateGraph` 返回的 subgraph 调 `midaSubgraphInstantiate`；
2. `execParams` 先传零，`midaSubgraphExecGetStackInfo` 拿到 `stack_size` 后分配
   `stack_mem` 再 instantiate 一次；
3. launch 用 `midaStreamLaunchSubgraphAsync`（同步版本返回 NOT_SUPPORTED）；
4. `midaLaunchConfig.stripe_names[32]` 是每项 5 字节的字符串数组，先留空，失败再填。

`tools/npu/o3npu.c` 已按这个顺序实现，`tools/probe/probe_npu_matmul.cpp` 与
ggml 后端共用同一条路径。实机结果待补。

---

## 18. 实机结果（2026-10-31）：构图成功，固件拒绝执行

完整记录见 `artifacts/probe-logs/2026-10-31-npu/RESULTS.md`。

### 18.1 已经确认的三件事

1. **64 个 kernel 模块真的加载了**
   `[XNNL]: [handle.cpp:81]:Created handle with 64 discrete ELF modules`
2. **内存路径要改用 `midaMemoryAlloc(flag=5)`**
   它同时返回 CPU 指针和非零设备 VA；vendor 的 `WeightImpl::parseLayer` 用的就是
   `midaMemoryAlloc(&mem, dev, size, 5, 0, "layerBuf")`。
   旧的 `flag=1 + AddressReserve + MapWithAddr` 可以放弃。
3. **构图与实例化全部成功**
   `xnnlMatmulCreateGraph` → 0，`midaSubgraphInstantiate` → 0，
   `midaStreamLaunchSubgraphAsync` → 0；只有 `midaStreamSynchronize` 失败。

### 18.2 固件返回什么

```
E kernel_runtime: [ipc_driver.cpp:174]:firmware response fail status 3 2
E kernel_runtime: [stream_impl.cpp:628]:wait task failed 0 2
E kernel_runtime: [stream_impl.cpp:1229]:item run failed...
```

伴随 `xr_dfx: coreid[NPU] exception[NPU_S_EXCEPTION] sub_exception[NPU_S_LOCKUP]`。
`midaErrorLaunchFailure(14)`；早期 flag=1 的路径是 `FWTimeout(10)`。

### 18.3 已穷举且无效的变量

`midaDeviceConfig` 的 count/mode、`l2bSize`(0–32MB)、`vaSize`(0–256MB)、
`cu_num`(1/2/4)、`performance_level`(0–3)、`ddr_ratio`(0/0.5/1)、
`stripe_names`、launch config 传 NULL、`_UMD_FACTORY_CU_MASK`、
`_UMD_FACTORY_DSMEM_LARGE`、四种权重布局组合 —— 结果全部是 14。

形状从 8×8 到 576×49152 全部一致地失败，说明与 shape 无关。

### 18.4 两个新线索

- `midaSubgraphExecGetStackInfo(exec, &info)` 返回 **1（失败）**，size=0。
  反编译显示 `SubgraphExecImpl::Instantiate` 在 `stack_size==0` 时会自己
  `MemoryImpl::Alloc(..., 4, "NPU_KernelStack")`，所以不是我们没给栈。
- `XRingXnnApi::initVppXnn("sample.bell")` 在
  `XnnEngineImpl::Init` 内 SIGSEGV（`tombstone_19`），
  栈为 `Xnn_RuntimeDeserializeEngineFromXnnPlan` → `XnnEngineImpl::Init`。
  `sample.bell` 与本机 runtime 版本不匹配，Track C oracle 暂不可用。

### 18.5 修正后的判定

| 环节 | 状态 |
|---|---|
| kernel 加载 | ✅ 64 模块 |
| 设备内存 | ✅ flag=5，CPU ptr + device VA |
| 图构建 | ✅ 所有测试形状返回 0 |
| subgraph 实例化 | ✅ 返回 0 |
| 任务下发 | ✅ launch 返回 0 |
| 固件执行 | ❌ `fail status 3 2` / `NPU_S_LOCKUP` |

**卡点不再是"有没有编译器"，也不是 launch 参数字段。**
剩余未知集中在 `SubgraphExecImpl::Instantiate` 生成的 **IB 命令缓冲**与固件契约：
要么 IB 里有固件不接受的字段，要么缺少一个 vendor 内部才有的初始化步骤。

### 18.6 下一步优先级

1. 反编译 `SubgraphExecImpl::Instantiate` 里 `NPU_IBCmd` 的填充逻辑，
   对照 `npu_launch_task_para` 的 0x54 字节消息逐字段核对；
2. 找一台能跑 vendor AI 服务的进程上下文（`system_app`）做对照，
   排除"固件只接受特定 domain/uid"；
3. 用 `XNLM` 真实模型（Qwen3）跑一次，作为"NPU 能执行"的 oracle；
4. 以上都不行时，才考虑 `libOpenVX`/NBG 那条 AOT 路线。

---

## 19. 厂家的调用链（静态逆向，2026-10-31）

按"直接看厂家怎么调"的思路，从 `libxnlm.so` 往下挖。

### 19.1 层次

```
libxnlm.so  (XNLM_Model_run → xnlm::Model::infer → ModelImpl)
   ├── libxnnl.so   xnnlDecoderLayer::createGraph / xnnlLLMDecode ...
   └── libxnumd.so  通过函数表直接调用 mida*
```

`libxnlm` 自己 dlsym 了整套 mida API（`strings` 可见 50 多个
`Loaded the mida... symbol successfully`），包括：

- graph 系：`midaGraphCreate`、`midaGraphAddSubgraphNode`、`midaGraphInstantiate`、
  `midaGraphExecGetSubgraphExec`、`midaStreamLaunchGraphAsync`
- subgraph 系：`midaSubgraphCreate`、`midaSubgraphInstantiate`、
  `midaSubgraphExecGetStackInfo`、`midaStreamLaunchSubgraphAsync`
- 内存系：`midaMemoryAlloc`、`midaMemoryGetDeviceVa`、`midaMemoryAddMap` 等

### 19.2 `xnlm::ModelImpl::launchPrefillGraph`

反编译（`/tmp/decomp-mp-prefill.md`）显示的标准流程：

1. 逐层 `xnnlDecoderLayer::createGraph(...)` —— 构建每层子图；
2. 对每层执行函数表 `+0x158` 的调用：
   `f(subgraph, subgraphReleaseCallback, alloc, this+0x70, ...)`
   —— 即把子图挂到 graph 上（带释放回调）；
3. 之后统一 `midaGraphInstantiate` + `midaStreamLaunchGraphAsync`。

`xnlm::ModelImpl::launchDecodeGraph` 是同一套，只有 batch/position 不同。

### 19.3 我们复刻的结果

在 `tools/npu/o3npu.c` 增加 `O3NPU_PATH=graph`：

```
midaGraphCreate            -> 0
midaGraphAddSubgraphNode   -> 0
midaGraphInstantiate       -> 0
midaStreamLaunchGraphAsync -> 0
midaStreamSynchronize      -> 14 (midaErrorLaunchFailure)
```

和 subgraph 路径**完全一样的 14**。`midaGraphExecGetSubgraphExec` 那一步会 SIGSEGV
（可能需要传 `midaGraphNode_t` 的另一种句柄），暂未继续。

### 19.4 这意味着什么

API 形态不是差异点：subgraph 路径、graph 路径、厂家自己的 graph 组合方式，
三者都返回 14。差异只可能来自：

1. **子图里的 kernel node 参数**（xnnl 内部生成，我们没改过，但可能依赖某个我们没设的
   `xnnlHandle` 状态）；
2. **固件侧的准入**：模型/任务签名、进程 domain、或 NPU 的 secure 状态；
3. **我们缺一个厂家内部的初始化步骤**（例如 `xnnlUmdWrapper` 的某个全局初始化）。

### 19.5 下一步（需要设备解锁）

设备重启后停在锁屏（`dumpsys user` = `RUNNING_LOCKED`），
`dev.j4ger.o3root` / `dev.j4ger.o3chat` 在解锁前对 PackageManager 不可见，
`am start` 返回 `aInfo is null`。需要先解锁，才能恢复 root 和 SELinux 规则。

解锁后最有价值的一步：**在厂家自己的进程里挂钩**（Frida / `LD_PRELOAD`），
抓一次真实成功的 `midaGraphAddSubgraphNode` / `midaStreamLaunchGraphAsync` 参数，
和我们的逐字段对比——这比继续盲试有效得多。

---

## 20. 突破：RMSNorm 能跑，matmul 不能（2026-10-31 01:21）

用 `xnnlRMSNormalizationCreateGraph` 做了一个最小对照实验：

```
== rmsnorm probe K=576 ==
rmsnorm create -> 0
rmsnorm instantiate -> 0
rmsnorm stack_info -> 1 size=0
rmsnorm launch -> 0
rmsnorm sync -> 0 (0 (midaSuccess))
RESULT ok (rc=0)
```

**NPU 能执行 kernel，launch/sync 路径本身没有问题。**

同时 XNNL 打出三条 tiling 警告（RMSNorm 自己的 L2B 太大，退化为 DDR TMA）：

```
W [host_pipeline_creator.cpp:243]: user indicates L2B reserved Bytes is bigger than L2bMaxSize! tile failed!
W [host_pipeline_creator.cpp:1022]: tiling failed! the L2B will be bypassed!!!
W [host_pipeline_creator.cpp:1138]: tile failed, just launch kernel once and user's tma will be from/to ddr
```

### 20.1 matmul 与 RMSNorm 的差别

matmul 运行时**没有**任何 XNNL 警告，直接 `stack mem is null` → 固件
`fail status 3 2`。说明问题在 `xnnlMatmulCreateGraph` 内部选择的 kernel/切分，
而不是 launch。

已知差别：

| | RMSNorm | matmul |
|---|---|---|
| 权重描述符 | `[1,1,1,K]` | `[1,1,K,M]` 或 `[1,1,M,K]` |
| 额外参数 | eps/axis | `transposedW`、epilogue、`uint32_t` |
| 权重预处理 | 无 | **可能需要 `xnnlMatmulDataReorder`** |

`libxnnl.so` 导出：

- `xnnlMatmulGetDataReorderSize`
- `xnnlMatmulDataReorder`
- `xnnlGemmSplitMNKernelReorder` / `xnnlGetGemmSplitMNKernelReorderSize`
- `xnnlGemmSplitKGwDataReorder` / `xnnlGemmSplitKGetGwDataReorderSize`

vendor 的 `WeightImpl::parseLayer` 很可能在加载权重时调用了这些 reorder，
而我们的 probe 直接把原始 fp16 权重喂给 `xnnlMatmulCreateGraph`。
**下一步就验证这一点。**

---

## 21. 为什么 matmul 不行而 RMSNorm 行：权重需要 reorder

`xnnlMatmulDataReorder` 反编译出的参数：

```c
int xnnlMatmulDataReorder(
    xnnlTensorDescriptor *inDesc, xnnlTensorDescriptor *wDesc,
    xnnlTensorDescriptor *outDesc, xnnlEpilogueDescriptor *epl,
    long *?, xnnlTensorDescriptor *?, void *?,
    float *?, uchar *?, uint8_t);
```

配套还有 `xnnlMatmulGetDataReorderSize`（7 个参数）和
`xnnlGemmSplitMNKernelReorder` / `xnnlGetGemmSplitMNKernelReorderSize`。

vendor 的 `WeightImpl::parseLayer` 里每个权重都带一个
**`reordered_weights` + `splitInfo`（xnnlWeightPatch）** 结构，
是从模型文件里直接读出来的预重排布局（见
`docs/generated/decomp/weight_impl_parseLayer.c` 第 930-990 行）。

我们的 probe 把**原始 fp16 权重**直接传给 `xnnlMatmulCreateGraph`：
- xnnl 返回 0（构图只检查形状）；
- 但 kernel 的 TMA descriptor 按"已重排"的布局计算 stride；
- 实际缓冲区是未重排的 → NPU 读到非法地址/跨步 → 固件 `fail status 3 2` / `NPU_S_LOCKUP`。

RMSNorm 没有权重重排这一步，所以能跑通——这与实验结果完全吻合。

### 21.1 下一步

1. 反编译 `xnnlMatmulGetDataReorderSize` / `xnnlMatmulDataReorder` 的完整参数语义；
2. 在 `o3npu_matmul_run` 里：先 query reorder size → 分配 reorder buffer →
   调 `xnnlMatmulDataReorder` → 用 reordered 权重调 `xnnlMatmulCreateGraph`；
3. 若 splitMN/splitK 路径还需要 `xnnlGemmSplitMNKernelReorder`，同样处理。

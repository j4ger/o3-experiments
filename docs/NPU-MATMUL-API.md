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

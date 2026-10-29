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

## 5. 还没拿到的

- `ParamCheck` 的具体约束（哪些形状/对齐/类型被拒）。
- `PolicyFunc` 如何决定 `policyMode_`（阈值、对齐要求）。
- `createGraphSplitMN` 内部如何把对象翻译成 mida 节点（真正的图构建）。

这三项都可以用同一套离线流程继续：Ghidra 快照 + 按地址反编译。

| 函数 | Ghidra 地址 |
|---|---|
| `xnnlMatmulOpImpl::ParamCheck` | `0x6cf8c0` |
| `xnnlMatmulOpImpl::PolicyFunc` | `0x6d23e4` |
| `xnnlMatmulOpImpl::createGraphSpecializedN` | `0x6d53dc` |
| `xnnlMatmulOpImpl::createGraphSplitMN` | `0x6d871c` |

## 6. 复现命令

```bash
G=/nix/store/…-ghidra-12.1.4/lib/ghidra
GHIDRA_INSTALL_DIR=$G npx -y rea-agents@6.7.0 function \
  artifacts/vendor/libxnnl.so 0x6d386c --snapshot /tmp/xnnl-snap.json --format md
```

注意 Ghidra 把镜像基址放在 `0x100000`，所以 ELF VMA `0x5d386c` 对应
Ghidra 地址 `0x6d386c`；直接用 VMA 会报 "No procedure exists"。

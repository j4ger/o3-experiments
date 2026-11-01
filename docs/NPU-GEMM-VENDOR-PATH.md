# 厂商 LLM 的 GEMM 路径（2026-11-01）

## 结论

Qwen3-0.7B VLM / MiMo 不走我们 probe 用的 `xnnlMatmulCreateGraph`
（该 API 选到 `MatmulOpPerChnSplitKernelL1B`，固件返回 `fail status 3 2`）。
厂商走的是 xnnl 内部的 **`GemmSplitMNKernel`** 系列，通过：

```
xnnlAddGemmSplitMNKernelNode
xnnlUpdateGemmSplitMNKernelNode
xnnlUpdateGemmSplitMNKernelNodeForNextToken
xnnlGemmSplitMNKernelReorder
xnnlGetGemmSplitMNKernelReorderSize
xnnlGetGemmSplitMNKernelName
```

## 设备上的实际组成

MiMo 进程 maps（`com.xiaomi.aiservice:mimoVlm`）：

```
/data/user_de/0/com.xiaomi.aiservice/taiyi/model/mimo_model_f3/19/libxnnl.so
/data/user_de/0/com.xiaomi.aiservice/taiyi/model/mimo_model_f3/19/libmomi.so
/data/user_de/0/com.xiaomi.aiservice/taiyi/plugin/mimo_dev_f3/25/lib/arm64-v8a/libMI_Genai_Runtime.so
```

Qwen VLM 对应：

```
/data/user_de/0/com.xiaomi.aiservice/taiyi/model/vlm_model_qwen3_0_7b_f3/15/base.json
/data/user_de/0/com.xiaomi.aiservice/taiyi/model/vlm_model_qwen3_0_7b_f3/15/lora_base_llm/task.json
/data/user_de/0/com.xiaomi.aiservice/taiyi/plugin/vlm_qwen3_0_7b_f3/18/plugin-vlm-vlm_QWEN3_0_7B_f3-1.apk
```

运行时日志：

```
[MOMI] setPerformance (O3): ddrPerfLevel=6, npuPerfLevel=5, cuNum=4
[XNNL] [gemm_splitMN_kernel.cpp:574]:[GemmSplitMNKernel] Policy miss
```

## GemmSplitMNImpl 字段（paramCheck 反编译）

| 偏移 | 含义 |
|---|---|
| +0x00 | `input_`（xnnlTensorBox*，null → "input_ == nullptr"） |
| +0x08 | `weight_`（null → "weight_ == nullptr"） |
| +0x20 | `output_`（null → "output_ == nullptr"） |
| +0x30 | transpose 标志（byte） |
| +0x48 | 错误字符串（std::string） |
| +0x60 | `allHeadNum_` |
| +0x64 | `curHeadId_` |

`xnnlAddGemmSplitMNKernelNode` 的 18 个参数构造 `GemmSplitMNImpl`：

```
param_4  → +0x00   (input)
param_5  → +0x08   (weight)
param_6  → +0x10
param_7  → +0x38
param_8  → +0x40
param_9  → +0x20   (output)
param_10 → +0x28
param_11/13 → +0x30
param_12 → +0x2c
param_14 → +0x18
param_15 → +0x98?
param_17 → GemmSplitMNKernelParaStorage*（可选）
param_18 → +0xb0?
```

## 厂商调用点

`xnnlLlmDecodeFfnLayer::createGraphWithKernelOp`（libxnnl 0x6d14e0–0x6d1538）：

```
x0 = handle
x1 = graph manager
w2 = 2
x3 = layerObj + 0x160   (input box)
x4 = layerObj + 0xb0    (weight box)
x5 = 0
x6 = sdp offset
x7 = 0
[sp+0x00] = x3 (same as param_4)
[sp+0x08] = 5
[sp+0x10] = 1
[sp+0x18] = 0
[sp+0x20] = 1 (byte)
[sp+0x28] = 0
[sp+0x30] = 0 (byte)
[sp+0x38] = 0
[sp+0x40] = 0
[sp+0x48] = 2
```

## 对 ggml 适配

- `MUL_MAT` 应实现为 `xnnlAddGemmSplitMNKernelNode`
- 权重布局用 `xnnlGemmSplitMNKernelReorder`
- decode 每 token 用 `xnnlUpdateGemmSplitMNKernelNodeForNextToken`
- 融合 epilogue：`gemmSplitMNEpilogueFusedWith*`（对应 ggml 的 `MUL_MAT` + `SILU/GATE` 组合）
- 性能参数：`ddrPerfLevel=6, npuPerfLevel=5, cuNum=4`

## 当前限制

`/data/user_de/0/com.xiaomi.aiservice` 受 SELinux MCS 类别 `s0:c512,c768` 保护，
即使给 `shell` 域加 `priv_app_36`/`priv_app_data_file` 的 allow 规则也无法读取；
`ksud sepolicy patch` 不能修改 MLS 约束。因此模型自带的 `libxnnl.so`
目前只能通过进程 maps 确认路径，无法直接 pull 出来做静态对比。

## 2026-11-01 离线确认结果（GemmSplitMN probe）

### xnnlTensorBox 真实布局（dumpInfo 反编译）

`xnnlTensorBox` 是普通结构体，不是 `xnnlTensorDescriptor`：

| 偏移 | 字段 |
|---|---|
| +0x00 | `tensorShape[6]` |
| +0x18 | `boxStart[6]` |
| +0x30 | `boxDim[6]` |
| +0x48 | `rank` |
| +0x50 | `global address`（DDR VA） |
| +0x58 | `layout`（NHWC=2） |
| +0x5c | `dataType`（HALF=1） |
| +0x60 | `location`（DDR=0, L2B=1） |
| +0x64 | `isQuant` |
| +0x65 | `isPerChn` |
| +0x66 | `isAsym` |
| +0x70 | `scale address` |
| +0x78 | `dTypeScale` |
| +0x80 | `zp address` |
| +0x88 | `dTypeZp` |
| +0x8c | `groupwiseCnt` |

### 调用结果

用上述 box 调用 `xnnlAddGemmSplitMNKernelNode`：

```
xnnl_shim: addGemmSplitMN(box) -> 3
[XNNL] [gemm_splitMN_kernel.cpp:1569]: call umd failed! umd status=11
[XNNL] [xnnlUmdWrapper.cpp:441]: kernel 'gemmSplitMNEpilogueFusedKernel'
       not found in any discrete module
```

`paramCheck`（input/weight/output/layout/shape）已全部通过，
失败点不在 API 参数，而在 **kernel module 查找**。

### 根因

`xnnlCreate` 日志：

```
Created handle with 64 discrete ELF modules
```

系统 `/vendor/lib64/libxnnl.so` 只带 64 个 discrete ELF 模块，不包含
`gemmSplitMNEpilogueFusedKernel`。厂商模型目录里的
`/data/user_de/.../model/mimo_model_f3/19/libxnnl.so` 才带 LLM kernel 模块
（maps 中可见该文件，但 MCS 保护无法读取）。

### 结论

- `xnnlAddGemmSplitMNKernelNode` 的 ABI/box 已经离线恢复并能通过 paramCheck；
- 但第三方进程用系统 `libxnnl.so` **无法执行 GemmSplitMN**，因为它缺少
  `gemmSplitMNEpilogueFusedKernel` 所在的 discrete ELF 模块；
- 要跑通 GemmSplitMN，必须先获得模型目录自带的 `libxnnl.so` 及其 kernel
  modules（或从厂商侧提供同等模块），这不是靠猜参数能解决的。

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

## 2026-11-01 CDN 复核：MiMo 模型包

用 `tools/query-ai-config.mjs`（secret 来自 APK `R.string.secret_key`）取到：

```
mimo_model_f3 v19
  fileName  mimo_model_f3-1.taiyi.model
  fileSize  1,665,610,611
  fileHash  3c445147bc06f3c353671956f4c1ad148ce34842479b626d6632d467fef229fb
  fileUrl   https://ai-model-cdn.engine.miui.com/ai-taiyi-platform-bucket/deploy/model/e21ab4b9-1bef-45ba-af6f-cd8f6dfb5f3a/mimo_model_f3-1.taiyi.model
```

用 HTTP Range 只取 ZIP 中央目录 + `libxnnl.so` 条目：

- 模型 ZIP 共 21 个条目，没有 `.k_abi` / kernel module 文件；
- 取出的 `libxnnl.so` 9,278,376 B，
  `sha256 = e8bc1b849fdefad35d6faed29b4aa4f99bb5edae5cd79a2bf54e2ced683961db`，
  与 Qwen 模型包里的 `artifacts/cdn/libxnnl.so` **完全相同**。

把 MiMo 版 `libxnnl.so` push 到设备并用 `O3NPU_XNNL_LIB` 加载后，仍然：

```
xnnl_shim: loading /data/local/tmp/libxnnl_model.so
addGemmSplitMN(box) -> 3
kernel 'gemmSplitMNEpilogueFusedKernel' not found in any discrete module
```

## 2026-11-01 复核修正：kernel 并没有缺失

上面的“根因”不成立。用 `tools/npu/list-kernels.mjs` 重新核对：

| 文件 | 大小 | kernel modules | kernels | `gemmSplitMNEpilogueFusedKernel` |
|---|---:|---:|---:|---|
| `artifacts/vendor/libxnnl.so`（系统 `/vendor/lib64`） | 8,716,144 | 64 | 686 | 有（module offset `0x0f0ac3`，62 kernels） |
| `artifacts/cdn/libxnnl.so`（Qwen/MiMo 模型包） | 9,278,376 | 66 | 690 | 有（module offset `0x0e08f1`，62 kernels） |
| `/tmp/libxnnl_mimo_model.so`（MiMo ZIP 取出） | 9,278,376 | 66 | 690 | 同上，sha256 与 CDN 完全一致 |

所以：

- “系统 `libxnnl.so` 只有 64 个模块，不包含 `gemmSplitMNEpilogueFusedKernel`”是错的；
- “只有模型目录自带的 `libxnnl.so` 才带 LLM kernel”也是错的；
- 之前只检查了 `module[39]`（MatmulOp）和 `module[44]`（RMSNormQ），没有检查真正包含
  GemmSplitMN 的模块（ELF 扫描顺序里的 `module[30]`；`xnnlGetDiscreteElf` 的索引顺序
  可能不同，需要上机确认）。

真正的失败链仍然是：

```
xnnlAddGemmSplitMNKernelNode
  -> GemmSplitMNImpl::addKernel
  -> xnnlUmdWrapper::GetFunction
  -> midaModuleGetFunction
  -> status 11
  -> kernel 'gemmSplitMNEpilogueFusedKernel' not found in any discrete module
```

也就是说：**kernel 在 ELF 里，但运行时没有把它注册/查找到**。下一步应查“名字是否一致、
模块是否被 `xnnlCreate` 收进 handle、模块 ABI 是否匹配”，而不是继续找缺失文件。

## 2026-11-01 静态补充：name 构造与 lookup

`xnnlGetGemmSplitMNKernelName`（libxnnl `0x667400`）是以下函数的公开包装：

```
getKernelName(std::string&, xnnlTensorBox*, xnnlActivationType_t,
              unsigned int, bool, unsigned int)
```

它先写基础名 `gemmSplitMNEpilogueFusedKernel`，再按 `actMode` 改写：

| 条件 | 结果 |
|---|---|
| `actMode == 0` | `gemmSplitMNEpilogueFusedKernel` |
| `actMode == 5` | `gemmSplitMNEpilogueFusedWithSiluKernel` |
| `actMode == 6` | `llmGemmSplitMNWithGelu` / `llmDecoderFfnGateGelu`（按 bool 分支） |
| `actMode == 7` | `llmGemmSplitMNWithSwigluKernel` |
| `allHeadNum != 0` | `llmGemmSplitMNWithHeadTransposeKernel`（覆盖前面的名字） |
| flags bit0..3 | 追加 `TransB` / `Gw` / `Ew` / `Sdp` 变体后缀 |

厂商 FFN 调用点传 `actMode=5`，而当前 probe 之前默认传 `actMode=0`，所以日志里看到的是
基础名。`xnnlUmdWrapper::GetFunction`（`0x5765d8`）先查 handle 的 name→function map，
查不到再遍历 handle 的 module 向量，逐个调用：

```
midaModuleGetFunction(void **out, void *module, const char *name)
```

注意参数顺序是 `(out, module, name)`，不是 `(module, name, out)`；之前 probe 的
`--funcs` 禁用很可能就是签名弄反。若 module 向量里没有任何模块返回成功，最终打印
`kernel '...' not found in any discrete module` 并返回 11。

### 决定性发现：`xnnlAddGemmSplitMNKernelNode` 是 `(graph, handle, ...)`

`xnnlAddGemmSplitMNKernelNode`（`0x665edc`）把 `x0` 存为 `x21`、`x1` 存为 `x20`，调用
`GemmSplitMNImpl::addKernel` 时传 `x1=x21, x2=x20`。而 `addKernel` 的第二个显式参数是
`xnnlHandle*`，会把它交给 `getKernelPara` 再调用 `xnnlUmdWrapper::GetFunction`。所以
`x0` 必须是 graph/subgraph，`x1` 必须是 xnnl handle。

厂商 `xnnlLlmDecodeFfnLayer::createGraph`（`0x6cfcb8`）prologue 把 `x1`（handle）存到
`this+0x2a0`、`x2`（graph）存到 `this+0x2a8`；`createGraphWithKernelOp` 在 `0x6d14e0`
取 `x0=[this+0x2a8], x1=[this+0x2a0]`，即 graph 在前、handle 在后。

之前 probe 的 `o3xnnl_gemm_splitmn` 调用的是 `(handle, subgraph)`，顺序反了，导致
`addKernel` 把 `subgraph` 当作 `xnnlHandle*` 去查 kernel，因此必然返回
`umd status=11`。现已修正为 `(subgraph, handle)`；下一次上机先直接重试 E0。

因此下一步的决定性检查是：先用修正后的参数顺序重试；若仍失败，再对
`xnnlGetDiscreteElfName` 列出的每个 module 调用
`midaModuleGetFunction(&fn, mod, "gemmSplitMNEpilogueFusedWithSiluKernel")`，确认
到底是“名字错”“module 没在 handle 里”还是“module ABI 不匹配”。

### 实测（2026-11-01 20:42）

修正为 `(subgraph, handle)` 后：

```
xnnl_shim: addGemmSplitMN(box, p3=0 p10=5 p18=2) -> 0
o3npu: stack_info rc=1 size=0 mem=0x0
K=576 M=576 T=1 max|NPU-CPU|=24.4512 MISMATCH
```

- `param3=2` 时 `addGemmSplitMN -> 0`，但 `midaSubgraphInstantiate` 报
  `queue id set is not continuous:2`；`param3` 是 queue id，单节点子图要传 0。
- `param3=0` 时 instantiate/launch/sync 全 0；XNNL 日志确认 kernel 为
  `gemmSplitMNEpilogueFusedWithSiluKernel`（`actMode=5`，`allHeadNum_=0`，
  `configType_=2`），但输出仍全 0。
- 当前卡点从“kernel 找不到”推进到“kernel 已加入但未写回输出”；下一步查
  `weightL1bBytes_=294912 > sharedMemBytes_=131072`、`stack mem is null`，
  以及厂商 `xnnlLlmDecodeFfnLayer::addAllocNodes` / `AddL2bReadNode` 的图构建步骤。
- 离线复核又发现探针读回顺序错误：`o3npu_matmul_run` 在 `stream_sync` 后用
  `dbuf_sync_cpu`（`midaMemoryCacheStart`）读输出，而 RMSNorm/unary 探针是先用
  `mem_cache_end` invalidate 再读。已改为读前 `dbuf_sync_dev`、读后
  `dbuf_sync_cpu`，构建通过。
- 设备复测 cache 修复：RMSNorm 仍数值正确（0/576 mismatch），说明读回路径没问题；
  GemmSplitMN 输出仍为预填 `1.0` pattern。
- 对照 RMSNorm DOT 发现 GemmSplitMN 少 Barrier；在 kernel 后添加
  `midaSubgraphAddBarrierNode(mode=1, queue 0)` 后，DOT 变为
  `1 queue, total node 2` / `Barrier_1_0 <- @Kernel_0_0:out0`，与 RMSNorm 一致，
  但输出仍为 `1.0` pattern。
- 下一步优先确认权重是否必须先经 `xnnlGemmSplitMNKernelReorder`；
  其次是 `weightL1bBytes_` / L2B 读写节点和 `stack mem is null`。

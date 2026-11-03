# NPU GemmSplitMN 下一步推进（2026-11-01）

## 0. 先修正结论

**MiMo 一定做 matmul**（QKV、FFN、LM head），厂商日志也明确出现
`[GemmSplitMNKernel]`。此前“模型包缺 kernel module”的结论已作废：

- `artifacts/vendor/libxnnl.so`：64 modules / 686 kernels，
  **包含** `gemmSplitMNEpilogueFusedKernel`（module offset `0x0f0ac3`，62 kernels）。
- `artifacts/cdn/libxnnl.so` / MiMo ZIP `libxnnl.so`：66 modules / 690 kernels，
  **同样包含** 该 kernel（module offset `0x0e08f1`，62 kernels）。
- 两边的 `gemmSplitMNEpilogueFusedKernel` 都有完整的
  `.k_text` / `.k_abi` / `.k_arg` / `.k_info` 节。

真正的问题是 **调用参数顺序反了**，不是文件缺失，也不是 kernel 不存在：

```
xnnlAddGemmSplitMNKernelNode -> 3
[gemm_splitMN_kernel.cpp:1569] call umd failed! umd status=11
[xnnlUmdWrapper.cpp:441] kernel 'gemmSplitMNEpilogueFusedKernel'
                         not found in any discrete module
```

静态证据链：

- `xnnlAddGemmSplitMNKernelNode`（libxnnl `0x665edc`）把 `x0` 存为 `x21`、`x1` 存为 `x20`，
  然后调用 `GemmSplitMNImpl::addKernel` 时传 `x1=x21, x2=x20`；
- `GemmSplitMNImpl::addKernel` 的第二个显式参数是 `xnnlHandle*`，它把这个 `x2` 交给
  `getKernelPara(..., handle)`，后者再调 `xnnlUmdWrapper::GetFunction(..., handle, name)`；
- 因此 `xnnlAddGemmSplitMNKernelNode` 的公开签名是 **`(graph, handle, ...)`**；
- 厂商 `xnnlLlmDecodeFfnLayer::createGraph` 的 prologue 把 `x1`（handle）存到 `this+0x2a0`、
  `x2`（graph）存到 `this+0x2a8`，调用点取 `x0=[this+0x2a8], x1=[this+0x2a0]`，再次确认
  **graph 在前、handle 在后**。

之前 probe 传的是 `(handle, subgraph)`，导致 `addKernel` 把 `subgraph` 当成 `xnnlHandle*`
去查 kernel，必然返回 11。已修正为 `(subgraph, handle)`。

## 1. 三个假设（按优先级）

| # | 假设 | 判别方法 |
|---|---|---|
| H0 | **`xnnlAddGemmSplitMNKernelNode(graph, handle, ...)` 参数顺序写反**（已确认并修复） | 用修正后的 probe 直接跑；若 node 构建成功，H0 成立 |
| H1 | `GemmSplitMNImpl` 实际查找的 name 与日志打印的 name 不同（可能带后缀/variant） | 调 `xnnlGetGemmSplitMNKernelName`，与 module 的 `.k_*` 节名逐字比对 |
| H2 | 包含该 kernel 的 module 没有被 `xnnlCreate` 收进 handle（加载失败/索引错/版本过滤） | `xnnlGetDiscreteElfName/Data/Size` + `midaModuleLoadData` + `midaModuleGetFunction` |
| H3 | module 能加载，但 kernel ABI 与当前 UMD/firmware 不匹配（`0x102e0000` vs 能跑的 RMSNorm `0x10260000`） | `midaModuleGetFunction` 对精确 name 的返回值；对比 system/CDN 两份 lib |

H0 是当前最高优先级：如果修正参数顺序后直接成功，后面的 H1/H2/H3 都不需要测试。
H1 成立则改参数/name；H2 成立查 `xnnlCreate` 模块表；H3 成立则转 NBG/预编译模型或 CPU/GPU 回退。

## 2. 离线工作（先做，不碰设备）

### A1. 反汇编 name 构造链

目标符号（`artifacts/vendor/libxnnl.so`）：

```
xnnlGetGemmSplitMNKernelName       0x667400
xnnlAddGemmSplitMNKernelNode      0x665edc
xnnlUpdateGemmSplitMNKernelNode   0x6662b8
```

需要回答：

1. `xnnlGetGemmSplitMNKernelName` 的入参、返回值、是否拼接 `actMode` / `transB` / `silu`；
2. `GemmSplitMNImpl::addKernel` 写入 `this+0xd9` 的字符串到底是什么；
3. `xnnlUmdWrapper::GetFunction` 用什么 key 查 module（全名、去前缀、还是结构体字段）。

### A2. 解析 module 30（或 CDN 的 module 24）

已确认该 module 的节：

```
.k_text.gemmSplitMNEpilogueFusedKernel
.k_abi.gemmSplitMNEpilogueFusedKernel
.k_arg.gemmSplitMNEpilogueFusedKernel
.k_info.gemmSplitMNEpilogueFusedKernel
```

离线把 `.k_info` 的 12 字节与 `midaModuleGetFunction` 的注册规则对齐，确认
section name 是否就是 lookup key。

### A3. 修正 probe 参数

厂商 FFN 调用点（`xnnlLlmDecodeFfnLayer::createGraphWithKernelOp`）的参数是：

```
param3  = 2
param10 = 5      (actMode = Silu)
param18 = 2
transB  = 1
```

当前 shim 已改为默认 `param3=2, param10=5, param18=2`，并支持环境变量覆盖：

```
O3NPU_GEMM_PARAM3 / O3NPU_GEMM_PARAM10 / O3NPU_GEMM_PARAM18
O3NPU_GEMM_ACT / O3NPU_GEMM_ALLHEAD / O3NPU_GEMM_FLAGS
```

`o3xnnl_gemm_name` 会调用 `xnnlGetGemmSplitMNKernelName` 并打印返回字符串。
该符号是 `getKernelName(std::string&, xnnlTensorBox*, xnnlActivationType_t,
unsigned int, bool, unsigned int)` 的公开包装（libxnnl `0x667400`）。

### A4. 扩展探针（已实现）

`tools/npu/o3npu.c` 现有 `--modules` / `--funcs` 已改成覆盖全部模块：

- `--modules`：遍历全部 `xnnlGetDiscreteElfName`，打印 index/name/size/load；
- `--funcs`：对每个模块调 `midaModuleGetFunction`，测试：
  - `gemmSplitMNEpilogueFusedKernel`
  - `gemmSplitMNEpilogueFusedWithSiluKernel`
  - `gemmSplitMNEpilogueFusedWithGwKernel`
  - `llmGemmSplitMNWithHeadTransposeKernel`
  - `llmGemmSplitMNWithSilu` / `llmGemmSplitMNWithSwigluKernel` / `llmGemmSplitMNWithGelu`
  - `llmGemmSplitMNKernel_ViTPostAttn`
- `--gemm-name`：用最小参数调 `xnnlGetGemmSplitMNKernelName`，打印返回字符串。

**重要修正**：`midaModuleGetFunction` 的真实签名是

```c
int midaModuleGetFunction(void **out, void *module, const char *name);
```

不是之前的 `(module, name, out)`。依据：`libxnumd.so:0x92344` 的反汇编，以及
`xnnlUmdWrapper::GetFunction` 在 `0x576840` 的调用方式（x0=out, x1=module, x2=name）。
这个顺序弄反会让 `--funcs` 直接失败或崩溃，也是之前 `--funcs` 被禁用的原因。

### A5. L2B/分配节点的离线确认（2026-11-01）

反编译 `xnnlLlmDecodeFfnLayer::addAllocNodes`（Ghidra 地址 `0x7d3bb4`，对应
nm `0x6d3bb4`）确认它只做 **8 个 `GraphManager::addAllocNode`**，参数来自
`this+0x2a8`（子图）和 `this+0x2d0` 的大小表，覆盖 FFN 的 gate/up/down 与
中间 buffer；它不是 `GemmSplitMNImpl` 自己的一部分，而是 `xnnlLlmDecodeFfnLayer`
图构建的外围节点。

`xnnlUmdWrapper::AddL2bReadNode`（`0x675a38` / nm `0x575a38`）本身只是
`[this+0x40](out, subgraph, queue, midaRWL2BNodeParams*)` 的薄包装。结论：
standalone `xnnlAddGemmSplitMNKernelNode` 不带这些 FFN L2B 节点是正常的；
当前优先验证的是输出 cache invalidate 顺序，而不是先补 L2B 图。

## 3. 设备实验（一次启动只测一个变量）

按顺序执行，任何一步失败就停在该层：

0. **E0 直接重试（最高优先级，已完成）**：用修正后的 probe 跑
   `O3NPU_GEMM=1 O3NPU_GEMM_PARAM10=5 O3NPU_GEMM_PARAM18=2`（transposed_w=1）。
   实测：
   - `param3=2`：`addGemmSplitMN -> 0`，但 instantiate 报
     `queue id set is not continuous:2`（`param3` 是 queue id，单节点子图要用 0）；
   - `param3=0`：`addGemmSplitMN -> 0`、instantiate/launch/sync 全部 0；
   - XNNL 日志确认 kernel 为 `gemmSplitMNEpilogueFusedWithSiluKernel`；
   - **输出仍为 0**，所以 node 构建/查找已经不是问题，下一步是 E5 的 L2B/stack/TMA。
1. **E1 模块表**：`--modules`，确认 gemm module 的 index/name/load=0。
2. **E2 函数查找**：`--funcs`，看 `midaModuleGetFunction` 能否拿到精确 name 的 fn 指针。
   - 失败 → H2/H3；
   - 成功 → H1（问题在 xnnl 层的 name/调用）。
3. **E3 name 对齐**：`--gemm-name` 用厂商参数（param3=2/param10=5/param18=2/transB=1）
   与当前参数各跑一次，打印返回 name，和 module 节名逐字比较。
4. **E4 node 构建**：用厂商参数调 `xnnlAddGemmSplitMNKernelNode`；若仍 status=11，
   用 `LD_PRELOAD` 包一层 `midaModuleGetFunction`，记录 module handle、name、返回值。
5. **E5 输出为零（当前卡点）**：node 已构建成功并 instantiate/launch/sync 0，但输出全 0。
   复测结果：

   - cache 读回顺序已修正（读前 `dbuf_sync_dev`、读后 `dbuf_sync_cpu`），
     RMSNorm 仍然数值正确；GemmSplitMN 输出仍为预填的 `1.0` pattern，
     说明不是 cache 读回问题。
   - 对比 RMSNorm DOT 发现 GemmSplitMN 少 Barrier 节点；已在 kernel 后
     `midaSubgraphAddBarrierNode(mode=1, queue 0)`，DOT 变为
     `1 queue, total node 2`，`Barrier_1_0` 输入 `@Kernel_0_0:out0`，
     与 RMSNorm 结构一致。
   - 但输出仍为 `1.0` pattern，所以 Barrier 只是补齐了图结构，
     kernel 本身仍未写回。

   2026-11-04 单变量复盘：

   - `O3NPU_HANDLE_L2B` 写 `handle+0x10` 会让下一次 addGemmSplitMN segfault，
     该偏移不是 l2bMaxSize；已从 probe 移除。
   - `O3NPU_L2B`（128K/512K/1M/8M）、`O3NPU_STACK`（256K/1M/4M）、
     `O3NPU_DMASPLIT`（0/1/2）、`O3NPU_TRANSPOSED_W=0` 都不改变 `1.0` pattern。
   - `O3NPU_L2BALLOC` 即使参数满足 `AllocL2BNodeExecImpl::ParamsCheck` 也 segfault，
     需要先恢复 `PrepareAllocFreeL2B` 的完整参数语义。

   下一层按优先级：

   1. 离线恢复 `midaAllocFreeL2BNodeParams` / `midaRWL2BNodeParams` 的完整字段表
      与 `PrepareAllocFreeL2B` / `PrepareRWDMA` 参数语义；
   2. 确认 `weightL1bBytes_=294912 > sharedMemBytes_=131072` 是否需要
      L2B alloc + read 节点；
   3. `stack mem is null` 暂不作为主假设（显式 stack 已试过，无变化）。
   `xnnlGemmSplitMNKernelReorder` 已恢复签名，但当前 `sdpBlkSize=0`、
   `groupwiseCnt=0`，优先级下调。

## 4. 判定树

```
E0 成功
 └─ 参数顺序就是根因，直接进入 E5；不需要再查 module/name
E0 仍失败
 ├─ E2 失败
 │   ├─ 所有 gemm kernel name 都失败 -> H2/H3：module 未注册或 ABI 不匹配
 │   │   ├─ 换 system lib / CDN lib 各试一次
 │   │   ├─ 检查 xnnlCreate 实际加载的 module 数
 │   │   └─ 仍失败 -> 停止 kernel 路线，转 NBG/CPU/GPU
 │   └─ 部分 name 成功 -> 用成功的 name 反向修正 GemmSplitMN 参数
 ├─ E2 成功但 E4 失败
 │   └─ xnnlUmdWrapper/GetFunction 的 key 或 handle 问题 -> 抓 name 与 handle 对比
 └─ E4 成功但 E5 失败
     └─ 权重 reorder / TMA 描述符问题 -> 反编译 GemmSplitMNImpl::getKernelPara / init*TmaDesc
```

## 5. 保底路线

如果 E2 证明 GemmSplitMN 在当前 UMD 上不可用：

1. 继续 `xnnlMatmulCreateGraph` + `xnnlMatmulDataReorder`（已有部分分析）；
2. 或使用 `llmGemmSplitMNKernel_ViTPostAttn` 这类已验证存在的变体；
3. 或转 NBG / `Xnn_Engine` 预编译模型；
4. ggml backend 保持 CPU 回退，不把未验证的 NPU 路径当默认。

## 6. 交付物与进度

- `docs/NPU-GEMM-VENDOR-PATH.md` 的“缺 module”结论已修正；
- `tools/npu/xnnl_shim.cpp` 已加入 `O3NPU_GEMM_PARAM*`、`o3xnnl_gemm_name`；
- `tools/npu/o3npu.c` 已把 `--modules` / `--funcs` 扩展到全模块，并修正 `midaModuleGetFunction` 参数顺序；
- `tools/probe/probe_npu_matmul.cpp` 已启用 `--funcs`、新增 `--gemm-name`；
- 构建验证：`tools/probe/build.sh /tmp/probe_npu_matmul_p0` 通过（arm64 Android）；
- 设备实测：`param3=0` 时 node/instantiate/launch/sync 全 0，kernel 名为
  `gemmSplitMNEpilogueFusedWithSiluKernel`，但输出仍为 0；
- 原始记录：`artifacts/probe-logs/2026-11-01-gemm-arg-order/`；
- 最终以“能否在第三方进程里拿到 `gemmSplitMNEpilogueFusedKernel` 的 fn 指针”为 Go/No-Go 判据。

# NPU 实机实验记录 2026-10-31

设备：lhasa / 2608BPX34C / OS4.0.26.0.XPNCMXM / Android 17
SELinux：Enforcing（本轮新增 vendor_data_file 读规则，仍被拒）
Root：o3p_chat.bin --shell（uid 0）

## 1. 模块加载

`logcat`：`[XNNL]: [handle.cpp:81]:Created handle with 64 discrete ELF modules`
→ 686 个 kernel 确实在 `xnnlCreate` 时全部装入。

## 2. 内存路径（关键修正）

`midaMemoryAlloc(flag=5)` 同时返回 CPU 指针与**非零设备 VA**：

```
o3npu: buf o3in  size=4096   flag=5 cpu=0x... fd=28080 va=0x456000
o3npu: buf o3w   size=663552 flag=5 cpu=0x... fd=28081 va=0x457000
o3npu: buf o3out size=4096   flag=5 cpu=0x... fd=28082 va=0x4f9000
```

这正是 vendor 自己的用法：`WeightImpl::parseLayer` 调
`midaMemoryAlloc(&mem, dev, size, 5, 0, "layerBuf")`，然后
`midaMemoryGetPtr` + `midaMemoryGetDeviceVa`。
旧的 `flag=1 + midaAddressReserve + midaMemoryMapWithAddr` 路径可以放弃。

## 3. 构图与实例化

对以下所有形状，`xnnlMatmulCreateGraph` 与 `midaSubgraphInstantiate` 均返回 0：

| K | M | T | 构图 | instantiate | launch | sync |
|---|---|---|---|---|---|---|
| 8 | 8 | 1 | 0 | 0 | 0 | 14 LaunchFailure |
| 16 | 16 | 1 | 0 | 0 | 0 | 14 |
| 32 | 32 | 1 | 0 | 0 | 0 | 14 |
| 64 | 8 | 1 | 0 | 0 | 0 | 14 / 10 |
| 576 | 576 | 1 | 0 | 0 | 0 | 14 |
| 576 | 1024 | 1 | 0 | 0 | 0 | 14 |
| 576 | 576 | 32 | 0 | 0 | 0 | 14 |
| 576 | 49152 | 1 | **2 (NOT_SUPPORTED)** | - | - | splitK |

## 4. 固件侧错误

```
E kernel_runtime: [ipc_driver.cpp:174]:firmware response fail status 3 2
E kernel_runtime: [stream_impl.cpp:628]:wait task failed 0 2
E kernel_runtime: [stream_impl.cpp:1229]:item run failed...
```
有时是 `firmware response npu timeout status`，并伴随
`xr_dfx: coreid[NPU] exception[NPU_S_EXCEPTION] sub_exception[NPU_S_LOCKUP]`。
→ 任务已下发到固件，但固件拒绝/挂死；不是编译问题，也不是权限问题。

## 5. 已排除的变量

| 变量 | 取值 | 结果 |
|---|---|---|
| `midaDeviceConfig` count/mode | 0/1/2/4/6, mode 0/1/2 | 14 |
| `midaSubgraphExecParams.l2bSize` | 0 / 1M / 2M / 8M / 16M / 32M | 14 |
| `midaSubgraphExecParams.vaSize` | 0 / 1M / 16M / 64M / 256M | 14 |
| `midaLaunchConfig.cu_num` | 1 / 2 / 4（6 → InvalidValue） | 14 |
| `performance_level` | 0 / 1 / 2 / 3 | 14 |
| `ddr_ratio` | 0 / 0.5 / 1.0 | 14 |
| `stripe_names` | 空 / cu0..cu3 | 14 |
| launch config 传 NULL（用默认 cu=4） | - | 14 |
| `_UMD_FACTORY_CU_MASK` | 0x1 / 0xf / 0x3f | 14 |
| `_UMD_FACTORY_DSMEM_LARGE` | 1 | 14 |
| 权重布局 transposedW / 描述符顺序 | 4 种组合 | 14 |

## 6. 另一个线索

`midaSubgraphExecGetStackInfo(exec, &info)` 返回 **1（失败）**，size=0。
反编译显示 `SubgraphExecImpl::Instantiate` 在 `stack_size == 0` 时会自己
`MemoryImpl::Alloc(..., 4, ..., "NPU_KernelStack")`，所以栈不是我们没给。

## 7. Track C（vendor 运行时）

`libxnnapi.so` 的 `XRingXnnApi::initVppXnn("/data/local/tmp/sample.bell")`
在 `xnn::runtime::impl::XnnEngineImpl::Init` 内 SIGSEGV（tombstone_19），
栈：`Xnn_RuntimeDeserializeEngineFromXnnPlan` → `XnnEngineImpl::Init`。
说明 `sample.bell` 与本机 runtime 版本不匹配（或缺少配套 weight/descriptor），
这条 oracle 暂不可用。

## 8. 结论

- **编译不是瓶颈**：64 个 ELF 模块已加载。
- **内存路径已对齐 vendor**：flag=5。
- **图谱构建/实例化全部成功**。
- 卡点收敛为：**固件拒绝我们通过 `midaStreamLaunchSubgraphAsync` 下发的 IB**，
  返回 `firmware response fail status 3 2`。这与 launch 参数无关（已穷举），
  更像是 IB/命令缓冲内容或 subgraph exec 的某个内部字段不满足固件契约。

## 9. 第二轮（重启 + 解锁后，2026-10-31 01:10-01:13）

### 9.1 新开的 UMD 日志（persist.vendor.npu.loglevel=5）

```
I kernel_runtime: [device_impl.cpp:211]:NPU UMD Version: 0.7.13 Date:Oct 12 2026 10:37:15
W kernel_runtime: [module_impl.cpp:428]:module no kernel
I [XNNL]: [handle.cpp:81]:Created handle with 64 discrete ELF modules
W kernel_runtime: [subgraphexec_impl.cpp:4253]:stack mem is null
E kernel_runtime: [ipc_driver.cpp:174]:firmware response fail status 3 2
```

两个新的关键警告：

1. **`module no kernel`** —— 64 个模块里至少有一个被 UMD 认为"没有 kernel"；
   如果恰好是 matmul 模块，`AddKernelNode` 可能拿到空 function，固件收到无效 IB 就会
   报 `fail status 3 2`。
2. **`stack mem is null`** —— `SubgraphExecImpl::Instantiate` 在没有
   `midaSubgraphExecParams.stackInfo` 时不会给 kernel 分配栈；
   反编译里只有当 `this+0xf8 != 0` 才走 else 分支，否则整段跳过。

### 9.2 显式给栈也没用

加 `O3NPU_STACK=262144`：

```
o3npu: buf o3stack size=262144 flag=5 cpu=0x... va=0x4fa000
o3npu: provided stack 262144 bytes
```

但日志仍然是 `stack mem is null`，说明这个警告来自另一处（可能是 xnnl 内部
另建的 subgraph），不是我们的 `stackInfo`。

### 9.3 重要：每次重启后的**第一条** NPU 命令就会 lockup

`logcat` 显示第一条命令后立刻出现
`NPU_S_EXCEPTION / NPU_S_LOCKUP`（`crash_history` 计数 +1）。
**之后所有测试都是在一个已经挂死的 NPU 上做的**，所以之前"14"的矩阵结果
只能说明"第一条命令就失败"，不能区分是哪个参数导致的。
→ 下一轮必须做到"一次重启只测一个变量"。

### 9.4 已拉取的补充 vendor 库

`libxninfer.so`、`libmisight.so`、`libbase.so`、`libmnne_crypto.so`、
`libxnn_cpu.so`、`libxnn_gpu.so`、`libxnn_mapping.so`、`libxnn_op.so`、
`FD_cipher.nb`（AON 模型样本）。

## 10. 关键推断：子图里可能没有 kernel 节点

用 `rizin` 查 `"stack mem is null"` 的 xref，落点是
`npu::umd::SubgraphExecImpl::GetStackInfo` —— 也就是说这条警告是我们调用
`midaSubgraphExecGetStackInfo` 时打的，不是 `Instantiate` 内部的报错。

结合反编译：

```c
uVar12 = local_148;            // 子图计算出的 kernel stack 总大小
if ((local_148 != 0) && ((this[0x741] & 1) == 0)) {
    if (*(long *)(this + 0xf8) == 0) {
        MemoryImpl::Alloc(..., uVar12, 4, "NPU_KernelStack");
    }
}
```

`local_148 == 0` ⇒ 连 `NPU_KernelStack` 都不会分配。
而 `GetStackInfo` 返回 1、size=0 也说明执行体里没有任何需要栈的 kernel node。

**推断：`xnnlMatmulCreateGraph` 返回 0，但没有把 kernel node 真正加进子图。**
这与 `module_impl.cpp:428: module no kernel` 的警告吻合：某个模块被 UMD 认为
没有可用 kernel，`AddKernelNode` 可能静默失败。

### 10.1 `midaDeviceReset` 无效

`midaDeviceReset -> 0`，但下一次 launch 仍然立刻产生一条新的
`NPU_S_LOCKUP`（crash_history 从 219 起继续增加）。
说明每次 launch 都会让 NPU 固件崩一次，reset 不能恢复；但也说明
"一次重启只能测一个变量"这个担心可以放下——每次测试本来就是一次新的崩溃。

### 10.2 下一步最值得做的验证

`libxnnl.so` 里有 `xnnlDemoAddCreateKernel` 和 `xnnlRMSNormalizationCreateGraph`，
都是比 matmul 简单得多的构图入口。用它们建一个最小子图再 launch：

- 若同样 `stack_info rc=1` / 崩溃 → 问题在 xnnl 的通用构图路径（AddKernelNode）；
- 若 RMSNorm 能跑 → 问题只在 matmul 的 specializedN/splitMN 分支。

这比继续调 mida 参数更接近根因。

## 11. RMSNorm 对照实验（关键）

`xnnlRMSNormalizationCreateGraph` 最小子图：

```
rmsnorm create -> 0
rmsnorm instantiate -> 0
rmsnorm stack_info -> 1 size=0
rmsnorm launch -> 0
rmsnorm sync -> 0 (midaSuccess)
```

→ **NPU 可以执行 kernel**，launch 路径正确。
matmul 失败与 launch 参数无关，问题在 `xnnlMatmulCreateGraph` 内部
（kernel 选择/切分/权重 reorder）。下一步验证 `xnnlMatmulDataReorder`。

## 12. 结论：matmul 需要权重 reorder

- RMSNorm 能跑（create/instantiate/launch/sync 全 0）。
- matmul 构图返回 0 但运行必然崩，且没有任何 XNNL tiling 警告。
- vendor 的权重是模型文件里预重排好的 `reordered_weights`（含 `splitInfo`）。
- 我们用原始 fp16 权重，TMA descriptor 与实际布局不匹配 → 固件 lockup。

下一步：调用 `xnnlMatmulDataReorder` / `xnnlGemmSplitMNKernelReorder`
生成重排权重，再建 matmul 图。

## 13. 算子探针实跑（2026-10-31 11:44，设备重新连接）

```
O3NPU_MEM_FLAG=5 ./probe_npu_matmul --ops --k 576

rmsnorm create -> 0
rmsnorm instantiate -> 0
rmsnorm stack_info -> 1 size=0
rmsnorm launch -> 0
rmsnorm sync -> 0 (0 (midaSuccess))
rmsnorm check: 0/576 mismatch
unary abs create -> 1
unary neg create -> 1
unary sign create -> 1

rmsnorm  : PASS (rc=0)
abs      : FAIL (rc=-7)
neg      : FAIL (rc=-7)
sign     : FAIL (rc=-7)
```

**RMSNorm 数值校验通过**：全 1 输入 + 全 1 权重，读回 576 个 fp16
与 1.0 的误差全部 < 2e-3，0 个 mismatch。RMSNorm 从"能执行"升级为"结果正确"。

### 13.1 abs/neg/sign 失败原因（日志原文）

```
E [XNNL]: [unary.cpp:124]: xnnl error! condition:
  (l2bMaxSize_ > 8 * 1024 * 1024) || (l2bMaxSize_ < 128 * 1024)
E [XNNL]: [unary.cpp:631]: xnnl error! xnnl status: 1
```

- `xnnlAbsCreateGraph` 的签名（反汇编确认）：
  `(handle, subgraph, uint32_t mode, inDesc, inPtr, outDesc, outPtr)`，我们传的是对的。
- 失败点是 `xnnlUnaryImpl::paramCheck` 里对 `l2bMaxSize_` 的检查。
- 设备侧 `midaDeviceGetLimit(0)` 返回 **8 MB**（`DeviceImpl::GetLimit` 读 `this+0x1b4`）。
- `xnnlCreate` 只分配 0x48 字节 handle 并把 +0x08..+0x40 清零，**没有从 device
  读 l2bMaxSize_**；所以 handle 的 `l2bMaxSize_` 始终是 0，unary 算子全部 BAD_PARAM。

结论：这不是我们参数写错，而是 `xnnlCreate` 没有初始化 unary 需要的
`l2bMaxSize_`（vendor 自己的路径可能有别的初始化，但我们目前没找到导出的 setter）。

### 13.2 补充

- `midaDeviceGetLimit(uint32_t *out, uint32_t limit, midaDevice_t)` 的参数顺序
  已修正（之前写成 `(dev, idx, &v)`，读到全 0 是错的）。
- `midaDeviceSetLimit(limit, value)` 已绑定，但对 limit=0 调用会 SIGSEGV，
  暂不继续尝试。

## 14. unary 算子第三参数 = l2bMaxSize（2026-10-31 11:53）

反汇编 `xnnlAbsCreateGraph` 的对象布局：

```
this+0x00 = handle
this+0x08 = subgraph
this+0x10 = param_3      ← xnnlUnaryImpl::paramCheck 直接读这个做 L2B 范围检查
this+0x18 = param_6 (desc)
this+0x20 = param_7 (ptr)
this+0x28 = param_4 (desc)
this+0x30 = param_5 (ptr)
```

所以 `xnnlAbsCreateGraph(handle, subgraph, uint32_t l2bMaxSize, inDesc, inPtr, outDesc, outPtr)`
的第三个参数不是 mode，而是 **l2bMaxSize**。之前传 0 必然 BAD_PARAM。

传 1 MB 后：

```
unary abs create -> 0
unary abs instantiate -> 0 (l2b=1048576)
unary abs launch -> 0
unary abs sync -> 0 (0 (midaSuccess))
```

**create/instantiate/launch/sync 全部通过**，不再报 `unary.cpp:124`。

但输出缓冲区没有正确写回：

```
unary abs chunk 0-63: 64 bad
...
unary abs check: 544/576 mismatch (in[0]=1.000000 out[0]=0.500000)
```

out 里仍是初始化时填的 0.5，说明 unary 的 kernel node 没有真正把结果写回
（和 matmul 一样，属于"图能建、能 launch，但 kernel 不产生数据"这一类）。

下一步：查 `xnnlUnaryImpl::makeGraphL2BAutoPipe` 生成的 kernel node，
以及 unary kernel 模块是否真的被加载。

## 15. matmul dispatcher 日志（2026-10-31 12:22，XNNL_LOG=0）

打开 `XNNLLogger::set_log_level(0)` 后拿到 matmul 内部全部参数。

### 15.1 K=576 M=576（splitMN 路径）

```
outGlobalDims:576,1,1,1 outBoxDims:448,1,1,1
outGlobalBytes=1152, output->needBytes=1152 blkN=448 blkM=1 blkB=1
inSmemBytes=1536 weightSmemBytes=516096 outSmemBytes=1024
M:1 K:576 N:576 dstFmt:4 srcActFmt:4 srcWeightFmt:2 transposeB:0 ...
sum=2560(should be <= 131072)
create graph, ret=0
```

### 15.2 K=576 M=64

```
weightGlobalDims:64,576,1,1 weightBoxDims:64,576,1,1
wGlobalBytes=73728 blkN=64 blkM=1 blkB=1
inSmemBytes=1536 weightSmemBytes=73728 outSmemBytes=512
sum=2048(should be <= 131072)
create graph, ret=0
```

### 15.3 关键差异：weight TMA 的目标是 L2B

matmul 三条 TMA 描述符：

```
in  : glbMemType:0  locMemType:0  l1BPri:0
w   : glbMemType:0  locMemType:1  l1BPri:1     ← 目标 L2B
out : glbMemType:0  locMemType:0  l1BPri:0
```

RMSNorm 对照（能跑）：

```
in/out 全部 locMemType:0, l1BPri:1
```

而且 matmul 的日志里**没有** `addAllocNode` / `addFreeNode` / `AddL2b*` 之类的节点记录，
说明这张 splitMN 图很可能**没有给 weight 的 L2B 目标分配 L2B 区域**，
固件在写 L2B 时拿到无效地址 → `fail status 3 2` / `NPU_S_LOCKUP`。

### 15.4 试过但无效

- `O3NPU_L2B` = 1MB / 4MB / 8MB / 16MB
- `O3NPU_VA` = 8MB / 16MB
- `O3NPU_REGNUM` = 1 / 2 / 4 / 8
- `midaSubgraphSetDMASplitMode` = 0 / 1 / 2 / 3
- `--transposed 0`（真正做了 [K][M] 转置）：weight 仍是 `locMemType:1`

全部仍是 `midaStreamSynchronize -> 14`。

### 15.5 下一步

反编译 `xnnlMatmulOpImpl::createGraphSplitMN`，确认它是否应该调用
`SubgraphImpl::AddL2bAllocNode`（`libxnumd` 导出了
`midaSubgraphAddL2BAllocNode` 对应的 C++ 方法），以及为什么没有调用。

## 16. matmul kernel 名称与 policy（2026-10-31 12:32）

打开 XNNL_LOG=0 后拿到关键行：

```
[matmul.cpp:242] policyMode_=0
[matmul.cpp:244] kernel name: MatmulOpPerChnSplitKernelL1B
```

- kernel 是 `MatmulOpPerChnSplitKernelL1B`，即 **per-channel + split + L1B** 变体。
- `policyMode_` 在 `xnnlMatmulCreateGraph` 里被硬编码为 0，无法通过 API 切换。
- weight TMA `locMemType:1` = L1B（不是 L2B），dims/strides 与 ggml `[M][K]` 布局一致：
  `global_strides:576,64,1 global_dims:576,64,1,1`（K 最快）。

### 16.1 又试过仍失败

- 常见方阵：1024×1024、2048×2048
- `O3NPU_DSMEM` = 1（dsmem_large）× `O3NPU_COUNT` = 0/1/4
- 之前的 L2B/VA/REGNUM/DMASPLIT/MM_LAST/transposed 全部无效

### 16.2 当前判断

splitMN 的构图、TMA 描述符、shared memory 都合法（`sum=2048/2560 <= 131072`），
但固件始终 `fail status 3 2`。说明问题更可能在
**`MatmulOpPerChnSplitKernelL1B` 这个 kernel 本身**（与当前固件不兼容，
或缺少 per-channel quant 参数），而不是 launch 配置。

## 17. 更多验证（2026-10-31 12:38–12:42）

### 17.1 RMSNorm 多尺寸都正确

```
K=576  : sync 0, check 0/576  mismatch
K=1024 : sync 0, check 0/1024 mismatch
K=2048 : sync 0, check 0/2048 mismatch
```

RMSNorm 是当前唯一一个**跨尺寸数值正确**的 NPU 算子。

### 17.2 matmul 加 per-channel scale 无效

用 `xnnlSetTensorDescriptorQuantParam(wDesc, perChn=true, asym=false, scale=1.0,
zp=null)` 给 weight 挂 per-channel scale（`O3NPU_WSCALE=1`）：

```
xnnl_shim: set_quant(w, perChn=1, n=64) -> 0
kernel name: MatmulOpPerChnSplitKernelL1B
firmware response fail status 3 2
```

调用成功但 kernel 名称、行为都不变，仍然失败。

### 17.3 unary HPC 管线确实建了 L2B 节点

abs 的 `HostPipelineCreator` 日志：

```
tensor[idx=0] shape=[576] rank=1 dataType=1 needBytes=1152 direction=0
tensor[idx=1] shape=[576] rank=1 dataType=1 needBytes=1152 direction=1
tensorIdx=0 l2bAction=ALLOC l2bType=RPING arL2bSize=65536
tensorIdx=0 l2bAction=RDMA  memVA=0x456000
tensorIdx=0 l2bAction=WDMA  memVA=0x457000
CreateQueuePipeline OK
```

管线完整（alloc/read/write/free 都有），但 sync 后输出仍是初始值 0.5。
说明 unary kernel 本身没有产生数据（与 matmul 同类的 kernel 级问题）。

## 18. midaDeviceSetLimit 的正确签名（2026-10-31 12:45）

反编译 `midaDeviceSetLimit` 发现真实签名是 **三个参数**：

```c
midaStatus midaDeviceSetLimit(midaDeviceLimit limit, uint32_t value, midaDevice_t dev);
```

之前按两参数绑定调用导致 SIGSEGV。修正后：

```
o3npu: set_limit[0]=12582912 -> 0
```

可以成功把 L2B limit 从 8MB 改成 12MB/16MB。

但即使把 L2B limit 抬到 12MB / 16MB：

- matmul：仍 `midaStreamSynchronize -> 14`
- abs：instantiate 0，但输出仍是初始值（576/576 mismatch）

说明问题不在 L2B limit。

## 19. 模块逐个加载结果（2026-10-31 12:50）

`--modules` 逐个调用 `midaModuleLoadData`：

```
modules: 65
module[00] load=11  name=all_headers      ← 仅头文件，失败属正常
module[01..64] load=0
  ...
module[39] load=0 size=23376  name=matmul
module[44] load=0 size=51448  name=rmsnormalization
```

**matmul 模块本身加载成功**，所以不是"kernel 没加载"。

### 19.1 kernel ABI 字段（.k_abi 第二字）

```
MatmulOpPerChnSplitKernel/L1B/SpecializedN : 0x102e0000
RMSNormalizationHalf_BlockReduce_TMA_L2B    : 0x10260000   ← 已验证能跑
unaryABSHalf                                : 0x20260000
```

matmul 的 ABI 字与能跑的 RMSNorm 不同（0x2e vs 0x26），怀疑 kernel 编译目标
与固件版本不完全匹配，但尚无直接证据。

### 19.2 weight 内存 flag 的影响

`midaMemoryAlloc` 的 flag 从 5（默认，cacheable）换成 1/3 时：

```
O3NPU_WFLAG=1 : midaStreamSynchronize -> 10 (midaErrorFWTimeout)
O3NPU_WFLAG=3 : 同上
```

比原来的 `fail status 3 2 / midaErrorLaunchFailure(14)` 进了一步（固件开始执行但超时），
说明内存属性确实影响 matmul kernel 的行为。下一步值得调查 vendor 实际使用的 flag。

## 20. 内存 flag / align 位语义（2026-10-31 13:00）

反编译 `midaMemoryAlloc` → `MemoryImpl::Alloc`：

```c
midaMemoryAlloc(midaMemory_t *out, midaDevice_t dev, uint64_t size,
                uint32_t flag, uint32_t align, const char *name)
// flag 被拆成 3 个 bit 传给底层 npu_buffer_alloc：
//   bit0 = flag & 1
//   bit1 = (flag >> 1) & 1
//   bit2 = (flag >> 2) & 1
// align 原样传给底层分配器
```

实测 flag 0..15：

| flag | 结果 |
|---|---|
| 0,2,4,6,8 | 分配失败（无 CPU 指针） |
| 1,3,9,11 | 分配成功，`midaStreamSynchronize -> 10 (FWTimeout)` |
| 5,7,13,15 | 分配成功，`-> 14 (LaunchFailure)` |

规律：bit0 必须为 1 才能拿到 CPU 映射；**bit2 决定走 device memory（launch failure）
还是 host memory（firmware timeout）**。flag=5（vendor 常用）是 device memory 路径。

给 `dbuf_reserve` 增加 `O3NPU_ALIGN` 后，align=128..2MB 都会把结果从
LaunchFailure 变成 FWTimeout，说明对齐参数确实影响分配路径，但仍未跑通。

## 21. 设备/子图配置探测（2026-10-31 13:05）

- `midaDeviceGet(1/2, ...)` 全部失败 —— 只有一个 NPU 设备（index 0）。
- `midaSubgraphConfig` 第一 dword 置 1/2/4/8 时 `midaSubgraphCreate` 直接
  SIGSEGV —— 该字段不是标量队列数，可能是回调/指针表，保持全 0 是对的。
- `midaDeviceSetLimit` 的 index 1..5 都可写（返回 0），但对 matmul 行为无影响。

结论：subgraph 配置和 device limit 都不是 matmul 失败的原因。

## 22. 运行时 patch 强制换 kernel（2026-10-31 13:08）

### 22.1 强制 specializedN

`xnnlMatmulCreateGraph` 里 `policyMode_` 是局部变量硬编码 0：
`cmp w8,#1` (0x7100051f) 决定走 specializedN 还是 splitMN。
在进程内 mprotect 后把该指令改成 `cmp w8,#0`：

```
xnnl_shim: policy insn @ 0x... = 0x7100051f
xnnl_shim: patched policyMode_ compare -> specializedN
xnnlMatmulCreateGraph -> 2 (NOT_SUPPORTED)   / 1 (BAD_PARAM)
```

SpecializedN 对 M=1 和 M>1 的 shape 都不接受，走不通。

### 22.2 强制非 L1B kernel

把 rodata 里的字符串 `MatmulOpPerChnSplitKernelL1B` 从第 25 字节截断成
`MatmulOpPerChnSplitKernel`：

```
xnnl_shim: kernel str now = MatmulOpPerChnSplitKernel
[matmul.cpp:244] kernel name: MatmulOpPerChnSplitKernel
firmware response fail status 3 2
```

**换成非 L1B 的 matmul kernel 仍然同样失败** —— 说明问题不在 L1B 变体，
而是整个 matmul kernel 家族在这个固件上都跑不起来。

## 23. xnnlSetDeviceVersion（2026-10-31 13:19）

反汇编 `xnnlSetDeviceVersion(handle, uint32_t)`：

```
0x5652cc  mov w0, wzr
0x5652d0  str w1, [x8]        ← handle+0x00 = deviceVersion
```

`xnnlCreate` 分配 handle 后只清零 +0x08..+0x40，**handle+0 从未初始化**，
所以 deviceVersion 默认是堆上的垃圾值。补调 `xnnlSetDeviceVersion` 测试：

| deviceVersion | matmul 结果 |
|---|---|
| 0x9200（chipRevision） | fail status 3 2 |
| 0x9000（chipModel） | fail status 3 2 |
| 1 / 2 / 3 / 0x100 / 0x200 | 全部 fail status 3 2 |

kernel 名称始终是 `MatmulOpPerChnSplitKernelL1B`，没有变化。

### 23.1 IPCDriver 的固件状态无名称表

`npu_msg_response_status_printf` 只打印 `response status %d`，没有
status→字符串映射；`firmware response fail status 3 2` 中的 3/2 无法进一步解码。

### 23.2 exec dump 无效

`midaSubgraphExecDumpDataToFile(exec, path)` 返回 0，但无论传文件还是目录都不产生文件。

## 24. matmul 结论汇总（2026-10-31 13:23）

经过 6 轮尝试，matmul 在该设备上的状态可以总结为：

| 检查项 | 结果 |
|---|---|
| 64 个 kernel 模块加载 | 全部成功（module[0] all_headers 除外，正常） |
| matmul 模块 | 加载 rc=0 |
| 图创建 | `create graph, ret=0` |
| TMA 描述符 / 共享内存 | 合法（sum 2048/2560 <= 131072） |
| L2B / VA / REGNUM / DMASPLIT / ALIGN / CU / PERF / DDR | 全试过，无效 |
| memory flag 0..15 | bit0 必须置位；device/host 路径分别表现为 launch failure / FWTimeout |
| 强制 specializedN | `NOT_SUPPORTED` / `BAD_PARAM` |
| 强制非 L1B kernel（字符串 patch） | 仍 `fail status 3 2` |
| deviceVersion 0x9200/0x9000/1/2/3/0x100/0x200 | 无变化 |

**唯一稳定可用的算子是 RMSNorm（K=576/1024/2048，数值 0 mismatch）。**

所有 matmul kernel 变体（L1B、非 L1B、SpecializedN）在该 NPU 固件
（UMD 0.7.13，chip 0x9000/0x9200）上都返回 `firmware response fail status 3 2`，
属于 kernel 级异常，不是调用参数或内存配置问题。

在找到新的固件/内核入口（例如 `libNPURuntime` + `.nb` 模型路径）之前，
继续在 `xnnlMatmulCreateGraph` 上调参不会改变结果。

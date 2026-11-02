# 离线准备状态（2026-11-03）

目标：在设备离线时把“上机调试/验收”需要的东西全部准备好。
设备命令统一用 `nix develop -c adb`，探针用 `nix build .#probe-npu-matmul` 构建。

## 1. 已完成并验证的离线项

| 项 | 证据/产物 |
|---|---|
| 设备提取 | `artifacts/device/20261103-015444/`，27 ok / 8 permission denied，`manifest.json` |
| Barrier 图结构 | RMSNorm DOT 对照；`midaBarrierNodeParams = uint32 mode`；GemmSplitMN DOT 已到 `1 queue, total node 2` |
| kernel 存在性 | `node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so`，含 `gemmSplitMNEpilogueFused*` |
| 参数顺序 | `xnnlAddGemmSplitMNKernelNode(graph, handle, queueId, ...)`，设备已确认 `addGemmSplitMN -> 0` |
| cache 读回顺序 | 读前 `dbuf_sync_dev`、读后 `dbuf_sync_cpu`；RMSNorm 数值正确 |
| 离线自检 | `tools/npu/offline-check.sh` |

## 2. `xnnlGemmSplitMNKernelReorder` 签名（Ghidra 0x7669b0）

```c
int xnnlGetGemmSplitMNKernelReorderSize(
    const void *inBox,      // xnnlTensorBox*
    const void *weightBox,  // xnnlTensorBox*
    const void *outBox,     // xnnlTensorBox*
    void       *param4,     // 当前实现里未使用，可传 NULL
    uint32_t   *sdpReorderSizeOut,
    uint32_t   *groupwiseScaleSizeOut,
    uint32_t    actMode,
    uint8_t     transposedW,
    uint32_t    configType);
```

```c
int xnnlGemmSplitMNKernelReorder(
    const void *inBox,
    const void *weightBox,
    const void *outBox,
    float      *sdpData,          // 现有 SDP/EW 数据指针，可 NULL
    uint8_t    *reorderedGwScale, // 现有 groupwise scale 输出缓冲
    void       *param6,
    uint32_t    actMode,
    void       *epilogueParams,   // xnnlEpilogueParams*
    uint8_t     transposedW,
    uint32_t    configType);
```

关键结论：当前 GemmSplitMN 测试参数是

```text
transposedW=1 actOp=5 gemmK=576 gemmN=576 configType=2
sdpBlkSize=0, gwqSmemBytes=0, groupwiseCnt=0
```

`xnnlGemmSplitMNKernelReorder` 实际只在 `sdpBlkSize>0` 或 `groupwiseCnt>0` 时
调用 `xnnlSdpDataReorder` / `xnnlGroupwiseScaleReorder`。当前 shape 两者都是 0，
所以**权重 reorder 不是当前输出为 0 的原因**，上机优先级应低于 L2B/stack。

## 3. L2B 参数结构（libxnumd 反汇编）

### `midaAllocFreeL2BNodeParams`（8 字节）

`AllocL2BNodeExecImpl::ParamsCheck`（0x6a3f8）：

```text
[+0x00] uint16 != 0      // size 低 16 位必须非零
[+0x04] uint16 == 0      // 必须为 0
```

`SetParams` 把前 8 字节整体交给 `PrepareAllocFreeL2B`。
可用：

```c
struct { uint32_t size; uint32_t reserved; } p = { 294912, 0 };
```

### `midaRWL2BNodeParams`（32 字节）

`ReadL2BNodeExecImpl::ParamsCheck`（0x6aa88）约束：

```text
[+0x00] & 0x3f == 0       // 地址 64B 对齐
[+0x08] & 0x3f == 0       // 地址 64B 对齐
[+0x10] & 0x3f == 0       // 长度 64B 对齐
[+0x14] != 0              // 类型/模式
[+0x18] >> 24 == 0        // 高位必须为 0
[+0x1c] < 8 MB
若 [+0x14] == 1，则 ([+0x18] | [+0x1c]) != 0
```

`SetParams` 会整块 32 字节复制进 node，并调用
`PrepareRWDMA(cmdType=2, params, out, w4)`。字段语义仍建议结合
`PrepareRWDMA` 的参数顺序确认，不能仅靠 ParamsCheck 反推全部含义。

### 当前 probe 上机开关

```text
O3NPU_BARRIER=1          # kernel 后加 Barrier(mode=1, queue=0)
O3NPU_OUT_PATTERN=1      # 输出预填 1.0，用于判定 kernel 是否写回
O3NPU_VIS=/path.dot      # 导出执行图 DOT
O3NPU_HANDLE_L2B=<bytes> # 写 xnnlHandle+0x10（L2B max size）
O3NPU_MEM_FLAG=5
O3NPU_GEMM=1 O3NPU_GEMM_PARAM3=0 O3NPU_GEMM_PARAM10=5 O3NPU_GEMM_PARAM18=2
O3NPU_TRANSPOSED_W=1
```

## 4. 设备恢复后的单变量顺序

1. 基线复测（已做过一次，保留复现命令）：

```bash
nix develop -c adb push result/bin/probe_npu_matmul /data/local/tmp/
nix develop -c adb shell "/data/local/tmp/o3p_chat.bin --shell 'cd /data/local/tmp; \
  O3NPU_MEM_FLAG=5 O3NPU_OUT_PATTERN=1 O3NPU_BARRIER=1 \
  O3NPU_GEMM=1 O3NPU_GEMM_PARAM3=0 O3NPU_GEMM_PARAM10=5 \
  O3NPU_GEMM_PARAM18=2 O3NPU_TRANSPOSED_W=1 \
  ./probe_npu_matmul --k 576 --m 576 --t 1 2>&1'"
```

期望：DOT `1 queue, total node 2`；当前实际输出仍为 `1.0` pattern。

2. 只改一个变量：`O3NPU_HANDLE_L2B=8388608` / `16777216`。
   若输出变为非 pattern，则 L2B pool 就是缺口。

3. 再试 `O3NPU_L2B=...`（实例化参数）与 `O3NPU_STACK=...`，每次只留一个变量。

4. 如果 L2B/stack 都无效，再回到 `xnnlGemmSplitMNKernelReorder`，用上面的签名做最小诊断，
   但注意当前 `sdpBlkSize=0`、`groupwiseCnt=0`，reorder 大概率不是直接原因。

## 5. 禁止项

- 不在设备上跑 `--funcs` 全模块扫描（上次挂死并掉线）。
- 不用 0x40 全零参数硬猜 `midaBarrierNodeParams` / L2B 结构；本次已经因此 segfault 过。
- 不同时修改 barrier + L2B + reorder；一次只留一个变量。

# GemmSplitMN 参数顺序实测（2026-11-01 20:42）

设备：`lhasa` 2608BPX34C，NPU UMD `0.7.13`，system `libxnnl.so`（64 modules）。

## 结果一：`xnnlAddGemmSplitMNKernelNode(graph, handle, ...)` 顺序修正后 node 构建成功

之前：
```
xnnl_shim: addGemmSplitMN(box) -> 3
kernel 'gemmSplitMNEpilogueFusedKernel' not found in any discrete module
```

修正为 `(subgraph, handle)` 后：
```
xnnl_shim: addGemmSplitMN(box, p3=2 p10=5 p18=2) -> 0
```

结论：**此前 `umd status=11` 的根因是 graph/handle 两个参数传反，不是 kernel 缺失。**
`xnnlAddGemmSplitMNKernelNode` 的 x0 必须是 graph/subgraph，x1 必须是 xnnl handle。

## 结果二：`param3` 是 queue id，单节点子图必须传 0

`param3=2` 时：
```
addGemmSplitMN -> 0
midaSubgraphInstantiate -> 1
[subgraph_impl.cpp:1686] queue id set is not continuous:2
[subgraph.cpp:359] subgraph instantiate fail
```

`param3=0` 时 instantiate 通过，launch/sync 也返回 0：
```
addGemmSplitMN(box, p3=0 p10=5 p18=2) -> 0
stack_info rc=1 size=0 mem=0x0
K=576 M=576 T=1 max|NPU-CPU|=24.4512 MISMATCH
output[0..7] = 0,0,0,0,0,0,0,0
```

## 结果三：kernel 已选中 `gemmSplitMNEpilogueFusedWithSiluKernel`

`O3NPU_XNNL_LOG=0` 的关键日志：

```
[GemmSplitMNPolicyKey] transposedW=1 actOp=5 gemmK=576 gemmN=576 configType=2
[GemmSplitMNKernel] Policy hash: 0xAFEC221749EF053B (M=1 K=576 N=576), configType_=2
final blockN=256
final blockM=1
weight globalDims:576,576,1,1 boxDim:576,256,1,1
weightTmaLsDesc ... locMemType:1 (L2B)
weight weightL1bBytes_=294912
inSmemBytes_=1536, weightL1bBytes_=294912, ... sharedMemBytes_=131072
actMode=5, allHeadNum_=0
transposedW_=1
configType_=2
kernel name: gemmSplitMNEpilogueFusedWithSiluKernel
thread block dim XYZ: {512, 1, 1}
dispatch dim XYZ: {1, 3, 1}
stack mem is null
```

所以当前问题已经不是“找不到 kernel”，而是 **kernel node 已加入、instantiate/launch/sync 都成功，但输出仍是 0**。
下一步应查：
1. `weightL1bBytes_=294912 > sharedMemBytes_=131072` 时是否需要额外的 L2B 分配/读写节点；
2. `stack mem is null` 是否意味着 GemmSplitMN 的 kernel stack 没有分配；
3. `midaSubgraphExecGetStackInfo` 返回 `rc=1 size=0` 的含义；
4. 对比厂商 `xnnlLlmDecodeFfnLayer::createGraph` 是否在 `createGraphWithKernelOp` 之后还调用了 `addAllocNodes` / `AddL2b*`。

补充：`O3NPU_PATH=graph`（走 `midaGraph*` 路径）在同样参数下直接 segfault（139），
raw graph 路径暂不可用。

## 结果四：离线复核发现读回 cache 顺序错误（待复测）

`tools/npu/o3npu.c` 的 `o3npu_matmul_run` 在 `stream_sync` 后先调用了
`dbuf_sync_cpu`（`midaMemoryCacheStart`），再读输出；而 RMSNorm/unary 探针的
正确顺序是同步后先 `mem_cache_end`（invalidate），再读。已在两个 launch 分支改为：

```
dbuf_sync_dev(a, &m->out);   /* 读前 invalidate */
... read ...
dbuf_sync_cpu(a, &m->out);   /* 读后 */
```

重新构建 `tools/probe/build.sh /tmp/probe_npu_matmul_p0` 通过；设备恢复后先复测
`param3=0 / p10=5 / p18=2 / transposed_w=1`，若输出非零则 cache 顺序就是第二个根因。

## 复现

```
adb push /tmp/probe_npu_matmul_p0 /data/local/tmp/
adb shell "/data/local/tmp/o3p_chat.bin --shell 'cd /data/local/tmp; \
  O3NPU_MEM_FLAG=5 O3NPU_GEMM=1 O3NPU_GEMM_PARAM3=0 \
  O3NPU_GEMM_PARAM10=5 O3NPU_GEMM_PARAM18=2 O3NPU_TRANSPOSED_W=1 \
  ./probe_npu_matmul_p0 --k 576 --m 576 --t 1'"
```

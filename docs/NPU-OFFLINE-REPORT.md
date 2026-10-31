# 离线逆向报告（截至 2026-10-31）

本报告汇总"不依赖设备也能推进"的全部工作与结论。设备相关记录见
`artifacts/probe-logs/2026-10-31-npu/RESULTS.md`。

## 1. 结论先行

- **NPU 可用**：RMSNorm（fp16, K=576）端到端执行且数值正确。
- **不需要编译器**：`libxnnl.so` 内嵌 64 个 ELF 模块 / 686 个 kernel，
  `xnnlCreate` 加载后即可用；旧结论已作废。
- **matmul 仍不可用**：构图/实例化/launch 都成功，但固件返回
  `midaErrorLaunchFailure` / `fail status 3 2` / `NPU_S_LOCKUP`。
- **unary（abs/neg/sign）**：第三参数是 `l2bMaxSize`（已确认），
  create/launch/sync 全部通过，但输出未写回（`HostPipelineCreator` 队列管线问题）。

## 2. 已完成的离线工作

| 项目 | 产物 |
|---|---|
| mida API 与内存模型 | `docs/NPU-MATMUL-API.md` §1–§17 |
| 64 模块 / 686 kernel 清单 | `tools/npu/list-kernels.mjs` |
| vendor 调用链（libxnlm → libxnnl → mida graph） | §19 |
| RMSNorm 对照实验 | §20 |
| 权重重排分析 | §21 |
| 状态与复现步骤 | §22 |
| 算子探针（rmsnorm/abs/neg/sign） | `tools/probe/probe_npu_matmul.cpp` `--ops` |
| unary 第三参数 = l2bMaxSize | §24 |
| ggml backend（host 验证 + NPU 自动回退） | `tools/ggml-o3npu/` |
| 一键离线自测 | `tools/ggml-o3npu/selftest.sh` |
| 构建环境固定到 `~/src/llama.cpp` | `tools/npu/build-env.sh` |

## 3. 关键结构

```
libxnlm.so   XNLM_Model_run / ModelImpl / xnnlDecoderLayer
   ├── libxnnl.so    xnnl*CreateGraph / xnnlMatmulDataReorder / HostPipelineCreator
   └── libxnumd.so   midaDeviceGet / midaMemoryAlloc(flag=5) / midaSubgraph* / midaGraph*
```

- `midaMemoryAlloc(flag=5)` 返回 CPU 指针 + 设备 VA，是 vendor 自己的用法。
- `midaLaunchConfig`：`cu_num`(+0), `performance_level`(+4), `ddr_ratio`(+8),
  `fw_timeout`(+0xc), `name`(+0x10), 5×`char[32]` stripe 名（+0x18..+0xB8）。
- `xnnlMatmulCreateGraph` 第 3 个参数是 `l2bMaxSize`（unary 已证实）。

## 4. 未解决

1. matmul 的 `xnnlMatmulCreateGraph` 为何生成"空转"图：
   - 需要 `O3NPU_XNNL_LOG=0` 打开 dispatcher 日志看 `kernel name` 与 M/K/N；
   - 需要确认 `xnnlMatmulDataReorder` / `xnnlGemmSplitMNKernelReorder` 的调用方式。
2. unary 的 `HostPipelineCreator::CreateQueuePipeline(..., 2, 1, 0)` 为何不写回输出：
   - 需要 decompile `HostPipelineCreator::CreateQueuePipeline` 与队列配置。
3. NBG/AON 路径（`libNPURuntime` + `*_cipher.nb`）尚未跑通，需要 `libmnne_crypto`。

## 5. 复现

```bash
# 构建
tools/probe/build.sh                       # /tmp/probe_npu_matmul
tools/ggml-o3npu/build.sh host /tmp/libggml-o3npu-host.so
tools/ggml-o3npu/selftest.sh               # 5 matmul + 3 copy 全 OK

# 设备（需要 root + SELinux 规则）
adb push /tmp/probe_npu_matmul /data/local/tmp/
adb shell "/data/local/tmp/o3p_chat.bin --shell 'cd /data/local/tmp; O3NPU_MEM_FLAG=5 ./probe_npu_matmul --ops --k 576'"
```

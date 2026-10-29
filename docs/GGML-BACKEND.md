# ggml / llama.cpp 集成

本文合并了后端集成进展、CPU 基线数据和离线/实机的分工分析。

---

## 第一部分：后端集成

目标：让 llama.cpp 通过动态后端机制，把 MUL_MAT 交给 NPU（我们的 `o3npu` 驱动）。

## 已完成：M1 管道验证

- `tools/ggml-o3npu/ggml-o3npu.cpp` 编成 `libggml-o3npu.so`（NDK、`-DGGML_BACKEND_DL`、链接 `libggml-base.so`）。
- llama.cpp 用 `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` 构建，
  `GGML_BACKEND_PATH=.../libggml-o3npu.so` 即可被加载。
- 设备上实测日志：

```
O3NPU: score asked
O3NPU: ggml_backend_init called
O3NPU: backend created
```

- `-ngl 99` 时调度器确实把 MUL_MAT 与视图类算子派给本后端，
  推理能跑完并出字（`Hello`），**不崩溃**。

## 已踩的坑

1. **视图算子必须处理**：`GGML_OP_RESHAPE`（op 36）会被派给拥有 buffer 的后端，
   不处理就 `graph_compute failed`。已按 no-op 处理
   `RESHAPE/VIEW/TRANSPOSE/PERMUTE`。
2. **`CONT`/`CPY` 用 memcpy**：M1 的简化实现。

## 已解决：数值正确（2026-10-29）

同一 prompt、`--temp 0`，用 **perplexity** 做数值判据（比单条文本可靠）：

| 模型 | CPU | 经后端 | 结论 |
|---|---|---|---|
| SmolLM2-135M **F16** | PPL 1.2730 | **PPL 1.2729** | 一致 |
| SmolLM2-135M **Q4_K_M** | PPL 1.2854 | PPL 1.2867 | 一致（量化权重回退 CPU） |

### 根因：buffer 的内存偏移用错了

`set_tensor` / `get_tensor` / `memset_tensor` 原本按
`buffer->context + tensor->view_offs + off` 计算地址。但张量在 buffer 内的偏移
存在 **`t->data`** 里；`view_offs` 只对视图张量有意义，普通张量是 0。
于是**每次拷贝都写到了 buffer 开头**，把权重全写坏 —— 而 matmul 本身一直是对的。

修正为 `t->data + off` 后，PPL 立刻与 CPU 一致。

### 离线单测

`tools/ggml-o3npu/selftest.cpp` 直接对拍我们的 matmul 与 `ggml_mul_mat`：

```
K=  64 M=  32 N=   8 a=f32 b=f32  max|diff|=2.38e-06  OK
K= 576 M=  64 N=  16 a=f32 b=f32  max|diff|=1.72e-05  OK
K= 576 M=  64 N=  16 a=f16 b=f32  max|diff|=6.00e-03  OK
K= 576 M= 576 N=   4 a=f16 b=f32  max|diff|=4.98e-03  OK
K=1536 M= 128 N=   4 a=f16 b=f16  max|diff|=6.10e-05  OK
```

（F16 的差异是 fp16 舍入，不是算法错误。）

同一套单测还覆盖了 `CONT`/`CPY` 路径（与 `ggml_cpy` 对拍）：

```
copy -> f32 transposed=0 max|diff|=0 OK
copy -> f16 transposed=0 max|diff|=0 OK
copy -> f32 transposed=1 max|diff|=0 OK
```


还有一个 ggml 的约束值得记住：**`ggml_mul_mat` 不接受转置的 src0**
（`GGML_ASSERT(!ggml_is_transposed(a))` 直接中止），所以"权重是转置视图"
这个假设从一开始就不成立。

### 主机验证配方

```bash
cmake -B build-host -S llama.cpp -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build build-host --target llama-cli llama-perplexity -j16
g++ -shared -fPIC -O2 -std=c++17 -DGGML_BACKEND_DL -Iggml/include -Iggml/src -Iinclude \
  -o libggml-o3npu.so tools/ggml-o3npu/ggml-o3npu.cpp -Lbuild-host/bin -lggml-base
LD_LIBRARY_PATH=build-host/bin GGML_BACKEND_PATH=$PWD/libggml-o3npu.so \
  build-host/bin/llama-perplexity -m model-f16.gguf -f text.txt -t 6 -c 128 --chunks 2 -ngl 99
```

## 下一步

1. 用 `nb[]` 重写 `o3npu_matmul`，先与 CPU 逐元素对齐（正确性优先）；
2. 正确实现 `CPY`/`CONT`（含类型转换）与 `cpy_tensor`；
3. 再把 matmul 换成真正的 NPU 路径：
   `xnnlMatmulCreateGraph` + `midaSubgraphInstantiate` + `midaStreamLaunchSubgraph`。
4. 性能基线：目前标量循环 pp 4.5 t/s / tg 2.9 t/s（无参考价值，仅证明链路）。

## 复现

```bash
adb push build-dl/bin/. /data/local/tmp/dlb/
adb push libc++_shared.so libggml-o3npu.so /data/local/tmp/dlb/
adb shell 'cd /data/local/tmp/dlb && \
  LD_LIBRARY_PATH=. GGML_BACKEND_PATH=$PWD/libggml-o3npu.so \
  ./llama-cli -m /data/local/tmp/smollm2-f16.gguf -p "Hello" -n 4 -t 6 -st -ngl 99'
```


---

## 第二部分：CPU 基线

目的：在把 NPU 接进 ggml backend 之前，先拿到纯 CPU 的参考性能，作为后续加速比的基准。

## 构建

```bash
NDK=/nix/store/...-android-sdk-ndk-29.0.14206865/libexec/android-sdk/ndk-bundle
cmake -B build-android -S llama.cpp -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF
cmake --build build-android --target llama-cli llama-bench -j16
llvm-strip cli-s bench-s      # 178 MB -> 14 MB, 118 MB -> 5.2 MB
```

产物是 AArch64 PIE，只依赖 `libc/libm/libdl`，直接推到 `/data/local/tmp` 就能跑。

## 结果（`llama-bench -p 32 -n 16 -r 3 -t 6`）

| 模型 | 大小 | 参数 | pp32 (t/s) | tg16 (t/s) |
|---|---|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | 98.9 MiB | 134.5 M | **323.17 ± 0.21** | **184.14 ± 9.36** |
| Qwen2.5-0.5B-Instruct Q4_K_M | 373.7 MiB | 494.0 M | **86.75 ± 1.66** | **48.26 ± 6.99** |

`llama-cli` 交叉验证（SmolLM2）：`Prompt 278.2 t/s | Generation 168.9 t/s`，与 bench 同量级。

## 一个必须记住的坑：线程数

同一模型、同一命令，只把 `-t` 从 6 改成默认的 10，结果变成：

```
pp32  8.28 ± 0.44 t/s
tg16  0.40 ± 0.19 t/s
```

**慢了 40~400 倍**。原因是这颗 SoC 是 10 核（1+3+6 的异构），
10 线程会同时压到大核和小核，调度器在小核上跑满导致长尾。

**结论：在这台设备上跑 llama.cpp 一律显式 `-t 6`（大核数），不要用默认值。**
后续所有 NPU 对比实验也必须沿用同一线程设置，否则数字不可比。

## 复现

```bash
adb push llama-bench /data/local/tmp/
adb push smollm2.gguf /data/local/tmp/
adb shell 'cd /data/local/tmp && ./llama-bench -m smollm2.gguf -p 32 -n 16 -r 3 -t 6'
```

## 接入方式：动态后端加载（不用改 llama.cpp 主干）

llama.cpp 原生支持把后端编成独立的 `.so` 并在运行时加载：

```cmake
-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON
```

```c
// ggml/src/ggml-backend-reg.cpp
ggml_backend_reg_t ggml_backend_load(const char *path);
void ggml_backend_load_all_from_path(const char *dir);   // 读 GGML_BACKEND_PATH
```

所以我们只需要产出 `libggml-o3npu.so`（导出 `ggml_backend_reg` 入口），
放进 `GGML_BACKEND_PATH` 目录即可被自动加载——**零主干修改**。

## 下一步

写一个最小 ggml backend：
- `supports_op` 只对 `GGML_OP_MUL_MAT` 返回 true；
- 内部把权重上传到 NPU 内存（`o3npu_alloc(..., cpu_visible=1)`），
  每步 `midaStreamLaunchSubgraph` 执行；
- 其余算子交给 `ggml_backend_cpu`，用 `ggml_backend_sched` 混合调度；
- 先要求 NPU 路径与 CPU 路径的输出**逐元素接近**，再谈速度。


---

## 第三部分：离线与实机的分工

结论先说：**数值正确性 100% 可以离线完成。**

原因很简单：ggml 后端是**纯用户态语义代码**，主机上有完全相同的调度器、相同的张量布局规则、
相同的 CPU 参考实现。设备唯一不可替代的东西是 **NPU 本身**。

## 一、可以完全离线（主机，不碰设备）

| 事项 | 为什么离线可行 | 工具 |
|---|---|---|
| 修 `o3npu_matmul` 的步长 bug | ggml 的 CPU 参考 `ggml_compute_forward_mul_mat` 就在源码里，按 `nb[]` 复刻即可 | `ggml-cpu.c` + 主机单测 |
| 视图算子语义（`RESHAPE/VIEW/TRANSPOSE/PERMUTE`） | 纯 ggml 定义，无硬件参与 | 主机端到端 |
| `CONT/CPY` 的类型转换与拷贝 | 同上 | 主机端到端 |
| buffer 的 `cpy_tensor` | 纯内存语义 | 主机单测 |
| 枚举"哪些算子会被派给后端" | 读 `ggml-backend-sched.cpp` 的分配逻辑静态推导 + 运行时打点 | 主机 |
| **端到端正确性**（`--temp 0` 输出一致性） | 设备上失败的那个测试，主机可以原样复现 | 主机构建 + 同一 GGUF |
| 后端调度开销测量 | 不含 NPU 时间，纯 CPU 调度 | 主机计时 |
| NPU 调用的参数映射设计 | 用已有的 `xnnl_api.h` 与 mida 反编译结果做静态设计 | 离线阅读 |

**离线闭环命令**（已验证可行）：

```bash
# 1) 主机构建 llama.cpp（动态后端模式）
cmake -B build-host -S llama.cpp -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build build-host --target llama-cli -j16

# 2) 主机构建我们的后端
g++ -shared -fPIC -O2 -std=c++17 -DGGML_BACKEND_DL \
  -Iggml/include -Iggml/src -Iinclude \
  -o libggml-o3npu.so tools/ggml-o3npu/ggml-o3npu.cpp -Lbuild-host/bin -lggml-base

# 3) 用同一个模型对比 CPU-only 与 offload 输出
GGML_BACKEND_PATH=$PWD/libggml-o3npu.so \
  build-host/bin/llama-cli -m smollm2-f16.gguf -p "The capital of France is" \
  -n 6 -t 6 -st --temp 0 -ngl 99
```

## 二、必须上设备（只有 NPU 才知道的）

| 事项 | 为什么不能离线 |
|---|---|
| `xnnlMatmulCreateGraph` 的真实参数语义 | 13 个参数的顺序/含义只能靠试；返回值与约束未文档化 |
| `midaSubgraphInstantiate` 的合法参数 | 空子图返回 1 说明有约束；`l2bSize/vaSize/stack_mem` 的取值未知 |
| NPU 执行结果与 CPU 的误差 | 硬件/量化才有 |
| 设备内存模型 | `o3npu_alloc(cpu_visible=1)` 指针的生命周期、`device VA` 何时非零、cache 同步 |
| 支持的 shape / 量化类型边界 | 驱动报错才有信息 |
| 真实性能（t/s、功耗、温升） | 硬件相关 |

## 三、为什么"先离线"能省大量时间

设备上每轮验证的代价：

- 468 MB 模型传输 + 加载；
- 标量循环下 pp 4.5 t/s，一次 6 token 的对比要 ~10 s；
- 出错只有一条 `graph_compute failed`，没有栈；
- 设备会掉线、会重启（root 与策略补丁全丢）。

主机上：

- 毫秒级迭代；
- 可以上 ASan/UBSan 直接抓越界与未初始化；
- 可以写单测，逐算子、逐 shape 对拍。

**把数值问题全部在主机上消灭，设备上就只剩一个变量：NPU 替换。**

## 四、验证：同一个 bug 已在主机复现

按上面的命令在主机构建后，用同一个 `smollm2-f16.gguf`、同样的参数：

```
CPU         : The capital of France is Paris
O3NPU offload: The capital of France is eleven trousers trousers
```

**与设备上的现象完全一致**（设备上是 `eleven Yangtze ...`，主机上是
`eleven trousers ...`——同为数值错误的随机发散）。
这说明：不需要设备就能复现、调试、验证修复。

## 五、建议顺序

1. **离线**：主机复现"Paris → 乱码"，用 `nb[]` 重写 matmul，实现 `cpy_tensor` 与类型转换，直到与 CPU-only 输出一致；
2. **离线**：加算子打点，统计"被派给后端的算子集合"，确认没有漏网的视图算子；
3. **设备**：把 matmul 的标量循环换成 `xnnlMatmulCreateGraph` + mida 执行，验证数值；
4. **设备**：性能与稳定性。



---

## NPU 路径的离线进展（未完成）

目标是把 `o3npu_matmul` 的标量循环换成真正的 NPU 调用：

```
xnnlMatmulCreateGraph -> midaSubgraphInstantiate -> midaStreamLaunchSubgraph
```

离线需要的三个语义都没有文档，只能从二进制里读：

| 目标 | 状态 |
|---|---|
| `xnnlMatmulCreateGraph`（15 个参数） | 已反汇编（`libxnnl.so` +0x5d386c，2516 字节）；参数到结构体的落栈与调用序列可见，但**未能恢复语义** |
| `npu::umd::SubgraphImpl::Instantiate`（解释 `err=1`） | 未开始 |
| `npu::umd::DeviceImpl::Init` | 未开始 |

**阻塞点**：本机没有可用的反编译器（Ghidra 未安装，IDA/Kuna/Hopper 均未配置），
纯 ARM64 手读 2.5 KB 函数来还原 15 个参数的成本过高。可行的推进方式：

1. 安装 Ghidra 后重新分析（REA 的 `analyze_function` 即可直接产出伪代码）；
2. 或者上设备用**试探法**：构造最小 matmul（K=N=M=16），逐参数变化看返回值与结果，
   通常几次就能锁定参数顺序。

在此之前，NPU 路径的代码不写死猜测的语义，避免产生看似可用实则错误的实现。

---

## 第四部分：NPU 路径接入（2026-10-30）

第三部分最后一段"NPU 路径的代码不写死猜测的语义"已经被替换：语义不再靠猜，
`docs/generated/mida_api.h` 与 `docs/generated/xnnl_api.h` 给出了完整结构体与函数原型。

### 代码结构

| 文件 | 作用 |
|---|---|
| `tools/npu/o3npu.c` | mida 绑定、设备缓冲池、subgraph 生命周期、`o3npu_matmul_run()` |
| `tools/npu/xnnl_shim.cpp` | 唯一需要 C++ 的部分（`xnnlSetTensorDescriptor` 收 `std::string`） |
| `tools/npu/o3npu.h` | 对外的 C 接口 |
| `tools/npu/vendor/` | 从 `docs/generated/` 复制的 vendor 头（含 `mida_types.h`，可直接编译） |
| `tools/probe/probe_npu_matmul.cpp` | 设备探针，走的就是后端同一条路径 |
| `tools/ggml-o3npu/ggml-o3npu.cpp` | `GGML_OP_MUL_MAT` → NPU，失败自动回退参考实现 |

### 环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `O3NPU_NPU` | `auto` | `auto` 能用就用；`0` 强制参考实现；`1` 必须成功 |
| `O3NPU_TRANSPOSED_W` | `1` | `1` 权重保持 ggml 布局并让 xnnl 转置；`0` 由我们预转置 |
| `O3NPU_L2B` / `O3NPU_VA` | `0` | `midaSubgraphExecParams` 的 l2bSize / vaSize |
| `O3NPU_CU` / `O3NPU_FW_TIMEOUT` / `O3NPU_STRIPE` | `1` / `10000` / 空 | launch config |
| `O3NPU_TRACE` | 关 | 打印 device/stream 与 `midaSubgraphExecPrintInfo` |

### 构建

```bash
tools/probe/build.sh                 # -> /tmp/probe_npu_matmul (arm64)
tools/ggml-o3npu/build.sh android    # -> /tmp/libggml-o3npu.so (arm64)
tools/ggml-o3npu/build.sh host       # -> /tmp/libggml-o3npu-host.so (x86_64, 回归用)
```

### 已完成的主机回归

- `selftest`：5 个 matmul + 3 个 copy 用例全部 OK（与改动前一致）。
- `llama-perplexity -m smollm2-f16.gguf -f README.md -t 6 -c 128 --chunks 2`：
  CPU `PPL = 4.1317`，经后端（参考实现回退）`PPL = 4.1319`，一致。

### 尚未完成

设备上的真实执行。判据与实验顺序见 `docs/NPU-MATMUL-API.md` §17 与本轮计划：
先看 logcat 的模块加载，再跑 `probe_npu_matmul --sweep`，最后才是 `llama-cli`。

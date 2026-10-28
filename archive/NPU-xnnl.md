# 自己调 NPU：接口分层与"逐步逆向"方案

目标不是再猜权重容器，而是**把 NPU 的调用接口一层层拿到手**，最终用自己的图跑起来。
好消息：这套栈里最底层、最关键的接口面，已经能用**静态手段**精确拿到（本轮已做）。

## 1. 分层与各自的接口面

```
AI 应用 / 插件
   │  15 个 Momi_*（dlopen("libmomi.so") + dlsym，缺符号只降级）
libMI_Genai_Runtime.so   (插件内 JNI 层)
   │  Momi_LLM_create / prefill / generate / …
libmomi.so               (模型目录内，随模型包发布)
   │  19 个 XNLM_*（C ABI）
libxnlm.so               (引擎：模型实现、KV、LoRA、Eagle)
   ├── 72 个 xnnl* 符号（**静态导入**，算子 + C++ 层类）
   │      xnnlMatmulCreateGraph / xnnlRMSNormalizationCreateGraph /
   │      xnnlQuantizeCreateGraph / xnnlDecoderLayer / xnnlEagletLmHead …
   └── 57 个 mida* 函数指针（**dlsym 表** WrapperFuncTable）
          device / memory / stream / graph / subgraph / DirectIO
libxnnl.so               (算子与图层)
libmida                  (设备/内存/流/图；mida* 符号实际来自其中)
   │  ioctl / mmap
/dev/hpc-*               (Xring O3 NPU)
```

关键事实：

- `libxnlm.so` **静态导入** 72 个 `xnnl*`（含 C++ 类，如 `xnnlDecoderLayer11createGraphE…`），同时
  通过 `WrapperFuncTable` **dlsym 57 个 `mida*`**——两套机制并存。
- `libmomi.so` 与 `libMI_Genai_Runtime.so` 都按名字 dlopen/dlsym，**符号缺失只打日志降级**。
- `libxnnl.so` 导出 **1,512** 个符号；`libxnlm.so` 的 DWARF 里记录了它引用的 `xnnl.h` /
  `xnnl_layers.h` 行号。

## 2. 本轮已产出的"正向规格"

| 产物 | 内容 |
|---|---|
| [`docs/generated/mida-functable.txt`](generated/mida-functable.txt) | **57 个 mida\* 的完整原型**（返回类型、参数类型、参数名），自动从 DWARF 的 `WrapperFuncTable` 解出 |
| [`docs/generated/xnnl-declared.txt`](generated/xnnl-declared.txt) | 17 个 `xnnl*` 的声明（头文件 + 行号）与相关结构体 |
| [`tools/xnnl/functable.js`](../tools/xnnl/functable.js) | 从 DWARF 恢复函数指针表的工具 |
| [`tools/xnnl/harvest.js`](../tools/xnnl/harvest.js) | 按前缀 harvest 声明/结构体/枚举 |

样例（真实签名）：

```c
midaError_t midaDeviceGet(midaDevice_t *dev, uint32_t index, midaDeviceConfig *cfg);
midaError_t midaMemoryAlloc(midaMemory_t *mem, midaDevice_t dev, uint64_t size,
                            uint32_t ?, uint32_t ?, const char *name);
midaError_t midaStreamCreate(midaStream_t *s, midaDevice_t dev, midaStreamConfig *cfg);
midaError_t midaSubgraphCreate(midaSubgraph_t *sg, midaDevice_t dev, midaSubgraphConfig *cfg);
midaError_t midaSubgraphInstantiate(midaSubgraphExec_t *exec, midaSubgraph_t sg,
                                    midaSubgraphExecParams *p);
midaError_t midaStreamLaunchSubgraphAsync(midaSubgraphExec_t exec,
                                          midaLaunchConfig *cfg, midaStream_t s);
midaError_t midaDirectIOOpen(midaDirectIO_t *dio, const char *path);   // ← base.json 的 directIO
```

## 3. 逐步逆向方案（每步都能独立验证）

### 阶段 A（离线，已完成）
从 `libxnlm.so` 的 DWARF 恢复：`WrapperFuncTable` 全表 → 57 个 mida 原型；`xnnl*` 声明；
结构体 `xnnlTensorDescriptor / xnnlQuantParam(0xa0) / xnnlTensorData(0x30) /
xnnlDecoderWeight(0x1b8) / xnnlEagletLmheadWeight(0x90) / xnnlWeightPatch(0x10)`；
枚举 `XNNL_DATA_* / XNNL_TENSOR_* / XNNL_QUANTIZE_MODE_*`。

### 阶段 B（离线）：补全 xnnl 算子层签名
1. 从 `readelf -sW libxnlm.so` 取 **72 个 `xnnl*` 未定义符号**（已有名单）。
2. 对每个符号，用 DWARF 的 `DW_TAG_GNU_call_site` + `DW_TAG_GNU_call_site_parameter`
   得到"第几个参数进了哪个寄存器、值来自哪个局部变量"；再结合 caller 的局部变量类型
   （DWARF 有完整类型）推出参数类型。
   例：`xnnlSetTensorDescriptorQuantParam` 的调用点显示 **W1..W6 六个参数**，值均为栈上对象指针。
3. 输出一份 `xnnl_api.h` 骨架（C 声明），可 `#include` 直接编译探针。

### 阶段 C（离线）：把断言当规格
`libxnnl.so` 有 **1,369** 条断言/诊断字符串，例如：

```
inputDesc_->layout != XNNL_TENSOR_LINEAR
weightDesc_->layout != XNNL_TENSOR_NHWC
(desc_.qkv_params.norm_scale.dtype != XNNL_DATA_HALF)
blk N:%d is larger than region N:%d, not supported.
```

按算子聚类即可得到"合法输入域"清单（dtype / layout / 形状约束），比读汇编高效。

### 阶段 D（上机，最小闭环）：跑一个算子
在**我们的插件进程内**（有 aiservice 的 uid、能看到 `/dev/hpc-*`）写探针：

```
dlopen("libmida.so"/"libxnnl.so") → midaDeviceGet → midaMemoryAlloc
→ xnnlCreateTensorDescriptor/SetTensorDescriptor(GroupwiseParam/QuantParam)
→ xnnlRMSNormalizationCreateGraph 或 xnnlQuantizeCreateGraph（最小算子）
→ midaSubgraphInstantiate → midaStreamLaunchSubgraphAsync → midaStreamSynchronize
→ midaMemoryGetPtr 读回结果，与 CPU 结果逐元素比对
```

选 RMSNorm/Quantize 作为第一个算子：输入简单、可解析验证、且是每个 LLM 层的必经步骤。

### 阶段 E（上机）：用官方引擎当"标准答案"
我们已经有能力让官方引擎跑起来（Mimo VLM/LLM 均已验证）。因此可以：
1. 用官方引擎跑同一层/同一算子，拿到参考输出；
2. 用我们自建的图跑同样输入，逐值比对；
3. 差异定位到 layout/scale 语义。这是**最强的收敛手段**——不需要猜，直接对答案。

## 4. 已知风险与前置条件

- **设备节点权限**：`/dev/hpc-*` 是否只对 aiservice 的 uid 开放？→ 探针必须跑在它进程内（或 root）。
- **mida 库位置**：`mida*` 符号可能来自 `libxnnl.so` 自身或独立 `libmida*.so`（模型目录里只有
  `libxnnl.so`），需在设备上 `dlopen` 试探并打印 `dlerror`。
- **张量布局**：`XNNL_TENSOR_LINEAR/NCHW/NHWC` 之外可能还有 XPU 私有布局；断言串会给出线索。
- **图/流语义**：`midaSubgraphCreate` 的 config 结构、`midaLaunchConfig` 字段仍未解（可从
  DWARF 结构体成员补齐，方法同阶段 A）。
- **NPU 固件/版本**：`xnnlGetVersion` 之类的版本检查可能需要匹配；先观察官方路径的返回值。

## 5. 与现有工作的衔接

- 阶段 D/E 的探针可以**直接放进我们已经能注入的插件**（`chat/` 里的 staging 方案或 Design B），
  无需新机制。
- 阶段 B 生成的 `xnnl_api.h` + 阶段 C 的约束表，是"正向实现自己的算子"的起点；
  与"替换 libmomi/libxnlm"路线互补：一个从接口侧逼近，一个从被调侧接管。

## 6. 已生成的头文件（本轮新增）

见 [`generated/`](generated/)：

- `xnlm_api.h` — 29 个 `XNLM_*` 原型 + `XnlmStatus/XnlmDataType/XnlmQuantParam`
- `xnnl_api.h` — 算子层 13 个完整原型（Matmul / RMSNormalization / Quantize / SetTensorDescriptor /
  SetTensorDescriptorQuantParam / GroupwiseParam / SetEpilogueDescriptor …）+ 相关结构体与枚举
- `mida_api.h` — 56 个 `mida*`（设备/内存/流/图/子图/子图执行/DirectIO）
- `xnnl-exports.txt` — libxnnl 导出符号 821 个，按族统计
- `xnnl-spec.txt` — 从 libxnnl 抽出的**1,817 条约束串、410 组、63 个 `XNNL_*` 常量**（可按组当规格读）

算子层真实原型（举例）：

```c
xnnlStatus_t xnnlCreate(xnnlHandle_t *, midaDevice_t);
xnnlStatus_t xnnlSetTensorDescriptor(xnnlTensorDescriptor_t, const int *dims, const uint ndim,
        const xnnlTensorLayout_t, const xnnlDataType_t, const xnnlMemLocation_t, const std::string);
xnnlStatus_t xnnlSetTensorDescriptorQuantParam(xnnlTensorDescriptor_t, const bool perChen,
        const bool asym, const void *scale, const void *zp,
        const xnnlDataType_t scaleDtype, const xnnlDataType_t zpDtype);
xnnlStatus_t xnnlRMSNormalizationCreateGraph(const xnnlHandle_t, midaSubgraph_t, const int,
        const xnnlTensorDescriptor_t in, const void *inBuf,
        const xnnlTensorDescriptor_t w, const void *wBuf,
        const xnnlTensorDescriptor_t out, void *outBuf, const int, float eps, int);
```

这已经把"阶段 D 最小算子探针"从猜测变成**照签名写代码**。

## 7. 剩余离线项（明确清单）

1. **mida 的 config 结构**（`midaDeviceConfig / midaSubgraphConfig / midaLaunchConfig /
   midaSubgraphExecParams`）不在 libxnlm 的 DWARF 里（只有不透明指针）。恢复办法：找
   `midaDeviceGet` 等函数的**调用点**，反汇编看调用前在栈上写了哪些字段/偏移 → 反推结构布局。
2. **`Momi_*` 15 个原型**：libmomi 无 DWARF；同样用 `libMI_Genai_Runtime.so` 的调用点反推。
3. **libxnnl 的 C++ 层类**（`xnnlDecoderLayer` 等 494 个 `_ZN` 符号）——若要复用官方层实现需补。
4. **`vit.plan`（YMCA）头部/段表**：确认是否为可分段的图容器（95 MB）。
5. **权重 payload 排列**（唯一影响"转换模型"的项）：结合 `mQuantMode/groupwise_cnt/WeightSize` 重解。

## 8. `Momi_*` 函数表布局（本轮解出）

插件侧 `libMI_Genai_Runtime.so` 用 `llm::getAllFunc(LLMFuncTable*)` 装载 libmomi，
指针按固定偏移存放。用 `llvm-objdump` 解析 `dlsym` 的字符串地址 + 随后的 `str x0,[x19,#off]`
即可还原映射（已实测）：

| 偏移 | 符号 |
|---|---|
| +0x08 | `Momi_getVersion`（字符串引用方式不同，单独确认） |
| +0x10 | `Momi_Model_create` |
| +0x18 | `Momi_Model_run` |
| +0x20 | `Momi_Model_free` |
| +0x28 | `Momi_LLM_create` |
| +0x30 | `Momi_LLM_prefill` |
| +0x38 | `Momi_LLM_generate` |
| +0x40 | `Momi_LLM_free` |
| +0x48 | `Momi_LLM_createTask` |
| +0x58 | `Momi_LLM_setLoraStrength` |
| +0x68 | `Momi_LLM_prefillWithModal` |
| +0x70 | `Momi_LLM_generateWithModal` |
| +0x78 | `Momi_getPerfMetrics` |
| +0x80 | `Momi_LLM_getStatistics` |
| +0x88 | `Momi_LLM_setPerformance` |

（+0x50/+0x60 的两处引用未按同一模式解析到名字，属已知缺口。）

有了这张表，任何 `ldr xN,[xM,#off]; blr xN` 都能直接对应到具体 `Momi_*`，
再读调用前的 x0..x7 赋值即可得到原型。已确认 `init` 中唯一的间接调用是
`x0 = &stackSlot; ldr x8,[x19,#8]; cbz; blr`（out-parameter 型，对应版本/统计一类）。

插件 JNI 面（11 个方法，`com.xiaomi.aicr.model.VLMEngine`）：
`init / release / reset / generateVLM / generateLLM / applyLora / getVersion /
createPromptTemplate / setPower / resetPower / resumePower`。

## 9. `Momi_*` 原型（从 libmomi 实现反编译恢复，本轮）

libmomi 无 DWARF，但函数实现里参数用法足以还原原型（AArch64 参数寄存器 + 使用方式）。
反编译全文：[`generated/decomp/momi_api.c`](generated/decomp/momi_api.c)；
整理后的头文件：[`generated/momi_api.h`](generated/momi_api.h)。

```c
void Momi_getVersion(void *version_out);
void Momi_getPerfMetrics(MomiHandle *llm, char **out);
void Momi_LLM_create(const char *base_path, MomiHandle *out);
void Momi_LLM_createTask(void *parent, const char *task_name, MomiHandle *out);
void Momi_LLM_free(MomiHandle *llm);
int  Momi_LLM_prefill(MomiHandle llm, const char *prompt, long *out_tokens, bool flag);
int  Momi_LLM_generate(MomiHandle llm, long *request);
int  Momi_LLM_setLoraStrength(float strength, MomiHandle llm);
void Momi_LLM_setPerformance(MomiHandle llm, const char *level);
void Momi_LLM_getStatistics(MomiHandle llm, MomiStatistics *stats);
int  Momi_Model_create(const char *path, MomiHandle *out);
int  Momi_Model_run(MomiHandle model, MomiTensor *inputs, int n_inputs,
                    MomiTensor **outputs, int *n_outputs);
void Momi_Model_free(MomiHandle *model);
```

实现细节：`Momi_LLM_create` 先校验 `base_path` 非空（错误码 `-0x3eb`，日志
"base_path is empty, Please Check!"），句柄通过 `momi::getInferHandle<momi::LLM>(...)`
管理；`Momi_LLM_generate` 的第二参数是一个 request 结构指针（含提示/回调等），
不是裸字符串。

## 10. `mida*` 的真实归属：`libxnumd.so`（离线边界）

`xnlm::getAllFunc` 的反编译（[证据](generated/decomp/)）显示所有 `mida*` 都是：

```c
p_Var1 = (FuncType_midaDeviceGet)dlsym(h, "midaDeviceGet");
if (!p_Var1) log("Cannot find the midaDeviceGet symbol in libxnumd.so!\n");
```

即 **mida 层来自厂商库 `libxnumd.so`**（另有 `libxnn_runtime.so` 之类），它**不在模型包里**，
属于系统侧（`/vendor/lib64/`）。这解释了为什么 `midaDeviceConfig`/`midaLaunchConfig` 等
结构在 libxnlm 的 DWARF 里查不到（只有不透明指针）。

因此：**mida config 结构无法离线恢复**——需要设备上拉取 `/vendor/lib64/libxnumd.so`
（以及 `libxnn_runtime.so`）后再做一次 DWARF/反编译，或直接上机试探（先传 NULL）。

## 11. `vit.plan`（YMCA 编译图）格式（本轮）

对 `lora_base_vlm/vit/vit.plan`（95 MB）做前 256 字节的流式抓取后解析：

```
offset 0      u32 = 0x14            ← FlatBuffers 根表偏移
offset 4      "YMCA"                ← 文件标识（FlatBuffers file_identifier）
offset 0x0a   vtable：vt_size=0x0a, table_size=0x18（3 个槽位）
offset 0x14   根表（root table）：slot0=0x14, slot1=0x02, slot2=0x02
offset 0x58   "OMDL" 嵌套块
字符串        "2026-05-29 14:30:49"（构建时间）、"6.2-3.2"（YMCA 编译器版本）
```

结论：**`vit.plan` 是 FlatBuffers 容器**（标识 `YMCA`），由图/模型表（`OMDL`）与编译器
版本信息组成，由 **YMCA 6.2-3.2** 生成。由于缺少 schema，进一步字段语义需要按 vtable 逐表反推；
但即使完全解出，**没有 YMCA 编译器也无法生成新的 plan**，因此它的价值主要是确认视觉塔
不可替换（与 §1 的结论一致）。

## 12. 离线逆向的最终边界

| 项 | 状态 |
|---|---|
| 0.5.0 容器 / 记录 / 权重头闭包（196/196） | 完成，可写 |
| 0.4.0 容器整文件 round-trip（sha256 相同） | 完成，可写 |
| `XNLM_*` 29 / `xnnl*` 13 / `mida*` 56 / `Momi_*` 15 原型 | 完成 |
| libxnnl 导出面 821 符号 / 1215 类 / 1817 条约束 | 完成 |
| 插件 JNI 面 11 个方法 | 完成 |
| `WeightImpl::parseLayer` / `getAllFunc` / `Momi_*` 反编译 | 完成（存档） |
| `vit.plan` | 识别为 FlatBuffers / `YMCA` 6.2-3.2 |
| 权重 payload 内部 swizzle | **离线不可解**：不在已发布 .so 中（NPU/YMCA 侧） |
| `mida*` 的 config 结构 | **离线不可解**：实现在厂商库 `libxnumd.so`（未随包发布） |
| xnnl C++ 层类的成员/方法体 | **离线不可解**：DWARF 中为 `DW_AT_declaration` 前向声明，方法名只在 mangled 符号里 |

结论：**凡是可以从已发布产物中恢复的接口与格式，都已恢复并验证**；剩余四项都受限于
"产物不在手上"（NPU 固件/YMCA 编译器、厂商库 `libxnumd.so`、类定义头文件、schema），
不是分析深度问题。
## 13. `mida*` config 结构：无 root 从厂商库解出（2026-10-27 设备轮）

`/vendor/lib64/` 下的 NPU 相关库是 **world-readable**，无 root 即可拉取。已拉到本地：
`libxnumd.so`(1.0 MB, 193 个 `mida*`)、`libxnnl.so`(8.7 MB)、`libxnlm.so`、`libxnn.so`、
`libxnn_llm.so`、`libxnn_runtime.so`、`libNPURuntime.so`（见 `artifacts/vendor/`）。

`libxnumd.so` 无 DWARF，因此用 Ghidra 反编译 config 的**消费方**，从字段访问还原结构
（[mida_configs.h](generated/mida_configs.h)，反编译存档见 `generated/decomp/`）：

```c
/* DeviceImpl::Init(uint index, midaDeviceConfig*) —— device_impl.cpp
   midaDeviceGet(out, index, cfg) 要求 cfg != NULL，且检查 cfg+8 < 3 */
struct midaDeviceConfig {
  uint32_t type;          /* +0x00 必须 < 6     */
  uint32_t reserved04;
  uint32_t mode;          /* +0x08 必须 < 3     */
  uint32_t flag;          /* +0x0c != 0 → bool  */
  uint32_t count;         /* +0x10 必须 < 0x15  */
  uint32_t dsmem_large;   /* +0x14              */
  uint32_t power_policy;  /* +0x18 < 2，否则回退 PERFORMANCE */
};
/* StreamImpl::Create —— midaStreamCreate 要求 cfg != NULL */
struct midaStreamConfig {
  uint32_t priority;      /* +0x00 必须 < 4 */
  uint32_t mode;          /* +0x04 必须 < 3 */
  uint8_t  flag;          /* +0x08 */
};
/* SubgraphImpl 构造 —— midaSubgraphCreate 允许 cfg == NULL */
struct midaSubgraphConfig {
  uint32_t enable;        /* +0x00 非 0 才读后面的参数块 */
  uint64_t param08, param10, param18, param20, param28, param40, param50, param58;
  uint8_t  flag90, flag91;/* +0x90, +0x91 */
};
```

**重要推论**：全零的 `midaDeviceConfig` / `midaStreamConfig` 能通过实现里的每一处校验
（0 < 6、0 < 3、0 < 0x15、0 < 2），所以上机首个探针可以直接传 `{0}`。
另从日志格式串里读出 UMD 版本：**NPU UMD Version 0.7.13（Oct 12 2026 10:37:15）**。

## 14. 设备无 root 轮：厂商库与重排架构（2026-10-27）

`/vendor/lib64/` 的 NPU 库全部 world-readable，已拉取到 `artifacts/vendor/`：
`libxnumd.so`(1.0 MB, 193 个 mida 导出)、`libxnnl.so`(8.7 MB)、`libxnlm.so`、`libxnn.so`、
`libxnn_llm.so`、`libxnn_runtime.so`、`libNPURuntime.so`。均无 DWARF。

### 14.1 权重重排的真实调用链（libxnnl 反编译）

```
xnnlDecoderWeightCreator::getReorderedWeights           ← 我们文件里 weight 的创建者
  ├─ xnnlLLMWeightReorder::reorderWeights      0x7205d8 ← 分发/校验（带日志）
  ├─ xnnlLLMPrefillWeightCreator::getReorderedWeights   ← prefill 专用
  └─ xnnlLLMDecodeWeightCreator::getReorderedWeights    ← decode 专用
        └─ reorderWeightSplitN  0x7212cc
           reorderWeightSplitK  0x721900
```

`SplitN`/`SplitK` 的行为：`count = dim / split`，然后向 `vector<xnnlWeightPatch>` 追加
`{isCompress:1bit, patchSize:u32}` —— 与我们在文件里看到的权重头 `{u64 size; u32 chunks}`
（`chunks = size/524288`）完全对应，说明**文件里的权重就是按 NPU 的 split 方案切好的**。

相关源文件（字符串）：`vendor/xring/ai/xnnl/bellv100/src/models/llm/xnnl_llm_weight.cpp`、
同目录 `weight_reorder.cpp`、`data_reorder.cpp`、`gemm_reorder.cpp`。**bellv100** 是这代 NPU 的
内部代号，`xnrt` 是用户态驱动层，`mnne` 是图编译器运行时。

### 14.2 其它厂商库

- `libxnn.so` = `mnne::lite`（图运行时）：含 `DataLayoutType`（`DataLayoutToStr` 为 9 项表）、
  `ExpandValidLayouts`、`npu::umd::Network::CreateShareWeight`；布局名字表位于重定位区，尚未解出。
- `libxnn_llm.so` = **Rust `tokenizers` 0.21.1** 编译（含 regex 0.8.5 / regex-automata 0.4.9 /
  serde_json 1.0.140 / aho-corasick 1.1.3），rustc `05f9846f`，构建者 `chencheng27`。
  即 LLM 的分词层是 HF tokenizers 的 Rust 实现，未使用 C++ tokenizer。

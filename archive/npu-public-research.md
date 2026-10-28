# Public-source survey: XRing O3 / XNNL NPU stack

Scope: web search + page reading only. No binaries downloaded or executed.
Date of survey: 2026-10-27.

Search-channel caveat: several hosts were unreachable from this environment
(`docs.qualcomm.com`, `developers.google.com`, `github.com` via the page-fetch
tool) and DuckDuckGo/Mojeek returned anti-bot pages (HTTP 202 "anomaly" /
403), so their empty results were discarded as unreliable. Negatives below are
based on repeated exact-phrase queries through the working search channel plus
direct checks of the obvious hosts (GitHub org listings, Xiaomi kernel drop
branches, vendor doc indexes). Treat "no evidence found" as "not locatable by
these queries", not as proof of non-existence.

---

## Q1 — XNNL / xnumd / xnrt / mnne-lite / YMCA / bellv100 / XRing NPU

### What does exist: XRing O3 as a product

XRing O3 launched 2026-08-24 and is documented only at the press-release level.

- 快科技/MyDrivers, 2026-08-24 — <http://m.mydrivers.com/newsview/1145905.html>
  > 玄戒O3的NPU架构经过全面重构，专为Xiaomi MiMo端侧大模型打造，具备200TOPS张量算力与3.13TFLOPS向量算力，端侧AI性能相较上代提升45%。

  Also: 60 MB total cache (12 MB L2, 16 MB L3, separate GPU/NPU caches, 16 MB
  system-level shared), "82ns 静态时延".
- OFweek, 2026-08-25 — <https://ee.ofweek.com/2026-08/ART-8330-2801-30699930.html>
  > 在NPU架构方面，玄戒O3专为Xiaomi MiMo端侧大模型而生，拥有200TOPS张量算力、3.13TFLOPS向量算力

  > 小米18 Fold折叠屏机型将首发搭载玄戒O3芯片，计划今年9月上市

  This is the only public text tying the device to the SoC; it is marketing
  copy, with no ISA, driver, compiler, or file-format detail.

### No evidence found: the internal identifiers

Exact-phrase queries for each of the following returned **no relevant
material** — only unrelated collisions (an adult site and a SoundCloud artist
for "xnnl", Instagram/YouTube profiles for "xnrt", AMD XDNA / Xilinx XRT
material for "xn*" strings):

| Identifier | Result |
|---|---|
| `XNNL` / `xnnl` (as an NPU library) | no evidence found |
| `xnumd` | no evidence found |
| `xnrt` | no evidence found |
| `mnne` / `mnne::lite` | no evidence found (one query returned literally "No results found") |
| `YMCA` compiler | no evidence found |
| `bellv100` | no evidence found |
| `xring_o3_asic` | no evidence found |

There is likewise **no** Xiaomi developer documentation, NDK, SDK, or public
header for an in-house NPU. Xiaomi's public AI-runtime footprint is its
open-source MACE engine, which targets CPU/GPU and does not document an XRing
NPU backend.

Corroborating negative: a Xiaomi end-side LLM talk from March 2026 (before O3
shipped) describes deployment exclusively on third-party NPUs —
<https://www.jinchutou.com/shtml/9ac77bd8f11030d701b0e8e5af7c7413.html>
(AiCon 2026, 黄武伟, 小米/大模型算法工程师), slide text:

> 平台 高通/天玑 GPU 高通平台 NPU 天玑平台 NPU MiLM 端侧部署实践

i.e. as of that talk the public story is Qualcomm/MediaTek NPU targets, with no
XRing/NPU-toolchain mention.

### Kernel source drop: no O3 branch

`MiCode/Xiaomi_Kernel_OpenSource` had **267 branches** at time of check; the
newest are `-w-oss` (Android 16 generation). There is **no** branch matching
`lhasa`, `xring`, `o1`, `o3`, or any `-x-` branch. So no XRing O3 kernel source
(and therefore no `xnumd`/`xnrt` driver source) is public.

- <https://github.com/MiCode/Xiaomi_Kernel_OpenSource>

### Community reverse engineering that does exist

Same-device and sibling-device security research, **bootloader/kernel only, no
NPU content**:

- `Ronnie15143/lhasa-research` — same device (Xiaomi 18 Fold, `lhasa`, XRing O3,
  kernel `6.18.21-android17-5`). Bootloader unlock chain, patch-level audit
  (effective 6.18.44), 11 ruled-out attack surfaces. README is in Chinese.
  <https://github.com/Ronnie15143/lhasa-research>
- `ayyy7128/ghostlock-xringo1` — "GhostLock XRING O1 root loader for validated
  6.6.77 and 6.6.118 kernels". <https://github.com/ayyy7128/ghostlock-xringo1>
- `Lw5959/xring-cve-2026-98260` — XRING O1 / Android 17 / 6.18.21 CVE notes.

These confirm an active but small XRing RE community, and confirm that **nobody
has published NPU-stack work**.

### Hiring signal

Xiaomi recruits for this stack, e.g. 新业务部-多媒体AI编译器开发工程师
(<https://xiaomi.jobs.f.mioffice.cn/index/position/7278592248237490284/detail>)
and AI编译器工程师
(<https://xiaomi.jobs.f.mioffice.cn/referral/campus/position/7621124013242321161/detail>).
Both pages are JS-rendered; the description body is not present in the HTML, so
no internal framework names could be quoted from them.

---

## Q2 — `XNLM` container / `taiyi` (太乙) model format

**No evidence found.**

- Exact queries for `XNLM` as a model container, `XNLM0.5.0`, and
  `taiyi.model` returned nothing on-topic.
- "Taiyi" is a heavily overloaded name in public ML, all unrelated to Xiaomi:
  - IDEA-CCNL **Taiyi-Stable-Diffusion-1B-Chinese-EN** (PaddleNLP/flagai)
    <https://paddlenlp.readthedocs.io/zh/latest/website/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/>
  - **Taiyi-LLM**, a biomedical bilingual LLM (DUTIR-BioNLP)
    <https://github.com/dutir-bionlp/taiyi-llm>
  - **Taiyi** image-text embedding (Towhee)
  - a Wuhan startup also named 太乙 (长江日报, 2025-06-21)

  Any of these will dominate search results; none describes a Xiaomi container.
- Xiaomi's public MiMo releases are ordinary HuggingFace research weights, not
  device containers:
  - <https://github.com/XiaomiMiMo/MiMo-V2-Flash>
  - <https://mimo.mi.com/docs/en-US/quick-start/faq/others>

So the `XNLM0.x` magic + XOR'd JSON metadata + `.taiyi.model` ZIP packaging
appears to be entirely undocumented outside the firmware.

---

## Q3 — `mida*` API (device / memory / stream / subgraph / graph)

**No evidence found.** No headers, SDK docs, leaked stubs, or third-party
write-ups for `midaDeviceGet`, `midaMemoryAlloc`, `midaMemoryMap`,
`midaStreamCreate`, `midaSubgraphCreate`, `midaSubgraphInstantiate`,
`midaStreamLaunchSubgraphAsync`, `midaDirectIOOpen`, `midaGraphAddSubgraphNode`,
or `midaSubgraphExecDumpDataToFile`.

The *shape* of the API is, however, a well-precedented design, and these public
analogs are useful for reasoning about semantics:

- **Google LiteRT Dispatch API** (NPU runtime integration) —
  <https://developers.google.com/edge/litert/next/dispatch>
  > The dispatch API is a set of vendor implemented functions LiteRT will use to
  > manage a session with an accelerator's runtime. These functions cover device
  > sessions, subgraph execution and buffer movement between the host and device.

  Same decomposition as `mida`: device, session/stream, subgraph, buffer move.
- **Huawei Ascend Graph API**: `CompileGraph`, `ExecuteGraphWithStreamAsync` —
  graph compile + async execution on a stream.
  <https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendgraphapi/atlasgeapi_07_0099.html>
- **Qualcomm QNN**: `QnnGraph_create` / `QnnGraph_execute` / `QnnContext`
  (graph + context + async execute) — the mainstream mobile-NPU equivalent.
  <https://docs.qualcomm.com/nav/home/tensors_and_memory_layout.html?product=924033590759186372>
- **AMD Ryzen AI** ONNX Runtime EP docs — another vendor NPU runtime surface.

The `midaSubgraphExecDumpDataToFile` export is worth noting as an
*engineering/debug* affordance: it implies the UMD can serialize per-subgraph
tensor data, which is exactly the hook needed to validate a weight-layout
hypothesis on-device without a disassembler.

---

## Q4 — Prior art: how mobile NPUs tile/swizzle 4-bit weights

No public document describes the *exact* combination in this stack
(per-output-channel FP32 scale + per-64-element INT8 group scale + two's
complement nibbles + 32x32 tile / channel interleave). But every ingredient has
public prior art. The most useful references:

### 4.1 Block scales + nibble packing in widely-read open code (llama.cpp / ggml)

`ggml/src/ggml-common.h` — <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h>

```c
#define QK4_0 32

typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
```

A 32-element block with a single FP16 scale and 16 bytes of packed nibbles.
Crucially the quantization is **offset-binary, not two's complement**, and the
nibble packing **interleaves the two halves of the block**:

`ggml/src/ggml-quants.c`, `quantize_row_q4_0_ref`:

```c
const float d  = max / -8;
...
const float x0 = x[i*qk + 0    + j]*id;
const float x1 = x[i*qk + qk/2 + j]*id;
const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));
y[i].qs[j]  = xi0;
y[i].qs[j] |= xi1 << 4;
```

So: element *j* → low nibble, element *j + 16* → high nibble. That is a
concrete, documented "split-half interleave" that a hardware engine can read as
one 32-byte burst. **Contrast with this stack:** if XNNL really uses two's
complement, its unpack is `sext4(nib)` rather than `nib - 8`, and any ported
llama.cpp-style kernel must be adjusted accordingly.

`Q4_K` adds a *hierarchy* of scales — the closest public analogue to
"per-channel FP32 + per-group INT8":

```c
#define QK_K 256
typedef struct {
    ...
    ggml_half d;    // super-block scale for quantized scales
    ggml_half dmin; // super-block scale for quantized mins
    ...
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
```

256-element super-block, eight 32-element sub-blocks, each sub-block scale
itself quantized to 6 bits. Two-level scaling exists precisely to avoid storing
a float per small group.

### 4.2 Groupwise / per-channel scaling in the quantization literature

- **GPTQ** — <https://arxiv.org/abs/2210.17323> — one-shot 3–4 bit weight
  quantization; per-channel scales with group size typically 128; the canonical
  reference for "groupwise scale" semantics.
- **AWQ** — <https://arxiv.org/abs/2306.00978>
  > Protecting only 1% salient weights can greatly reduce quantization error.
  Activation-aware per-channel/per-group scaling, explicitly "hardware-friendly
  ... low-bit weight-only quantization" for on-device LLMs.
- **Microscaling (MX) data formats** — <https://arxiv.org/abs/2310.10537>
  > MX data formats that combine a per-block scaling factor with narrow
  > floating-point and integer types for individual elements.

  This is the formal name for the pattern in question: a shared scale per block
  (MXFP4 uses an 8-bit power-of-two exponent per 32 elements; MXINT8 uses an
  int8 scale). An **INT8 group scale over 64 elements** is essentially an
  integer-microscaling layout, and MX is the right vocabulary to describe it.

### 4.3 Patents on fine-grained scales and layout/tiling

- **US12412082B2** — "Fine-grained per-vector scaling for neural network
  quantization" (NVIDIA) —
  <https://patents.google.com/patent/US12412082B2/en>
  > Quantizing smaller sets of the parameters using a shared scale factor
  > improves accuracy compared with quantizing larger sets of the parameters.

  Claims per-vector (sub-channel) scale factors — the same motivation as a
  per-64-element INT8 scale.
- **US20250086445A1** — "Inner product convolutional neural network
  accelerator" (Intel) —
  <https://patents.google.com/patent/US20250086445A1/en>
  > a mapping circuit to access a three-dimensional input matrix stored as a
  > Z-major matrix

  Explicit vendor-defined major-order ("Z-major") and INT4 execution blocks
  ("a BNN (INT1) 604, a TNN (INT2) 608, and a CNN (INT4) 612"). Useful as
  evidence that non-obvious major orders are patented/claimed territory.
- **CN113469350B** (Chinese) — tile blocking for an NPU matrix unit, from the
  published text:
  > 将输入特征图与卷积权重矩阵按16x16的尺寸分块，将对应分块的数据送入矩阵计算单元完成计算

  A concrete 16x16 weight/activation blocking scheme — the same family as the
  32x32 tiles seen here, just a different tile size.

### 4.4 Fractal / tiled tensor formats (Huawei Da Vinci)

Ascend's `FRACTAL_NZ` is the best-documented "swizzled" tensor format in a
mobile-adjacent NPU (documented in CANN operator-development guides):

> FRACTAL_NZ是分形格式，如Feature Map的数据存储，在cube单元计算时，输出矩阵的数据格式为NW1H1H0W0

Source (CANN 8.0.RC2 Ascend C 自定义算子开发指南, PDF):
<https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC2/developmentguide/opdevg/Ascendcopdevg/CANN%208.0.RC2%20Ascend%20C%e8%87%aa%e5%ae%9a%e4%b9%89%e7%ae%97%e5%ad%90%e5%bc%80%e5%8f%91%e6%8c%87%e5%8d%97%2001.pdf>

and the community doc (NHWC → NC1HWC0 conversion rules):
<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha01/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html>

Prior art for the *reason* fractal formats exist (cube-unit feed patterns):
Da Vinci architecture paper, "Da Vinci: A Scaleable Architecture for Neural
Network Computing" (Huawei Munich Research Center):
<https://pc.schihei.de/content/files/2023/08/Da-Vinci---A-Scaleable-Architecture-for-Neural-Network-Computing--updated-v7-.pdf>

### 4.5 Other vendor documentation entry points

- Qualcomm, "Tensors and Memory Layout" (QNN/HTP tensor + weight layout):
  <https://docs.qualcomm.com/nav/home/tensors_and_memory_layout.html?product=924033590759186372>
- MediaTek NeuroPilot FAQ: <https://developer.mediatek.com/ai/642fd80a4054c120c42632ae.html>
- (Both are JS-rendered; bodies could not be read from this environment, so no
  quotes — listed as starting points only.)

### 4.6 Synthesis for this device

If the observed layout is "FP32 per-output-channel scale + INT8 per-64 group
scale + 4-bit two's complement", then relative to public practice it is:
- **MX-like** in structure (per-block shared scale, §4.2) but with an *integer*
  group scale rather than an E8M0/E4M3 float — closer to MXINT8 than MXFP4.
- **Unlike ggml Q4_0** in two ways that matter for any reimplementation: signed
  two's complement instead of +8 offset-binary, and 64-element groups instead of
  32-element blocks.
- Consistent with **32x32 tiling** being a matrix-unit feed format (cf.
  CN113469350B's 16x16 and Ascend's fractal tiles), i.e. an implementation
  choice to make each tile a contiguous burst, not a semantic requirement.

---

## Q5 — FlatBuffers graph format with identifier `YMCA`

**No evidence found.** No public reference to a FlatBuffers schema, compiler, or
graph container using the 4-byte file identifier `YMCA`, in English or Chinese,
alone or with "flatbuffers"/"NPU"/"compiler".

General context (standard FlatBuffers behaviour, not a vendor disclosure): a
FlatBuffers buffer begins with a uoffset to the root table, and if the schema
declares `file_identifier`, those **4 bytes sit at offset 4** and are checked by
`Verifier`. `YMCA` at offset 4 is therefore consistent with a schema-declared
identifier, exactly as `TFL3` marks TensorFlow Lite FlatBuffers. Format
reference: <https://flatbuffers.dev/>

Suggested cheap confirmation: enumerate other `.plan`/graph blobs in the
firmware and compare bytes 4–8 (`YMCA` vs `TFL3` vs others); a shared identifier
across many graphs plus an embedded semver string ("6.2-3.2") and build
timestamp ("2026-05-29 14:30:49") indicates a single schema-versioned emitter,
i.e. the "YMCA" toolchain name belongs to the *compiler*, not the graph.

---

## Bottom line

1. XRing O3 is publicly known only as a launch product (200 TOPS, MiMo-tuned);
   there is no public NPU SDK, driver source, or compiler documentation.
2. `XNNL`, `xnumd`, `xnrt`, `mnne`, `YMCA`, `bellv100` have **zero** public
   footprint — expect to reverse these from binaries only.
3. `XNLM` containers and `taiyi` packages are undocumented publicly; "Taiyi"
   searches are dominated by unrelated projects.
4. `mida*` is undocumented, but its API decomposition matches LiteRT Dispatch /
   Ascend Graph / QNN graph-context patterns, which is enough to infer
   lifecycle and ownership rules.
5. The int4 layout has abundant *component* prior art (MX formats, ggml Q4_K
   hierarchical scales, GPTQ/AWQ group scales, fractal/Z-major tiling) but no
   public source matches the exact scale/encoding combination.

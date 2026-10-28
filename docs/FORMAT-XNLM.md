# XNLM 权重格式与"现成模型能否转换"的结论

所有数字都由本仓库工具从**真实 CDN 文件**取得（HTTP Range，未整包下载），命令在文末。

## 1. 包结构（`*.taiyi.model` = ZIP）

VLM 基座包 `vlm_model_qwen3_0_7b_f3-1.taiyi.model`（702,394,069 B，21 条目）：

```
  413,798,400  llm/llava_qwen3.bin            ← LLM 权重（XNLM 容器）
  311,164,928  embedding/embedding_fp16.bin    ← 裸 FP16 矩阵：151936×1024×2 = 311,164,928（精确）
   16,572,881  tokenizer/tokenizer.json        ← HF tokenizer（标准格式）
       94,988,064  lora_base_vlm/vit/vit.plan  ← 编译好的 NPU 图（magic YMCA，95 MB）
          478  lora_base_llm/task.json         ← 后端/采样/kvcache/LoRA 配对描述
          982  lora_base_vlm/task.json
        1,820  base.json                       ← 架构参数（name/backend/dims/rope/swa）
  1,463,168 libmomi.so · 2,750,304 libMI_Genai_Runtime.so · 7,544,584 libxnlm.so
  9,278,376 libxnnl.so · 5,899,632 libtokenizers.so · 1,794,776/1,152,936 libc++*
```

结论：**包是 ZIP，条目与名字固定**；`embedding` 是**无头裸 FP16**（尺寸可精确推算）；
`tokenizer` 是标准 HF json；**视觉塔是预编译图**（`vit.plan`）；LLM 与 LoRA 用下面的 XNLM 容器。

## 2. XNLM 容器：两个版本、两种自描述程度

两个版本都以 16 字节 ASCII 魔数开头，随后是一个 **u64 字段**：
`0.4.0` 里它**确证**是 JSON 元数据长度（454，JSON 紧接其后且以 `}` 结尾）；
`0.5.0` 里它是 413（VLM）/ 776（mimo），其后**不是 JSON**（是二进制区），
该字段的确切语义尚未定论（元数据长度或 tensor 数等候选）。

| 文件 | 魔数 | u64 | 是否带 tensor 名 |
|---|---|---|---|
| LoRA 适配器 `text_extraction_lora.bin`（9,064,448 B） | `XNLM0.4.0:de3aeb` | **454** | **是**（JSON + 命名块） |
| VLM 基座 `llm/llava_qwen3.bin`（413,798,400 B） | `XNLM0.5.0:9eeb2e` | **413** | **否**（全文件扫描无任何名字） |
| mimo 基座 `llm/hybrid_swa_compressed.bin`（1,348,247,552 B） | `XNLM0.5.0:9eeb2e` | **776** | 否 |

`0.4.0` 已被完整解析（这给出了格式的"骨架"）：

```
[0..16)   魔数 "XNLM0.4.0:de3aeb"
[16..24)  u64 metaLen = 454
[24..478) JSON 架构元数据（见下）
[478..4096) 每层块大小表：u64 × 28 = 323584，其余补零，整体对齐到 4096
[4096..EOF) 28 个层块，每块 323,584 B：
              块首 = 该层 tensor 记录序列（u64 nameLen + name + 描述字段）
              之后 = 该层权重 payload
```

验证：`4096 + 28 × 323,584 = 9,064,448` **恰好等于文件大小**；`decoder.layers.N.self_attn.q_proj.lora_A`
的偏移对 N=0/1/2/27 分别是 `4096 + N×323584 + 8`，逐层精确吻合。

JSON 元数据（454 B，逐字段）：

```json
{"model_type":"llava_qwen3","hidden_size":1024,"num_attention_heads":16,
 "num_hidden_layers":28,"ctx_size":40960,"vocab_size":151936,
 "decoder_kv_heads":8,"intermediate_size":3072,"split_embedding":1,
 "v_head_dim":0,"attention_value_scale":1,"bottleneck_size":0,"head_dim":128,
 "lora_rank":16,"lora_alpha":16,"kv_data_bytes":1,
 "has_q_norm":1,"has_k_norm":1,"weight_compress":0}
```

即：**这个容器自己声明了全部架构超参**（且与 Qwen3‑0.6B 完全一致：hidden 1024、16/8 头、
head_dim 128、28 层、vocab 151936、intermediate 3072、QK‑norm 存在）。
tensor 命名规范也从文件里直接读出：
`decoder.layers.<N>.self_attn.{q,k,v,out}_proj.lora_{A,B}`、
`decoder.layers.<N>.mlp.{gate,up,down}.lora_{A,B}`。

`0.5.0` 的差别：**没有任何名字**，u64 之后直接是数据（尾部为 0 填充）。因此基座权重的
**层内 tensor 顺序、dtype、量化编码完全由编译进库的模型实现决定**：`libxnlm.so` 里可见
`vendor/xring/ai/xnlm/o3/src/impl/models/{qwen3_vl,qwen2_5_vl,hybrid_swa_compressed,
hybrid_swa_compressed_eagle}_model_impl.cpp` 与 `weight_impl.cpp`、`lora_weight_impl.cpp`，
以及 `XnlmTensorDesc`（`std::map<std::string, XnlmTensorDesc>`）、`actualLayerOffset/Size`
等符号。

## 3. 运行时/后端约定（`task.json`，930 B）

```json
{"type":"task_config","name":"llavaqwen3-Task",
 "llm":{"backend":{"type":"franklin3",
    "backend_extensions":{"max_input_len":4096,"context_size":4096,
      "performance_config":{"cu_num":4,"ddr_perf_level":6,"npu_perf_level":5}},
    "lora":{"lora_path":"text_extraction_lora.bin","lora_name":"text_extraction",
            "version":"20260713","base_version":"20260713"}},
   "sampler":{"type":"argmax"},
   "kvcache":{"type":"f3_kvcache"}},
 "llm_acc":{"cpu_perf":{"thread_num":0,"cpu_mask":"0x0f0"}}}
```

`franklin3` = F3（Xring O3）NPU 后端，与 `libxnlm` 里的
`Franklin_PRECOMPILE/.../vendor/xring/ai/xnlm/o3` 路径一致；`lora.version` 必须与
`base_version` 配对（引擎日志里也会打印 `LoRA Version` 与 `base-version` 的比对）。

## 4. 现成模型能不能转过去：分层结论

**可以转的**（有真实机会）：

1. **同架构的 LLM 权重**。官方 LLM 就是 Qwen3 系（hidden 1024 / 28 层 / 16‑8 头 /
   head_dim 128 / vocab 151936 / intermediate 3072 / QK‑norm），且这些超参在容器里是
   **显式声明**的。若目标模型（例如 Qwen3‑0.6B 的微调版）张量集合与形状完全一致，
   转换只需要：HF 名字 → XNLM 名字/顺序映射、按容器的 dtype/量化编码写入 payload、
   保留模板文件的头与层块布局。
2. **LoRA 适配器**：格式**完全自描述**（JSON + 命名块 + 层块大小表），可以由我们
   直接写出/改写；`weight_compress:0` 说明默认不压缩。
3. **embedding**：裸 FP16 矩阵，尺寸 = vocab×dim×2，直接可写。

**不能（或代价极高）的**：

4. **视觉塔**：`vit.plan` 是厂商编译器产出的 NPU 图（magic `YMCA`，95 MB）。没有
   Xiaomi 的 xnnl/bellv100 编译器就无法为新视觉塔生成——所以"换一个 VLM 的视觉编码器"
   这条路基本封死。
5. **新架构**：`0.5.0` 基座没有名字表，层内顺序/dtype 由**编译进去的 impl** 决定。
   换架构（Llama/Mistral/任意 MoE）需要厂商新增 impl `.so`，我们无法生成。
6. **量化编码未解出**：payload 具体编码（block 大小 / scale 类型 / 是否压缩）尚未定论。
   已排除"简单 nibble + 每块一个 FP16 scale"的交织假设（统计判别最高只有 90.8%，
   不是干净的 100%）；mimo 的文件名本身就带 `compressed`，提示可能有熵编码。

## 5. 决定性实验（设备离线也能做，建议下一步）

**用公开权重解出量化编码**，一次实验就能把"能不能转"变成确定结论：

1. 从 HuggingFace 取 `Qwen/Qwen3-0.6B` 的 `model.safetensors`，只 Range 取
   `model.layers.0.self_attn.q_proj.weight`（FP16/BF16，约 4 MB）。
2. 在我们已下载的 `llm/llava_qwen3.bin` 里定位同一张量（层块结构已解析，层 0 的块
   从 4096 开始；按 impl 顺序或按尺寸匹配）。
3. 对齐后试解：dtype（FP16/INT8/INT4）、block 大小、scale 位置与类型（对齐残差最小者胜）。
4. 若解出 → 写 `tools/xnlm/convert.py|js`（HF → 0.5.0 容器，模板式：沿用官方头/层块、
   替换 payload），并可用"改写官方模型的一个张量、其余保持不变"在设备上验证加载。
5. 若解不出（例如熵编码）→ 结论定为"权重只能经小米工具链导出"，转换路线关闭。

## 6. 复现命令

```bash
# 包内条目（不解压）
node tools/ziprange.js "<modelUrl>" --list
# 流式取某条目开头 N 字节（解压后的内容）
node tools/ziphead.js "<modelUrl>" "llm/llava_qwen3.bin" 65536 --out /tmp/head.bin
# 整条目下载并解压（小文件用；权重 413 MB 也可）
node tools/zipfetch.js "<modelUrl>" "text_extraction_lora.bin" /tmp/lora.bin
# 已下载文件的结构核对
node -e '...'   # 见本文件第 2 节的偏移/尺寸断言
```

已知常量（服务端 `artifacts/ai-config.json` 内）：VLM 基座
`…/f0cd9c71-1725-4447-a35b-32fc60b00427/vlm_model_qwen3_0_7b_f3-1.taiyi.model`；
LoRA `…/18934fbf-2a12-4d59-89d7-2d98483c5dc7/model_vlm_lora_text_extraction_f3-1.taiyi.model`；
mimo 基座 `…/e21ab4b9-1bef-45ba-af6f-cd8f6dfb5f3a/mimo_model_f3-1.taiyi.model`。
## 7. 补充证据（2026-10-27，payload 编码仍未解出）

- LoRA payload **不是 FP16**：在层 0 块（323,584 B）内扫描"连续合理 FP16 值"，
  最长只有 **21 个值（42 字节）**；若真是 FP16 权重，应出现数十 KB 的连续合理段。
  同时存在 **4,084 字节**长的"窄值段"（每字节 ≤8 或 ≥248），更像 int8 量化后的
  近零权重。基座 `0.5.0` 的早期区段则出现 `0x33/0x44/0x55/0x77` 这类半字节重复模式。
- 因此 payload 很可能是 **int8 / int4（+ 块内 scale）**，但 block 大小、scale 位置/类型、
  是否进一步熵编码（mimo 文件名即带 `compressed`）尚未确定。
- 已排除的假设：`[nibble 组][FP16 scale]` 简单交织——按组大小 8/16/32/64/128 与
  scale 2B/4B 逐一统计判别，"合理 scale 占比"最高只有 90.8%（组 128/2B），
  不是干净的 100%，说明不是这种布局。
- 顺带确认：`0.5.0` 基座文件中**不存在任何 tensor 名**（全文件扫描 `self_attn`、
  `embed_tokens`、`lm_head`、`blk.0` 均为零命中），而 `0.4.0` 的 LoRA 全部带名。

## 8. 容器已解出（三个真实文件算术验证）+ mimo 分析（2026-10-27 第二轮）

工具：[`tools/xnlm/header.js`](../tools/xnlm/header.js)（头/块表解析，`--json`/`--block`）、
[`tools/xnlm/scan.js`](../tools/xnlm/scan.js)（区域剖面：熵/nibble/int8 特征/FP16 连续段）。

### 8.1 通用容器布局（0.4.0 与 0.5.0 同构）

```
[0..16)     ASCII 魔数 "XNLM<ver>:<tag>"
[16..24)    u64 metaLen
[24..24+L)  meta blob（0.4.0 = JSON；0.5.0 = 二进制）
[..]        u64 块大小表：0, size0, size1, … 以 0 结束
[..4096)    补零，payload 从 4096 开始
[4096..)    块依次排列，长度取自块表
```

实测校验（`node tools/xnlm/header.js <file>`）：

| 文件 | 块数 | 块表 | 4096+Σ | 文件大小 | 结论 |
|---|---|---|---|---|---|
| VLM 基座 `llm/llava_qwen3.bin` | 29 | 9,150,464×28 + 157,581,312 | 413,798,400 | 413,798,400 | **精确** |
| LoRA `text_extraction_lora.bin` | 28 | 323,584×28 | 9,064,448 | 9,064,448 | **精确** |
| mimo 基座 `llm/hybrid_swa_compressed.bin` | 37 | 32,669,696×27 + 32,656,600×9 + 172,171,264 | 1,348,247,552 | 1,348,247,552 | **精确** |

即：**块表本身就是自描述的偏移表**，转换器可以据此精确计算每个块的写入位置。

### 8.2 两个模型的量级推断（由块尺寸/参数量反推）

| 项 | VLM（28 层，hidden 1024） | mimo（36 层，hidden 2048） |
|---|---|---|
| 每层块 | 9,150,464 B ≈ **0.58 B/参数**（4-bit + scale） | 32,669,696 B（每第 4 层少 **4,096 B**，9 层如此） |
| 尾部块 | 157,581,312 B ÷ lm_head 155,648,000 参数 = **1.012 B/参数 → 8-bit** | 172,171,264 B ÷ (151936×2048) = **0.553 B/参数 → 4-bit** |
| embedding | 独立文件 311,164,928 B = 151936×1024×**2**（FP16） | 独立文件 622,329,856 B = 151936×2048×**2**（FP16） |

mimo 的 36 层里"每第 4 层少 4096 B"是一个稳定结构特征（9 层），与其实例化参数里的
SWA（`sliding_window_size: 128`）周期一致，可作为解析校验点。

### 8.3 VLM 层块内部布局（已定位到张量级）

用 `scan.js --fp16runs` 在层 0 块（9,150,464 B）里找到：

| 偏移（块内） | 长度（值） | 判定 |
|---|---|---|
| 9,536 | 32,768 | q_proj 的 per-group scale（group=64：2048 行 × 1024/64=16） |
| 1,232,768 | 16,384 | k_proj scale（1024×16） |
| 1,849,024 | 16,384 | v_proj scale |
| 2,470,528 | 32,768 | o_proj scale（1024 行 × 2048/64=32） |
| 3,685,056 / 5,523,712 / 7,353,152 | 49,152 ×3 | gate / up / down scale（3072×16 或 1024×48） |
| 9,143,424 / 9,145,536 | 1,024 ×2 | input_layernorm / post_attention_layernorm（FP16） |
| 9,147,648 / 9,147,968 | 128 ×2 | q_norm / k_norm（FP16） |

**七个 scale 数组的长度序列 `[32768,16384,16384,32768,49152,49152,49152]` 与
q/k/v/o/gate/up/down 在 group=64 下的 scale 数量逐个精确相等**——这基本确定了
"4-bit 权重 + 每 64 个一组 FP16 scale"的量化方案与张量顺序。

**norm 与公开权重的比对**（关键证据）：用 HF `Qwen/Qwen3-0.6B` 对应张量逐值比较
（注意 safetensors 的 `data_offsets` 是**相对数据段**的，绝对偏移要 `+8+headerLen`）：

```
文件 @9,143,424 (FP16): 0.1660, 0.6719, 0.5430, 0.6914
HF  input_layernorm   : 0.1357, 0.7070, 0.5898, 0.7266
文件 @9,145,536       : 0.4590, 0.0002, -1.5000, 0.7422
HF  post_ln           : 0.4746, 0.0008, -1.4922, 0.7578
```

数值接近但**不完全相同**（差 ~3–15%）→ 官方权重是**小米微调/量化后的 Qwen3-0.6B 衍生版**，
不是公开 checkpoint 原样。含义：拿公开权重做"已知明文"只能做**统计/单调**对齐，
不能期望逐字节一致。

### 8.4 仍未解出的部分（唯一剩余关口）

4-bit 打包的**具体偏移与 nibble 约定**未定：以 q 的 scale 数组之后（块内 75,072）为起点，
试了 4 种约定（lo/hi 先、有/无符号、减 8 偏移）与 HF q_proj 的相关性，全部 ≈0；
考虑到官方权重是微调版，正确解码应仍有 0.8+ 的相关性，所以是**起点或打包方式不对**。
剩余两条确定路径：

1. **反汇编** `libxnlm.so` 里 `weight_impl.cpp` 的解析函数（魔数/块表已定位到数据结构，
   `XnlmTensorDesc` 是 `map<string, ...>`），读出每张量的 (offset, dtype, dims, block, scale)。
2. **暴力对齐**：以块内已知 anchor（norm 在 9,143,424、scale 数组位置）为约束，
   在层块内枚举数据起点/行填充/转置，用 q_proj 行 0 的相关性判定（单次枚举成本很低）。

两者任一成功，即可写 `tools/xnlm/convert.js`（模板式：沿用官方头/块表，替换 payload），
同架构模型的转换路线就打通；若反汇编显示是熵编码/私有压缩，则结论改为"必须官方工具链"。

### 8.5 mimo 的配置面（本次一并抓取）

`base.json`（`backend.type=o3`，async，`share_mode=true`，`directIO=true`，
`max_input_len=8192`，`cu_num=4/ddr_perf=4/npu_perf=2`，
`llm_info`: ctx 8192, vocab 151936, **head_dim 192**, **pos_id_dim 32**, **embedding_dim 2048**，
`kvcache=o3_kvcache`，rope_theta 5e6 + swa{theta 10000, window 128}）；

`lora_base_llm/task.json`（EAGLE 投机解码：draft_length 3、num_branches 4、
max_target_tokens 16、topk 4、`trimMap_24k.bin`、draft_vocab 24000）。

## 9. 突破：元数据加密方式 + 层块记录格式（2026-10-27 第三轮）

工具：[`tools/xnlm/inspect.js`](../tools/xnlm/inspect.js)（解密元数据、解析块表与层块记录）、
[`tools/xnlm/header.js`](../tools/xnlm/header.js)、[`tools/xnlm/scan.js`](../tools/xnlm/scan.js)。
DWARF：`libxnlm.so` 带完整调试信息（`.debug_info` 2.6 MB），用 `llvm-dwarfdump` 可直接读出
结构体/变量名——本轮的结构就是这样确认的。

### 9.1 元数据是"用魔数自身 XOR"加密的 JSON

`WeightImpl::parseConfig` 的 DWARF 变量里有 `configDecrypted`（vector&lt;unsigned char&gt;）、
`configSize`、`jsonVal`（nlohmann::json）、`layerSizeData`（uint64_t*）→ 元数据先解密再解析 JSON。
解密算法是 `xnlm::xorCipher`，密钥就是 **16 字节魔数本身**：

```
密文 23 44 6c 6d 12 43 5a 4a 55 56 66 11 1c 12 57 47 62 6e 6e
XOR  XNLM0.5.0:9eeb2e（循环）
＝   {  \n  ' ' ' ' "   m  o  d  e  l  _  t  y  p  e  "  :  ' '
```

`0.4.0`（LoRA）是**明文** JSON，`0.5.0`（基座）是同一 schema 的加密 JSON——所以之前只看到随机字节。

**VLM 基座（解密后）**：`model_type=llava_qwen3, hidden 1024, heads 16, layers 28, ctx 40960,
vocab 151936, kv_heads 8, intermediate 3072, split_embedding 1, head_dim 128, kv_data_bytes 1,
has_q_norm/k_norm 1, weight_compress 0`

**mimo 基座（解密后）**：`model_type=hybrid_swa_compressed, hidden 2048, heads 32, layers 36,
ctx 262144, vocab 151936, kv_heads 4, intermediate 8192, v_head_dim 128,
attention_value_scale 0.707, head_dim 192, sliding_window_size 128,
hybrid_layer_pattern=[1,1,1,0]×9, partial_rotary_factor 0.334, has_q_norm/k_norm 0,
**weight_compress 1**`

注意 `hybrid_layer_pattern=[1,1,1,0]×9` **正好解释**了 §8.2 观察到的"每第 4 层块少 4,096 B"
（9 个 0 层）——两个独立观察互相印证。

### 9.2 层块 = 逐张量记录（名字同为 XOR 加密）

0.5.0 的层块与 0.4.0 同构，只是名字被 XOR 加密、且**没有全局名字表**：

```
[u64 nameLen][name ^ magic][描述字段…][payload]   × 每层张量数
```

`libxnlm.so` 的 `XNLM_DTYPE` 枚举：`FLOAT32=0, FLOAT16=1, INT4=2, UINT4=3`。
`norms` 的描述是 `[u32 dtype=1][u32 size]` 后紧跟 FP16 数据（1024×2=2048、128×2=256，实测吻合）。

**VLM 层 0 的 13 条记录**（层 1 偏移完全相同 → 层内布局统一）：

| 记录偏移 | dtype | 名称 | 到下一记录 |
|---|---|---|---|
| 0 | 6 | `decoder.layers.0.self_attn.q_proj` | 1,227,840 |
| 1,227,840 | 6 | `decoder.layers.0.self_attn.k_proj` | 616,256 |
| 1,844,096 | 6 | `decoder.layers.0.self_attn.v_proj` | 616,000 |
| 2,460,096 | 6 | `decoder.layers.0.self_attn.qk_mm` | 2,752 |
| 2,462,848 | 6 | `decoder.layers.0.self_attn.qkv_mm` | 2,752 |
| 2,465,600 | 6 | `decoder.layers.0.self_attn.out_proj` | 1,205,312 |
| 3,670,912 | 6 | `decoder.layers.0.feedforward.w1` | 1,838,656 |
| 5,509,568 | 6 | `decoder.layers.0.feedforward.w2` | 1,838,656 |
| 7,348,224 | 6 | `decoder.layers.0.feedforward.w3` | 1,795,136 |
| 9,143,360 | 1 | `…self_attn_layer_norm.weight` (2048 B) | 2,112 |
| 9,145,472 | 1 | `…final_layer_norm.weight` (2048 B) | 2,112 |
| 9,147,584 | 1 | `…self_attn.q_norm.weight` (256 B) | 320 |
| 9,147,904 | 1 | `…self_attn.k_norm.weight` (256 B) | — |

命名是 llama.cpp 风格（`feedforward.w1/w2/w3`、`self_attn_layer_norm`），比 HF 多出
融合张量 `qk_mm`、`qkv_mm`（NPU 专用打包）。dtype=6 是"量化权重"（不在 `XnlmDataType` 枚举内，
应为 xnnl 侧的权重量化类型）。

### 9.3 payload 编码：确定到"4-bit + group-64 scale"，排列仍未解

以 q_proj 为例（HF 形状 2048×1024 = 2,097,152 参数）：

- 区间总长 **1,227,799 B ≈ 0.585 B/参数**；
- 区间内有一个 **32,768 值的 FP16 数组**（@block+9,536），恰好等于 2,097,152/64 →
  **每 64 个权重一个 FP16 scale**；
- 其余为打包数据（≈0.55 B/参数）。

**行主序已被穷举排除**：在 q_proj 整个区间内按步长扫描数据起点 × 4 种 nibble 约定，
与 HF `q_proj` 行 0 的最高相关性只有 **0.145**；列主序、tile(32/16/8) 行/列序、
group-major 等布局同样为 0。结论：payload 用的是 **NPU 友好的分块/重排布局**
（`libxnlm` 有 `ReorderWeight`、`libmomi` 有 `Layout::BSP`），排列规则写在
`WeightImpl::parseLayerDirectIO` 的代码里，需要按行号精读（DWARF 行号可直接定位）。

### 9.4 对转换可行性的影响

现在已经"可写"的部分大幅增加：

| 层 | 状态 |
|---|---|
| 容器头（魔数 + **加密 JSON** + 块表 + 4096 对齐） | **完全可写**（密钥已知） |
| 层块记录（名字 + 描述 + 张量区间） | **可写**（名称规范与顺序已解出） |
| norm 张量（FP16，dtype=1） | **可直接写** |
| 量化张量 payload（4-bit + group-64 scale） | 量化方案已定，**排列待解** |
| mimo（`weight_compress=1`） | payload 还叠了一层压缩，短期不可写 |

只要 §9.3 的排列被解出（读 `parseLayerDirectIO`，或继续用 HF 当已知明文枚举 tile 几何），
同架构（Qwen3-0.6B 系）的转换就能真正落地；LoRA（0.4.0）则已经是完全可写格式。

### 9.5 布局复测（修正 scale 索引后仍为负）

上一轮的布局测试有个方法错误：非行主序布局下，scale 应按**存储顺序**每 64 个 nibble 共享一个
（`sc[i>>6]`），而不是按行主序的组号。修正后重测：

| 布局 | 结果 |
|---|---|
| 行主序 `i=r*C+c` | 0 |
| 列主序 `i=c*R+r` | 0 |
| tile 64×64 / 32×32 / 16×16（行序、列序两种内部序） | 0 |
| group-64-major（每 64 列一块，块内按行） | 0 |

且在 q_proj 区间内对数据起点做步长扫描（步长 16 B，4 种 nibble 约定）最高相关性 0.145。
因此**不是这些初等排列**；剩下的可能是 NPU 私有重排（若干级 tile + 通道交错）、
payload 前还有一段（q_proj 区间内、scale 数组之前的 9,495 B 未知结构），
或权重存储 dtype 与猜测不同（dtype=6 的语义未定）。下一步应精读
`WeightImpl::parseLayerDirectIO`（DWARF 行号可直接定位）而不是继续枚举。

## 10. 量化 payload 追查（2026-10-27 第四轮）：已排除项与关键量化事实

工具：[`tools/xnlm/probe.js`](../tools/xnlm/probe.js)（在给定张量区间内自动定位 FP16 数组，
与 HF 同名张量做顺序/排序相关性、比值、取值多样性统计）。

### 10.1 已确认

以 VLM 层 0 `q_proj`（区间 `[0,1,227,840)`，块内偏移）为例：

```
region bytes 1227840; scale run @+9536 len=32769 (65538 B)
scale[0..3]  4.301e-4 2.146e-4 2.148e-4 1.842e-4
amax[0..3]   5.176e-2 4.175e-2 3.040e-2 2.954e-2      (HF 行主序 per-64 组 amax)
ordered log-corr   -0.0005   ← 与行主序组序**无关**（被置换）
sorted  log-corr    0.9589   ← 数值分布与 per-64 组 amax 分布高度一致
distinct scale values 177 of 32768   ← 只有 177 种取值（≈8 bit）
median amax/scale   454.4    ← 但**逐组比值不恒定**（首组是 100，中位 454）
```

结论（比上一节更严谨的版本）：

- 该 65,536 B FP16 数组**确实是与 per-group(64) 幅值相关的数组**（分布吻合），
  但**逐组比值不恒定**，所以"scale = amax/K、K 固定"的简单模型不成立；
  它更可能是**经过量化（177 档）或归一化**的组尺度，真正 dequant 还要乘一个
  尚未定位的全局因子/系数（可能在 9,495 B 的头部或描述字段里）。
- 由于只有 177 档、并行重复极多，**无法从 scale 数值反解组的置换**（秩对应法实测
  得到的映射是随机噪声），所以"用 scale 顺序当布局证据"这条路走不通。

### 10.2 已排除的布局/方法（都可复现）

| 尝试 | 结果 |
|---|---|
| 行主序 / 列主序 + 存储序 scale（`sc[i>>6]`） | 0 |
| tile 64/32/16（行内序、列内序两种） | 0 |
| group-64-major（每 64 列一块，块内按行） | 0 |
| 数据起点步长扫描（步长 16 B × 4 种 nibble 约定） | 最高 0.145 |
| "每组 64 权连续 32 字节" + 与 HF 前 40 组匹配 | 最高 0.36（负相关，不成立） |
| 用 scale 排序做秩对应反解置换 | 映射呈随机噪声（177 档导致并列过多） |
| 自相关找重复周期（lag 2..200） | 无任何峰值 |
| 熵剖面（4 KB 窗） | 全区间"结构化/混合"交替，无大段纯随机区 |

### 10.3 新得到的旁证

- `libxnnl.so` 的符号显示运行时存在 **`getReorderedBufferSize`** 与多个
  `xnnl*WeightCreator`（`xnnlLlmTransformerQkvWeightCreator`、`xnnlLLMPrefillWeightCreator`、
  `xnnlEagletWeightImpl::setWeights` 等）——权重在**加载时被重排**进 NPU 缓冲，
  说明文件里的排列是"转换器侧的源布局"，不是 NPU 最终布局。
- `xnnlTensorLayout_t` 的取值里有 `XNNL_TENSOR_LINEAR / NCHW / NHWC` 等，
  且 `libxnnl` 字符串里大量出现 `layout != XNNL_TENSOR_NHWC` 一类断言。
- `parseLayerDirectIO` 的 DWARF 局部变量：`layerBufHostAddr`、`layerBufDeviceAddr`、
  `gIndices`、`xnnlMatmulNum`、`xnlmMatmulNum`、`totalAllocSize`、`postRmsNameSize` 等，
  其调用目标全是 PLT 桩（`0x710d0` 等），符号名需查 `.rela.plt`（下一轮可直接做）。

### 10.4 下一步（明确、可执行）

1. **读代码而不是枚举**：对 `WeightImpl::parseLayer`（0x5ee6c）做 `pdc`/Ghidra 伪反编译，
   重点看它如何用描述字段算出每张量的数据指针与解包参数；同时把 `0x710d0/0x70f70/
   0x710c0/0x71120` 这几个 PLT 桩解析成 `xnnl*` 函数名（读 `.rela.plt` 即可）。
2. **找全局系数**：在 9,495 B 头部与描述字段里定位 `amax/scale` 那个可变系数
   （首组 100、中位 454 → 说明它是逐组的，可能又是一层 8-bit 量化的指数表）。
3. 若代码路径仍不可读，**上机动态验证**（需要 root/KSU）：让引擎加载我们改过 payload 的
   文件，观察报错/输出，用差分定位字段语义。

## 11. 容器可写性的字节级验证 + 类型系统（2026-10-27 第五轮）

### 11.1 三份头文件（从 DWARF 生成）

| 文件 | 内容 |
|---|---|
| [`generated/xnlm_api.h`](generated/xnlm_api.h) | **29 个 `XNLM_*` 原型**（含 `KVCache_*`/`Memory_*`/`Model_*`）+ `XnlmStatus`、`XnlmDataType`、`XnlmQuantParam`（含成员偏移） |
| [`generated/xnnl_api.h`](generated/xnnl_api.h) | **算子层签名**：`xnnlCreate/setTensorDescriptor/MatmulCreateGraph/RMSNormalizationCreateGraph/QuantizeCreateGraph/SetEpilogueDescriptor` 等 13 个完整原型 + `xnnlDataType_t/xnnlTensorLayout_t/xnnlQuantizeMode_t` 等枚举 + `xnnlQuantParam`(0xa0)/`xnnlTensorData`(0x30)/`xnnlDecoderWeight`(0x1b8) 结构 |
| [`generated/mida_api.h`](generated/mida_api.h) | **56 个 `mida*` 原型**（设备/内存/流/图/子图/DirectIO） |

关键枚举（实测）：

```
XnlmStatus:       SUCCESS=0 BAD_PARAM=1 NOT_SUPPORTED=2 TIMEOUT=3
                  INTERNAL_ERROR=4 EXCEED_MAX_SEQ_LEN=5 EXCEED_MAX_CTX_LEN=6 DEVICE_EXCEPTION_ERROR=7
XnlmDataType / xnnlDataType_t:
                  FLOAT32/FLOAT=0 FLOAT16/HALF=1 INT4=2 UINT4=3 INT8=4 UINT8=5
                  INT16=6 UINT16=7 INT32=8 UINT32=9 INT64=10
```

`XnlmQuantParam`（0x28）与 `xnnlQuantParam`（0xa0）完整成员：

```c
struct XnlmQuantParam {           /* xnlm 侧 */
  XnlmDataType mDtype; bool mIsPerchn; bool mIsAsym;
  XnlmDataType mScaleDtype; XnlmDataType mZpDtype;
  uint32_t mGroupwiseCnt; uint32_t mScaleSize; uint32_t mZpSize;
  uint32_t mWeightSize; uint32_t mChannelCnt; xnnlQuantizeMode_t mQuantMode;
};
struct xnnlQuantParam {           /* xnnl 侧 */
  bool is_asym; xnnlQuantizeMode_t mode; int groupwise_cnt; int channel_cnt;
  xnnlTensorData scale;      /* +0x10, 0x30 bytes */
  xnnlTensorData zp;         /* +0x40 */
  xnnlTensorData groupwise_scale; /* +0x70 */
};
```

`mWeightSize / mChannelCnt / mQuantMode / groupwise_cnt` 这几个字段正是权重 payload 缺的那部分语义
（"每组多少个、原始字节数、量化模式"），下一轮可据此重新解释层块里那段 9,495 B 头部。

### 11.2 字节级可写性验证

| 验证 | 命令 | 结果 |
|---|---|---|
| 0.4.0 容器 round-trip | `node tools/xnlm/lora.js rebuild /tmp/lora.bin /tmp/lora.rebuilt.bin` | **sha256 完全一致**（9,064,448 B）→ 容器框架（魔数+JSON+块表+4096 对齐+块）模型精确 |
| 0.5.0 加密元数据 | 解密 → 再 XOR 加密 | **sha256 完全一致** → 元数据可写（密钥=16 字节魔数） |
| 0.5.0 块表算术 | `tools/xnlm/header.js` | 三个模型均 `4096+Σ块 = 文件大小`，精确 |

即：**0.4.0 整文件、0.5.0 的头/块表/元数据，我们都能写**；仍只差量化 payload 的内部排列。

### 11.3 待解：层记录里的 `dtype=6`

权重记录的描述字段首值恒为 **6**，而两个枚举里 `6 = INT16`——与"每参数 0.585 字节"矛盾。
因此该字段要么不是 dtype（是"量化权重块"标记），要么属于另一套编号。结合 11.1 新增的
`mQuantMode/mWeightSize/mGroupwiseCnt`，下一轮应把该描述按 `XnlmQuantParam` 重新对齐解释。

## 12. 层记录结构完全解出（Ghidra 反编译 + 算术闭合）

用 `nixpkgs#ghidra` 12.1.4 headless 反编译 `libxnlm.so`（DWARF 原型保留），
产出：[`generated/decomp/weight_impl_parseLayer.c`](generated/decomp/weight_impl_parseLayer.c)。
`xnlm::WeightImpl::parseLayer(WeightImpl*, char* layerPtr, int layerId, bool isEagle)` 的
局部变量名直接来自 DWARF：`name / inputQuantParam / weightQuantParam / offset / nameSize /
weightQuantAddr / sdpAddr / gwAddr / weightAddr`。

### 12.1 记录布局（q_proj 实测，逐段偏移）

```
0         u64 nameLen = 33
8..41     name（XOR 魔数）"decoder.layers.0.self_attn.q_proj"
41        XnlmQuantParam prefill（0x28）：
            mDtype=6  mIsPerchn=0  mIsAsym=0  mScaleDtype=0  mZpDtype=6
            mGroupwiseCnt=0  mScaleSize=4  mZpSize=2  mWeightSize=0
            mChannelCnt=1  mQuantMode=1(PERTENSOR)
121        scale 数据（mScaleSize=4，64 对齐 → 到 185）
185        zp 数据（mZpSize=2，64 对齐 → 到 249）
249        XnlmQuantParam decode（0x28）
320        每通道 FP32 scale 数组：2048 × 4 B = 8192，到 8512
8532       全 0 填充到 9536
9536       groupwise scale 预填：32768 × 1 B（dtype=5 UINT8）
75072      64 字节头 {size=16896, dtype=0}   + 数据（FP32）
92032      64 字节头 {size=32768, dtype=5}   + 数据（UINT8 groupwise）
124864     64 字节头 {size=17408, dtype=0}   + 数据（FP32）
142336     64 字节头 {size=32768, dtype=4}   + 数据（INT8 groupwise）
175168     64 字节头 {size=1048576, dtype=2} ← **XNNL_DATA_INT4**
179264     ↑ 4096 对齐后的权重 payload（1,048,576 B）
1227840    下一条记录（k_proj）—— 179264 + 1048576 **恰好等于** 1227840 ✓
```

`dtype` 枚举与 xnnl 一致：`2=INT4`、`4=INT8`、`5=UINT8`、`0=FLOAT`。
两组 groupwise scale 分别对应 `prefill_groupwise_scale`（dtype=5）和
`decode_groupwise_scale`（dtype=4），各 32,768 个 = 2,097,152/64（**group=64** 确认）。

### 12.2 关键确认

- **权重就是 int4**，1,048,576 B = 2,097,152 参数 / 2，dtype=`XNNL_DATA_INT4`。
- **每通道 FP32 scale 是自然通道顺序**：与 HF q_proj 的行 amax 顺序相关 **0.965**（排序后 0.998），
  比值 amax/scale ≈ **101.7**（不是 8，说明还有 groupwise scale/全局系数叠加）。
- **文件里的字节是原样交给 xnnl 的**（`memcpy` → `reordered_weights.data`），
  即文件存的就是 NPU 期望的排列，引擎不做重排。
- nibble 直方图呈 V 形（0 与 15 双峰）→ **二补码 int4**（0 与 -1 最多），不是"signed+8"。

### 12.3 仍未解：tensor 内部排列（swizzle）

已排除（在**正确** payload 偏移 179,264 上重测）：行主序、列主序、tile16/32/64（两种内部序）、
group-64-major，配合 per-channel/groupwise scale 与两种 nibble 顺序，相关性均 ≈ 0。
512 B 窗口均值与 HF 行均值相关仅 0.41，且逐行最佳匹配呈"递减"（镜像/交织迹象）。

剩余候选：`XNNL_TENSOR_SWIZZLE`（枚举值 4）所定义的 NPU 私有 tile 交织，或按 32/64 通道
做位平面/扇区交织。下一步应从 `libxnnl` 的 `getReorderedBufferSize`/`setWeights` 代码
（Ghidra 可用）反推该排列，或用一个已知极小的张量（如 `q_norm` 之外的 `qk_mm`，2,752 B）
反推几何。

## 13. 通用记录解析器与全张量验证（2026-10-27 第七轮）

工具：[`tools/xnlm/record.js`](../tools/xnlm/record.js)
（`node record.js <model.bin> --layer 0 [--json]`）：靠"名字解密 + 权重头闭包"自动切分记录，
不依赖任何硬编码偏移。

### 13.1 通用规则（所有张量一致）

```
权重头 = { u64 size; u32 chunks }          // chunks = size / 524288
payload = 权重头偏移 + 4096
记录结束 = payload + size                  // 逐张量精确成立
```

`chunks` 实测：q_proj=2、k_proj=1、v_proj=1、out_proj=2、w1/w2/w3=3，
正好等于 `size / 524288`——所以该字段不是 dtype（此前误判），而是**分块数**。

### 13.2 VLM 层 0 实测（13 条记录，9 个 int4 权重块）

| 张量 | 记录跨度 | 权重头 | size | chunks | payload | 闭合 |
|---|---|---|---|---|---|---|
| self_attn.q_proj | 1,227,840 | 175,168 | 1,048,576 | 2 | 179,264 | exact |
| self_attn.k_proj | 616,256 | 87,872 | 524,288 | 1 | 91,968 | exact |
| self_attn.v_proj | 616,000 | 87,616 | 524,288 | 1 | 91,712 | exact |
| self_attn.out_proj | 1,205,312 | 152,640 | 1,048,576 | 2 | 156,736 | exact |
| feedforward.w1 | 1,838,656 | 262,592 | 1,572,864 | 3 | 266,688 | exact |
| feedforward.w2 | 1,838,656 | 262,592 | 1,572,864 | 3 | 266,688 | exact |
| feedforward.w3 | 1,795,136 | 217,472 | 1,572,864 | 3 | 221,568 | exact |
| qk_mm / qkv_mm | 2,752 | — | — | — | — | 无权重块（FP16/小张量） |
| 两个 layer_norm + q_norm + k_norm | 2,112 / 2,112 / 320 / 2,560 | — | — | — | — | FP16 内联 |

尺寸与参数量的关系全部吻合 `size = params / 2`（int4）：
q_proj 2,097,152 → 1,048,576；k/v 1,048,576 → 524,288；w1/w2/w3 3,145,728 → 1,572,864。

### 13.3 边界（离线不可解，需上机或编译器）

权重 payload 的**内部排列（swizzle）**仍然未知：在正确 payload 偏移上，行/列主序、
tile 16/32/64、group-64-major、n 路去交织、位平面拆分均不能与 HF 权重对上。
`libxnnl.so`（无 DWARF）反编译出的 `getReorderedBufferSize`/`setWeights` 只是 C++
容器脚手架，真正的索引公式不在其中——它属于 NPU 侧（YMCA 编译产物/固件）。
因此这一项**离线到此为止**，后续需要：上机差分（改 payload 观察引擎输出）或取得 YMCA 编译器。

### 13.4 全模型验证（28 层）

对 VLM 模型全部 28 层逐层解析：

```
weight blocks: 196   exact closures: 196   ->  ALL EXACT
```

即 `payload + size == 下一条记录起点` 在 **196/196** 个权重块上成立（每层 7 个：
q/k/v/out/w1/w2/w3，外加两个小张量 qk_mm/qkv_mm 与 4 个 norm 无权重块）。
记录布局模型可视为已验证。

## 14. payload 排列问题解决：**没有 swizzle，就是行主序 int4**（设备轮修正）

前几轮判"未解"的原因是**偏移错误**：payload 起点是 `权重头 + 4096 = 179,264`，
而当时按 175,232（= 权重头 + 64）解码，正好差 4,096 B = 8,192 nibble = **8 行**，
足以让相关性归零。修正后重测（此前在正确偏移上只测过"每行最大值"，而它恒为 8、无信息）：

| 检验 | 结果 |
|---|---|
| 行主序 + 二补码，仅用 per-channel scale | 元素级相关 **0.831** |
| 逐行拟合系数 `a_r / pc[r]` | 均值 **6.81**，**CV 0.177**（5–95%: 4.30–8.62） |
| **对照：错开 1 行** | **−0.01**（归零） |
| **对照：错开 1024 行** | **−0.02** |
| **对照：行序反转** | **−0.01** |
| 符号一致率 | 81.1% |
| 行内四等分段相关 | 0.764 / 0.740 / 0.745 / 0.748（均匀） |
| 每组自由 (scale, offset) 拟合上界 | R² = 0.739 |
| 每行 RMS(q) | 均值 3.45，CV 0.075 |

结论：**权重是行主序、二补码 int4，dequant ≈ `q × pc[r] × 6.81`**（`pc[r]` 为记录 +320 处
每通道 FP32 数组，自然通道序）。错位对照归零、正确对齐稳定，排除行/列/tile/交织任何排列。

残留 ~17% 不是布局：残差标准差约为量化步长的 **8 倍**（量化噪声应仅 `a/√12`），
说明官方 q_proj 与公开 HF Qwen3-0.6B 的 q_proj 是实质不同的张量——与"小米微调版"一致。

**转换含义**：既然布局是普通行主序 int4，就可以**直接把自己的 int4 权重写进 payload**
（配 per-channel FP32 scale + group=64 groupwise 数组），不需要任何 NPU 私有 swizzle。
剩余待定：groupwise 数组的编码（100–250，疑为指数码）与常数 6.81 的来源。

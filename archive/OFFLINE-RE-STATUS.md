# 离线逆向：总账（哪些做完了、哪些必须上机）

状态：`DONE` = 离线可复现且有证据；`PENDING` = 离线仍有手段但未做；`DEVICE` = 只能上机。

## A. 容器与可写性

| 项 | 状态 | 证据 |
|---|---|---|
| 0.5.0 头（魔数/u64 metaLen/4096 对齐/块表） | DONE | `tools/xnlm/header.js`；三个模型 `4096+Σ块=size` |
| 0.5.0 元数据（JSON XOR 16 字节魔数） | DONE | 解密/再加密 sha256 一致 |
| 层块逐张量记录（`[u64 len][name^magic][desc][payload]`） | DONE | `inspect.js --layer N` |
| 0.4.0 整文件 | DONE（可写） | `lora.js rebuild` → **sha256 与原文件一致** |
| 权重 payload 内部排列 | PENDING（见 E） | 多种布局+scale 判据均为 0 |

## B. 接口面（"自己调 NPU"的核心）

| 项 | 状态 | 产物 |
|---|---|---|
| `XNLM_*` 29 个原型 + 枚举/结构 | DONE | `generated/xnlm_api.h` |
| `xnnl*` 13 个算子原型 + 枚举/结构 | DONE | `generated/xnnl_api.h` |
| `mida*` 56 个原型 | DONE | `generated/mida_api.h` |
| `Momi_*` 15 个原型（从实现反编译） | DONE | `generated/momi_api.h` + `decomp/momi_api.c` |
| 插件 JNI 面 11 个方法 | DONE | — |
| libxnnl 导出 821 / 类 1215 / 约束 1817 | DONE | `generated/xnnl-*.txt` |
| `midaDeviceConfig` 等 config 结构 | **BLOCKED** | 实现在 `libxnumd.so`（厂商库，未随包发布）→ 上机拉取 |
| `vit.plan` | 已识别 | FlatBuffers / `YMCA` 6.2-3.2，无 schema |

## C. 模型包与安装

| 项 | 状态 |
|---|---|
| `.taiyi.model` ZIP 结构、`taiyi_hash.json` 自校验、资源目录布局 | DONE（见 NOTES §22–26） |
| 无 root 替换路径（TOCTOU）与 Design B（root） | DONE（`tools/designb/`、`chat/`） |

## D. 设备侧（离线不可替代）

- `/dev/hpc-*` 权限、`mida*` 实际所在库、`xnnlGetVersion` 版本检查
- 最小算子探针（RMSNorm/Quantize）跑通 → 逐值比对 CPU
- 用官方引擎当"标准答案"做差分（阶段 E）
- `midaDirectIO` 行为验证（对应 base.json 的 `directIO`）

## E. 权重 payload：**已解 —— 行主序 int4，无 swizzle**

payload = 权重头 + 4096；数据为行主序、二补码 int4；每通道 FP32 scale 在记录 +320（自然通道序）；
dequant ≈ `q × pc[r] × 6.81`。证据：逐行拟合系数均值 6.81 / CV 0.177，而**错开 1 行、错开 1024 行、
行序反转三种对照全部归零（−0.01）**；元素级相关 0.831、符号一致率 81%、行内四段均匀。

此前判"未解"纯粹是偏移错了 4,096 B（差 8 行）。残留 17% 来自官方权重与公开 HF 权重的实质差异
（残差约为量化步长的 8 倍），不是布局。

→ **转换可行性结论改变：可以直接写行主序 int4 权重，不需要 NPU 私有排列。**

## F. 结论

离线能做的"规格化"工作已基本完成：**容器可写、三层 API 有头文件、规格语料成表、插件契约明确**。
剩下两项（Momi 原型、mida config）可用反汇编继续；权重排列与设备行为必须上机。

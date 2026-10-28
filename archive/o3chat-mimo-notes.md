# O3 Chat — Mimo 路径研究笔记（归档）

原始日志，保留供参考。当前应用默认走 VLM。

## 失败现象

`Mimo` chip 选择 `ai_api = mimo_dev` 时，请求到达 `MimoVlmService` /
`BaseMimoVlmPlugin`，但 native prefill 失败：

```text
BaseMimoVlmPlugin: LLM prefill fail: status=-1
CALLBACK status=6 err=10803 len=0
```

`10803` = `ERROR_PROCESS_FAILED`。

## 根因（2026-10-27，无反证）

反编译两个插件（`plugin-mimo_vlm-*.taiyi`、`plugin-vlm-*.taiyi`）后确认：
**不是 API 不匹配，而是走了错误的入口。**

| 入口 | 是否应用 base LoRA | 是否需要客户端 `json_source` |
|---|---|---|
| `postTask`（当时用的） | **否** → `isApplyLoRA=false` → prefill `-1` | 否（插件自己构造 ChatRequest） |
| `executeChat` / `postChat` | **是**（`<modelDir>/lora_base_llm`） | **是**，为空则 `chat request is empty`（10803） |

`jsonSource` 不随 parcel 传递：`AIBody.distribute()` / `collect()` 只有在
`share_memory_enable = true` 时才把它放进 bundle。官方 `LLMKit` 总是开启
share memory 并挂 `ShareMemoryPool`；手写客户端设了 `false`，
所以插件始终看到空 `jsonSource`。无帧传输时不需要真正的内存池，
**只需要把标志打开**。

## 正确序列

1. 首次 Mimo 发送前发 `ai:method:llm_init`（20 s 看门狗、一次重试）；
2. 再发 `executeChat`，`share_memory_enable = true`，并带 `ChatRequest` 形状的
   `json_source`。

预期的设备日志：

```text
LLM init : .../mimo_model_f3/19/base.json
llm load succeed!              CALLBACK status=7 err=0
jsonSource: {"model":"mimo_dev_f3",...}
applyLoRA success, path=.../mimo_model_f3/19/lora_base_llm
LLM prefill ... / generate success
```

若出现 `applyLora failed, path=…, code=…`，说明模型目录不完整
（`lora_base_llm` 需含 `hybrid_swa_compressed_eagle.bin` + `trimMap_24k.bin`）。

## 实测（2026-10-27，无 root）

```text
MIMO_STATUS mimo_dev full=1.94 GB left=0 B state=0
LLM init time 337 ms                 CALLBACK status=7 err=0
applyLora total 217 ms   EAGLE engine initialized: draftLength=3 numBranches=4 maxTargetTokens=16
LLM Prefill: 1230.954 ms  (100 tokens, 79.98 t/s)
METRICS ttft=2178 tps=14.9 tok=10 total=2834     # 首次
METRICS ttft=197  tps=15.0 tok=10 total=839      # 热引擎，TTFT 降约 10 倍
```

VLM 不受影响：`ttft=799 tps=89.1 tok=33`。

## 实验开关

```bash
--es model mimo     # 选择 Mimo
--ez init true      # 发 ai:method:llm_init
--ez chat true      # 发 executeChat + json_source
--ez mimostatus true  # 查询安装状态后退出
```

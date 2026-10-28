# 玄戒 O3 / 超级小爱 端侧模型与 NPU 调用调查

日期：2026-10-25  
设备：`2608BPX34C`（`lhasa`），Android 17 / HyperOS `OS4.0.26.0.XPNCNXM`，SoC `xring_o3_asic`

## 1. 开发环境

已添加：

- `flake.nix`：`android-tools 37.0.0p1`、`apktool 3.0.3`、`jadx 1.5.6`、JDK 17、clang/cmake/ninja、scrcpy 等
- `.envrc`：`use flake`
- `flake.lock`

使用：

```bash
nix develop
# 或
direnv allow
```

已验证 `adb devices -l` 能看到设备。

## 2. 已拉取的 APK

| 包 | 版本 | 路径 | SHA-256 |
|---|---|---|---|
| `com.miui.voiceassist`（超级小爱） | 8.2.65.4714 | `artifacts/pulled/voiceassist-8.2.65.4714-base.apk` | `caafb6930e7a9b86c46ea9fc3ab00fbd76c826026fa0f23260f39737b42c8ffa` |
| `com.xiaomi.aiservice`（AI 服务/端侧运行时） | 4.2.12_8560455_260910_cn | `artifacts/pulled/aiservice-4.2.12-base.apk` | `6ce8f577c58da4f9dcc6093b5b3eaf7c527d7ddb3af19a2af066bb925f7236c4` |

## 3. native 库盘点

超级小爱 APK 里与推理有关的库：

- `libMNN.so`（2.6 MB）：MNN 运行时，含 KleidiAI/SME CPU kernel 和 NNAPI 字符串；没有 QNN/SNPE/HIAI 依赖
- `libCVLM-DLA.so`（59 KB）：目标检测（DLA Detector），不是 LLM
- 其余为语音、图像、React Native、Hermes 等库

结论：超级小爱 APK 本身不是 NPU LLM 推理后端；真正运行时在 `com.xiaomi.aiservice` 的私有数据/进程中。

## 4. 端侧模型位置

`com.xiaomi.aiservice` 的 `dumpsys diskstats` 显示 **4.63 GB App Data**，与两个端侧模型体积相符。

外部存储只看到空目录：

```text
/sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi/
  model/vlm_model_qwen3_0_7b_f3/15/        (空)
  plugin/mimo_dev_f3/25/                   (空)
  plugin/llm_runtime/13/                   (空)
  param/vlm_param/3/                       (空)
  param/mimo_param/4/                      (空)
```

实际 `.taiyi.model` 位于 `/data/data/com.xiaomi.aiservice/...`（或 `/data/user/0/...`），`adb shell` 无权限，当前也没有 `su`：

```text
ls: /data/data/com.xiaomi.aiservice: Permission denied
/system/bin/sh: su: inaccessible or not found
```

因此**不 root 无法导出这两个模型**。外部目录中的 `model_vlm_lora_text_extraction_f3` 等小文件只是 LoRA/占位文件。

## 5. NPU 访问链

### 5.1 厂商 UMD

设备上存在：

```text
/vendor/lib64/libNPURuntime.so
/vendor/lib64/libNBGlinker.so
```

`libNPURuntime.so` 导出 `npu::umd::Device`、`Memory`、`Network`、`Stream`、`Task` 等 C++ API；`libNBGlinker.so` 负责加载/链接 NBG 网络，并依赖 `libmimodelsecurity.so` 做模型解密/校验。

设备节点：

```text
/dev/hpc-cdev        npu_device
/dev/hpc-heap        npu_device
/dev/hpc-mdr         npu_service_device
/dev/hpc-rpmsg       npu_device
/dev/xvp_queue       npu_device
/dev/npu_freq_qos_*  npu_device
```

### 5.2 SELinux 限制

`vendor_sepolicy.cil` 中允许访问 `npu_device` 的域包括：

```text
platform_app
priv_app
system_server
hal_audio_default
hal_camera_default
hal_c2_x
vendor_xring_vpp_service
theme
minetd
hal_npu
```

**没有 `untrusted_app`**。普通第三方 APK 即使自带 `libNPURuntime.so`，也会在 `open("/dev/hpc-*")` 或 `ioctl` 时被 SELinux 拒绝；此外该库位于 `same_process_hal_file`，不是 app 可用的 public library。

### 5.3 厂商 Binder 服务

```text
service list:
  vendor.xring.intelligent.IIntelligentService/default
```

这是厂商自定义的 intelligent/NPU 服务，但 AIDL 库和服务二进制对 shell 不可读，当前无法直接还原接口。

## 6. AIService 的 LLM Binder 通道

`com.xiaomi.aiservice` 暴露：

```text
com.xiaomi.aiservice.llmmanager.LLMManagerService
  permission: com.xiaomi.aiservice.permission.AI_ABILITY
  protectionLevel: normal
```

因此普通 APK 可以声明并获取 `AI_ABILITY`，然后 bind。Binder 接口：

```text
com.xiaomi.taiyi.sdk.llm.ipc.ILLMManager
  authorize(AIRequest)
  sendTask(AIRequest)
  cancelTask(AIRequest)
  release(AIRequest)
```

请求是 `AIRequest`（Parcelable，包含 `AIHeaders`、`AIBody`、`IAICallback` binder）。`LLMKit` 已经实现了完整客户端流程（`postLLM`、`postChat`、`executeChat` 等）。

但是 `authorize()` 会校验：

- `headers.packageName`
- `headers.appSignature`（签名哈希）
- `headers.appCode`（AES 加密的 JSON，含 packageName + signatureHash）

空 appCode 或包名/签名不匹配会返回 `ERROR_AUTHORIZE_FAILED`。`persist.taiyi.auth.disable` 只在特定测试构建/条件下才可能跳过。普通第三方应用没有 Xiaomi 签发的 appCode，无法通过授权。

即使授权通过，模型也是在 `com.xiaomi.aiservice:llm` 这个 **priv_app** 进程里运行，调用方只能使用 LLM 任务 API，拿不到原始 NPU 张量/网络接口。

## 7. 可行性结论

| 路径 | 普通用户态 APK | 说明 |
|---|---|---|
| 直接 `dlopen` `libNPURuntime.so` + `/dev/hpc-*` | 不可行 | SELinux 拒绝 `untrusted_app`；库不在 public.libraries |
| 绑定 `vendor.xring.intelligent` 服务 | 未验证/大概率不可行 | 接口不可读，服务很可能有 SELinux/权限限制 |
| 绑定 `LLMManagerService` 跑本地 LLM | 条件可行 | 需要 Xiaomi 签发的 appCode；只能走 LLM API，不是裸 NPU |
| 系统/平台签名 APK | 可行 | `platform_app`/`priv_app` 有 `npu_device` 权限 |
| root + permissive/自定义 domain | 可行 | 可直接用 UMD 或导出模型，但会破坏系统完整性 |
| Android NNAPI / LiteRT | 未发现厂商对普通 app 开放 | 当前 vendor 走自有 `libNPURuntime`，不是标准 NNAPI 路径 |

**结论：在未 root、未取得 Xiaomi 平台签名/appCode 的情况下，普通用户态 APK 不能直接调用该 NPU。**  
最现实的“合法”路线是向 Xiaomi 申请端侧 AI SDK/appCode，通过 `LLMManagerService` 使用本地 LLM；或者把 APK 做成 priv-app/平台签名。直接提取 `libNPURuntime.so` 做通用 NPU 调用，在 SELinux enforcing 下走不通。

## 8. 下一步可选

1. **提取模型**：需要临时 root（设备 Download 里有 `ReSukiSU_v4.2.0-rc3` APK，但当前未启用 `su`）。root 后可 pull `/data/data/com.xiaomi.aiservice` 下的模型并分析 `.taiyi.model`/NBG 格式。
2. **静态逆向 NPU 库**：继续用 REA/Ghidra 分析 `libNPURuntime.so`、`libNBGlinker.so`，还原 NBG 网络格式和 ioctl 协议。
3. **做 Binder PoC**：构建一个最小 APK，声明 `AI_ABILITY` 并 bind `LLMManagerService`，验证未授权时确实返回 `ERROR_AUTHORIZE_FAILED`（不绕过授权）。
4. **走官方/系统签名路线**：如果目标是产品化，优先申请 Xiaomi 端侧 AI 能力或使用标准 Android NNAPI/LiteRT。

## 9. 证据文件

```text
artifacts/pulled/voiceassist-8.2.65.4714-base.apk
artifacts/pulled/aiservice-4.2.12-base.apk
artifacts/system/libNPURuntime.so
artifacts/system/libNBGlinker.so
artifacts/system/libnpumanager.so
artifacts/libs/arm64/libMNN.so
artifacts/jadx/aiservice/            # AIService JADX 源码
artifacts/apktool/aiservice/         # AIService manifest/resources
artifacts/diskstats.txt              # 各包存储占用
```

## 10. `authorize` 接口审计（静态发现，未执行未授权调用）

接口已还原到方法级：

```text
com.xiaomi.taiyi.sdk.llm.ipc.ILLMManager
  transaction 1: authorize(AIRequest) -> int
  transaction 2: sendTask(AIRequest)
  transaction 3: cancelTask(AIRequest)
  transaction 4: release(AIRequest)
```

服务端实现位于 `q2/a.java`，授权逻辑位于 `u1/a.java`，客户端调用位于 `l4/d.java`。

### 10.1 缺陷 A：授权状态没有在服务端强制执行

`LLMKit` 客户端在 `onConnected()` 里先调用一次 `authorize(createServiceRequest(...))`，成功后才对队列中的任务调用 `sendTask()`（`l4/d.java:275`、`:296`）。

但服务端 `q2/a.java` 的 `sendTask()`（`:86`）没有再次调用 `u1.a.a(request)`，也没有检查该 Binder 调用者是否已经授权、是否在已授权客户端表中。`authorize()` 本身也没有把“已授权”状态写入任何服务端会话；它只做一次判断并返回错误码。

结果：授权只是**客户端 SDK 的约定**，不是服务端的访问控制。自定义 Binder 客户端可以跳过 transaction 1，直接发 transaction 2。

### 10.2 缺陷 B：`authorize()` 信任调用方传入的 `AIHeaders.uid`

`u1/a.java:31-32`：

```java
Log.d("AuthManager", "Process.SYSTEM_UID: 1000");
if (uid != 1000) {
    // 才进入 appCode / packageName / signature 校验
}
return true; // uid == 1000 直接成功
```

这里的 `uid` 来自 `request.getHeaders().getUid()`，即调用方自己构造的 `AIHeaders`，不是 `Binder.getCallingUid()`。`AIHeaders` 还提供 public `setUid(int)`。因此，调用方可以把该字段设成 1000，使 `authorize()` 直接返回成功；服务端没有把 header UID 与真实 Binder UID 做一致性校验。

### 10.3 影响

- `LLMManagerService` 只受 `com.xiaomi.aiservice.permission.AI_ABILITY` 保护，而该权限的 `protectionLevel` 是 `normal`，普通第三方 APK 可以声明并获取。
- 结合缺陷 A，第三方 APK 可以在没有 Xiaomi 签发 `appCode` 的情况下，直接向 `sendTask` 提交本地 LLM 任务。
- 结合缺陷 B，即使只调用 `authorize`，也可以伪造 `uid=1000` 通过校验。
- 该绕过拿到的是 `LLMManagerService` 的 LLM 任务 API，不是裸 NPU；模型仍在 `com.xiaomi.aiservice:llm` 私有进程中执行，调用方不能直接提交任意 NBG 网络或读取 NPU 张量。

### 10.4 修复建议

1. 在 `sendTask`/`cancelTask`/`release` 每个 transaction 内使用 `Binder.getCallingUid()` 和 `Binder.getCallingPid()` 重新校验，不信任 `AIHeaders`。
2. 不要把 `uid == 1000` 当作授权短路；系统 UID 也应校验包名/签名或直接拒绝外部 Binder 调用。
3. 服务端维护“已授权 client”状态，并把它绑定到 Binder 调用者身份（UID/PID/包名/签名），而不是绑定到请求头。
4. 将 `AI_ABILITY` 改为 `signature` 或 `privileged` 级别，或增加 `android:permission` 之外的签名校验。

> 以上为静态审计结论，未在设备上执行未授权的 `sendTask` 调用。该问题属于本地授权绕过，建议先向 Xiaomi 安全响应中心提交，再决定是否做受控 PoC。

## 11. 最小复现 APK

已构建 `poc/taiyi-authorize-poc.apk`（17 KB，debug 签名）。

- 只声明 `com.xiaomi.aiservice.permission.AI_ABILITY`（normal）
- 绑定 `com.xiaomi.aiservice.LLMManagerService`
- 先直接 `sendTask`，不调用 `authorize`
- 再分别用真实 UID + 空 app_code、伪造 `headers.uid=1000` + 空 app_code 调用 `authorize`
- 通过 `llm_client` Binder 接收 `onTaskChange`，打印 status/code/响应 body
- prompt 仅为 `Reply with exactly POC_OK and nothing else.`

构建：

```bash
nix develop
./poc/build.sh
```

运行：

```bash
nix develop
./poc/run.sh
```

当前状态：APK 已构建成功，但在 `adb install` 前设备 USB 掉线（`adb devices` 为空），尚未取得实机日志。设备重新连接后需要安装并采集：

```text
AUTHORIZE_CONTROL_REAL_UID_EMPTY_CODE   # 预期非 0
AUTHORIZE_FORGED_UID_1000               # 若为 0，证明缺陷 B
SENDTASK_DIRECT_TRANSACT                # 若 true 且收到回调，证明缺陷 A
CALLBACK_RECEIVED
CALLBACK id=... status=... err=...
RESP_BODY json_result=...
```

注意：若只看到 `SENDTASK_DIRECT_TRANSACT=true` 而没有回调，只能证明 Binder 入口未拦截，不能证明模型实际执行；需要拿到 `CALLBACK_RECEIVED=true` 和响应体作为最强证据。

## 12. 实机复现结果（2026-10-25 13:42）

PoC 已安装并自动运行，设备构建：

```text
Xiaomi/lhasa/lhasa:17/CP2A.260605.016/OS4.0.26.0.XPNCNXM:user/release-keys
```

关键结果：

```text
SERVICE_CONNECTED com.xiaomi.aiservice/.llmmanager.LLMManagerService
STEP 1 sendTask without authorize
SENDTASK_DIRECT_TRANSACT=true
CALLBACK_RECEIVED=true
AUTHORIZE_CONTROL_REAL_UID_EMPTY_CODE=10109   # ERROR_AUTHORIZE_FAILED
AUTHORIZE_FORGED_UID_1000=0
```

`sendTask` 之后，`com.xiaomi.aiservice:vlm`（pid 12706）实际加载并运行了
`vlm_model_qwen3_0_7b_f3`：

```text
Taiyi-Plugin-BaseVlmPlugin: [vlm_qwen3_0_7b_f3:18] applyLoRA success
Taiyi-Plugin-BaseVlmPlugin: [vlm_qwen3_0_7b_f3:18] VLM generate start
Taiyi-Plugin-BaseVlmPlugin: [vlm_qwen3_0_7b_f3:18] generate success: token=33
TaiyiPoc: CALLBACK id=poc-... status=7 err=0
TaiyiPoc: RESP_BODY error_code=0
TaiyiPoc: RESP_BODY ans=I am an AI language model ...
```

说明：

- 这是一次**未调用 `authorize`、未提供 `app_code`** 的普通第三方 APK 调用；
- VLM 实际生成 33 个 token，返回 `status=7`、`error_code=0`；
- 由于 PoC 只发了 1×1 白色 bitmap，插件日志为 `imageCount=0`，模型回答内容本身无意义；这证明的是“模型被执行”，不是图像理解质量；
- `authorize` 对照：真实 UID + 空 app_code 返回 `10109`（授权失败）；伪造 `headers.uid=1000` 返回 `0`。

完整证据日志：`artifacts/poc-evidence.log`  
复现说明：`poc/EVIDENCE.md`  
PoC APK SHA-256：`62562e998ba9db7beca4dabe574ca8921ecbeef948c43f16f5c352017bc25c97`

## 13. 能否运行官方下载以外的 LLM

结论：**通过当前 AIService / LLMManagerService 通道，基本不能运行任意第三方 LLM。**

证据链：

1. `LLMManagerService.sendTask(AIRequest)` 只从 `AIHeaders.ai_api` 取一个 API 名称，
   然后 `configuration.e.i(api, ...)` 按名称查 `AIConfig`；请求里没有模型路径字段。
2. `AIConfig` 的 `models` 是 `AIModel` 列表（`BASE_LLM`/`LORA`/`VIT`/`LIBS`），
   实际文件由 `AIResource.path` 给出；`AIResource` 只有 `id/version/path/type`，
   路径来自 Xiaomi 下发的资源记录，不是调用方传入。
3. `AIPluginManager` 从 `AIDetail.getResources()` 取插件 APK、lib 目录和模型路径；
   `PluginProperty` 的 `apkPath/libDir/entry` 也来自官方资源记录。
4. 实际模型目录是：
   `/data/user_de/0/com.xiaomi.aiservice/taiyi/model/vlm_model_qwen3_0_7b_f3/15/lora_base_llm`
   不是 GGUF/ONNX 这类通用格式；`libNBGlinker.so` 中能看到
   `decryptModelInMemoryV2`、`libmimodelsecurity.so` 等模型解密/校验链路。
5. 当前配置里 `hasLLMApi=false`、`hasVLMApi=true`；能跑的是官方
   `vlm_qwen3_0_7b_f3`，不是任意模型。

因此：

| 路线 | 可行性 | 说明 |
|---|---|---|
| AIService 跑官方已配置模型 | 可以 | 受 `api`/下载配置/插件限制 |
| AIService 跑第三方 GGUF/ONNX | 基本不可行 | 没有模型路径入口，格式/插件/解密不匹配 |
| root 后替换模型文件/改配置 | 理论上有条件 | 需要同格式、同插件、可能还要过解密/校验；不确定性高 |
| 独立 App + llama.cpp/MNN-LLM | 可行 | 可跑 GGUF/MNN，但走 CPU/GPU，不是 NPU；7B 在手机上要评估内存/热 |
| 独立 App 直接调厂商 NPU | 不可行 | 需要平台/priv-app 权限、NBG 工具链和模型加密链路 |

对聊天 App 的影响：如果只做“官方模型客户端”，可以继续；如果要支持用户自选
GGUF/Qwen/Llama，应该把 transport 抽象成两个 backend：

- `AiserviceBackend`：官方模型，当前可跑 VLM 0.7B
- `LlamaCppBackend`：本地 GGUF，CPU 推理，不依赖 AIService

不要把“任意模型”宣传成 AIService 的能力；它做不到。

## 14. 设置页“下载本地 LLM”按钮的接口

按钮链路：

```text
ModelLocalActivity
  → ModelDetailActivity / ModelDetailContentActivity (trampoline)
  → v2.h.b(context, scene, apiSet, pendingIntent, uiFlag)
  → com.xiaomi.taiyi.sdk.download.a (AIDownload)
  → bind DownloadService
  → IDownloadManager.start(AIRequest)
  → DownloadEngine (com.xiaomi.aiservice.download.m)
```

关键接口：

```text
IDownloadManager
  transaction 1: register(AIRequest, IDownloadListener)
  transaction 2: unregister(AIRequest)
  transaction 3: start(AIRequest)
  transaction 4: cancel(AIRequest, scene)
  transaction 5: pause(AIRequest, scene)
  transaction 6: query(AIRequest) -> AIResponse
```

下载请求不是 URL，而是 `AIRequest` 头里的字段：

```text
ai_scene          scene
ai_api_list       API 名称列表，例如 llm / vlm / mimo_dev / llm_runtime
ai_resource_list  资源 ID 列表（只能命中本地下载数据库里已有的资源）
is_llm
download_ui_flag
pending_intent
...
```

`DownloadService.onBind` 会读取绑定 Intent 里的 `AIRequest`，并调用
`u1.a.a(aIRequest)`；这和前面的 `authorize` 是同一套逻辑，因此
`headers.uid=1000` 也可以绕过。之后 `IDownloadManager.start` 不再做授权检查。

按钮实际用的 scene 是 `local_xiao_ai`，对应 API 集合：

```text
llm_runtime, mimo_dev, framework, qcom_qnn,
mimo_lora_offline_xiaoai, sm_auditor
```

另一个 scene `text_llm` 的 API 集合包含：

```text
creation, llm, vlm, mimo_dev, llm_runtime,
vlm_lora_qwen_caption, mimo_lora_offline_xiaoai, sm_auditor,
vlm_lora_qwen_text_extraction, vlm_lora_image_extraction,
vlm_lora_gallery_caption, vlm_lora_entity_relation_graph
```

因此：

- **可以**用自建 `AIRequest` 触发官方配置中存在的其他模型/资源下载；
- 例如把 `ai_scene` 设为 `text_llm`，`ai_api_list` 设为 `["llm"]`，如果本地配置里有
  `llm` 的 `AIConfig`，`DownloadEngine` 会下载对应资源；
- **不能**传入任意 URL 或本地 GGUF 路径；`DownloadEngine.i()` 只通过
  `configuration.e.s(apiList)` / 资源数据库解析官方资源；
- 如果 `llm` 配置不存在或资源 ID 不在本地数据库，返回“no need download”/错误。

注意：`DownloadService` 的权限是 `AI_ABILITY`（normal），第三方 APK 可以 bind；
`onBind` 的授权绕过意味着可以在用户不点按钮的情况下触发官方模型下载（消耗流量/存储）。
这是另一个独立问题，建议一并报告，但不要在没有确认大小的情况下盲目 `start`。

## 15. 下载接口实测（query-only）

用 `DownloadProvider.call("query")` 发了一个 `uid=1000` 的 `AIRequest`（未下载任何东西），
成功返回了各官方 API 的资源大小。`DownloadProvider` 的 `u1.a.a()` 同样被
`headers.uid=1000` 短路，说明下载侧也有同一授权问题。

实测结果：

| scene / API | full_size |
|---|---:|
| `local_xiao_ai`（按钮默认） | 2,250,644,076 B ≈ 2.25 GB |
| `text_llm`（完整集合） | 3,443,281,564 B ≈ 3.44 GB |
| `text_llm` + `llm` | **0** |
| `text_llm` + `vlm` | 872,148,536 B ≈ 0.87 GB |
| `text_llm` + `mimo_dev` | 2,079,643,818 B ≈ 2.08 GB |
| `text_llm` + `mimo_vlm` | 0 |
| `text_llm` + `llm_runtime` | 57,866 B（插件，不是模型） |
| `text_llm` + `creation` | 0 |
| `text_llm` + `cloud_api` | 0 |
| `text_llm` + `sm_auditor` | 44,746,478 B |
| `text_llm` + `framework` | 14,472,084 B |
| `text_llm` + `qcom_qnn` | 0 |

结论：

- `llm` API 没有资源（`full_size=0`），设备上没有可下载的独立文本 LLM；
- 现有官方模型只有 `vlm`（0.87 GB）和 `mimo_dev`（2.08 GB）；
- 设置页按钮的 `local_xiao_ai` 是 Mimo/runtime 组合，不是另一个 LLM；
- `text_llm` 的 3.44 GB 是 VLM + Mimo + LoRA/审计等资源的合集，不是新模型；
- 因此“通过下载接口换一个官方 LLM”不可行；
- 但第三方 APK 可以伪造 `headers.uid=1000` 调用 query/start，触发官方资源下载，
  这是独立的未授权下载/流量问题。

证据：`artifacts/download-probe.log`  
探测代码：`poc/src/com/example/taiyi_authorize_poc/DownloadProbeActivity.java`

## 16. HTTPS / MITM 评估

静态检查结果：

- `AndroidManifest.xml` 没有 `android:networkSecurityConfig`，也没有
  `android:usesCleartextTraffic="true"`；targetSdk 36，默认不允许 cleartext。
- 配置接口使用 `OkHttpClient.Builder()` 默认配置：
  - 没有 `TrustManager` / `X509TrustManager`
  - 没有 `HostnameVerifier`
  - 没有 `CertificatePinner` / 证书 pin
  - 使用系统 TrustStore + OkHostnameVerifier
- 配置端点是 `https://ai.engine.miui.com/api/backend/deploy/list`。
- 另一个 `HttpURLConnection` 工具类（`n3/a`）也没有 trust-all / 自定义 HostnameVerifier。
- 文件下载交给 Android `DownloadManager`（`content://downloads/my_downloads`），
  由系统进程处理 TLS；下载代码允许 `http`/`https`/`ftp`/`ed2k`/`magnet` scheme。
- 配置响应没有额外签名校验；请求侧只有 `sign` HMAC，响应信任完全依赖 TLS。
- 下载侧存在 `fileSize` / `fileHash` 校验逻辑（`ERROR_DOWNLOAD_HASH_NOT_MATCH`），
  但 `fileHash` 和 `fileUrl` 都来自同一份配置响应；如果配置被 MITM，二者可一起改。
- 插件 classloader 从配置资源路径加载，未看到明显的 APK 签名校验。

结论：

- **有标准 HTTPS 校验**：系统 CA + hostname verification，没有 trust-all。
- **没有证书 pinning**：不能抵抗系统级 CA 被替换/设备被 root 后安装恶意系统 CA。
- **普通用户安装的 CA 默认不被 app 信任**（targetSdk 36，无 NSC）；
  但 `DownloadManager` 是系统组件，其 user-CA 行为需要实机验证。
- **真正的 MITM 风险**：
  1. 设备已 root / 攻击者能安装系统 CA → 可解 TLS；
  2. 配置里如果存在 `http://` 资源 URL → 可直接替换文件；
  3. 配置响应无签名 → TLS 一旦被破，`fileUrl` 和 `fileHash` 可一起伪造；
  4. 插件 APK 加载链没有明显签名校验 → 最坏情况是 AIService 进程内代码执行。
- 非 root、无可信系统 CA 的情况下，远程 MITM 门槛较高；但这套设计没有 pinning，
  不属于“高保证”更新链路。

## 17. 无 root 离线测试下载链接

结论：**可以。** 不需要 root，也不需要 MITM；逆向配置请求的签名后，可以直接从主机访问
`https://ai.engine.miui.com/api/backend/deploy/list`，拿到完整 `configList` / `resourceList`
和模型 CDN URL。

签名算法（`e3/a.m`）：

1. 参数按 key 排序，拼成 `key=value&...`
2. 末尾拼接 `R.string.secret_key`
3. UTF-8 字节做 Android `Base64.NO_WRAP`
4. 对 Base64 ASCII 做 MD5
5. 输出 32 位大写十六进制

请求同时带：

- Query：`t1.a.a()` 设备条件 + `timestamp` + `sign`
- Header：model/product/protocol/region/hyper_os/android_os/ota_version/package_name/version_code/version_name/caller_* + timestamp + sign
- Body：`{"conditions": {...}}`

实测结果（2026-10-25）：

- `stage=1`（正式）：43 个 config，92 个 resource，**92/92 全部 HTTPS**
- `stage=0`（preview）：50 个 config，额外出现 `mimo_vlm/mimo_vlm_cloud_f3`
- 正式 LLM：`vlm`（`vlm_qwen3_0_7b_f3`）、`mimo_dev`（`mimo_dev_f3`）
- preview 额外 LLM：`mimo_vlm`（`mimo_vlm_cloud_f3`，2,016,256,677 B ≈ 2.0 GB）
- 小资源 `patch@3` 下载后 SHA-256 与配置一致：
  `46d0c246a759e86d051ff27fee8f2f47a2961a6f22731afb3d76ab9db62966dc`
- CDN `ai-model-cdn.engine.miui.com` 返回 `200`、`Accept-Ranges: bytes`，无需 cookie/auth
- 证书：DigiCert GeoTrust G2，`*.engine.miui.com`，TLS 1.2/1.3

意义：

- 可以离线（host-side）枚举官方模型/插件、检查 `fileUrl`/`fileHash`、用 HEAD/Range 验证 CDN；
- 下载链没有设备绑定/鉴权，静态 `secret_key` 在 APK 资源里，配置接口本质上是公开可查询的；
- 但这只能“拿到模型文件”，不能让正式版 AIService 加载 preview 的 `mimo_vlm`；
  要让服务使用它，仍需要配置注入/root，或服务端把 stage 切到 preview。

## 18. 模型回库/安装链的注入风险

结论：**有注入风险，主要是 ZipSlip + 插件无签名校验；正常用户没有入口，但配置/CDN/root 可控时可以运行自备模型/插件。**

链路：

```text
DownloadManager 下载 .taiyi.model
  → x3/e onGroupStateChange 校验 fileSize/fileHash
  → t2.c.a → v2.g.i
  → m3.a.o 解压 ZIP 到 taiyi/<type>/<id>/<version>
  → 读取 ZIP 内 taiyi_hash.json 校验解压文件
  → 注册到 AIResources / AIConfig
  → AIPluginManager 用 BaseDexClassLoader 加载插件 APK
```

缺陷：

1. **ZipSlip**：`m3.a.o` 使用
   `new File(dest, zipEntry.getName())`，没有 canonical path 检查；
   恶意 ZIP entry 如 `../../../../shared_prefs/...` 可写出目标目录。
2. **插件无签名校验**：`c.c` 直接
   `BaseDexClassLoader(property.getApkPath(), ...)`；
   插件包内没有 `PackageManager` 签名校验/证书 pin。
3. **完整性校验的信任根在配置/ZIP 内**：
   `fileHash` 来自配置；`taiyi_hash.json` 来自 ZIP 本身；
   配置被篡改或 ZIP 被替换时，攻击者可以同时提供匹配的 hash。
4. **路径字段未清洗**：`fileName` 来自配置，拼进下载路径；
   `id`/`version` 也来自配置，未看到 `..`/分隔符校验。

威胁模型：

| 攻击者能力 | 能否注入自备模型/插件 |
|---|---|
| 普通用户，无 root/代理 | 不能：只能触发官方配置资源 |
| 只控制 CDN 文件 | 不能：下载 hash 不匹配 |
| 控制配置响应（root/系统 CA/MITM/服务端） | 能：改 `fileUrl` + `fileHash`，配合 ZipSlip/插件加载 |
| root | 能：直接替换 app-private 文件，或无签名加载插件 |

因此“用户自己提供的模型”在正常 UI 下没有入口；但如果用户能改配置或拿到 root，
当前链路没有足够的密码学/路径隔离来阻止其模型或插件被执行。

修复建议：

- 解压前 canonicalize，拒绝绝对路径、`..`、symlink，限制 entry 数量/大小；
- 插件 APK 校验小米签名/证书 pin；
- 配置响应签名，`fileHash` 不与应用配置同源；
- `taiyi_hash.json` 做签名，而不是仅作为 ZIP 内清单；
- 清洗 `fileName`/`id`/`version`，禁止路径分隔符；
- 下载仅允许 HTTPS + host allowlist/pinning。

## 19. 官方模型包解剖（host 侧 CDN Range，2026-10-26，未碰设备）

方法：`vlm_model_qwen3_0_7b_f3-1.taiyi.model`（702,394,069 B）支持 HTTP Range，
只拉 ZIP 尾部中央目录列出全部 21 个条目；小文件按 local header 偏移 + Range 取出
deflate 解压。插件包 `plugin-vlm-*.taiyi`、`plugin-llm_runtime-*.taiyi`、
`plugin-framework-*.taiyi` 和 `vlm_param.json` 直接整包下载。全部归档在
`artifacts/cdn/`。

### 19.1 模型包结构（ZIP，21 条目）

```text
libmomi.so                      1.46 MB   LLM 引擎（Momi_LLM_* C API）
libMI_Genai_Runtime.so          2.75 MB   JNI 壳（Java_com_xiaomi_aicr_model_VLMEngine_*）
libxnlm.so                      7.54 MB   XNLM 权重格式运行层（dlopen libxnumd.so）
libxnnl.so                      9.28 MB   NPU graph 层（源码路径泄露 vendor/xring/ai/xnnl/bellv100）
libtokenizers.so                5.90 MB   HuggingFace tokenizers（Rust）
llm/llava_qwen3.bin             413.8 MB  权重，magic "XNLM0.5.0:<githash>"，私有格式
embedding/embedding_fp16.bin    311.2 MB  fp16 embedding 表
tokenizer/tokenizer.json         16.6 MB  标准 HF tokenizer.json（ChatML 特殊 token）
lora_base_vlm/vit/vit.plan       95.0 MB  编译好的 ViT 网络，magic "YMCA"（NPU plan）
base.json / lora_base_*/task.json         引擎配置（见下）
.DS_Store                                 （macOS 打包残留）
```

### 19.2 base.json 关键内容

```json
"backend": { "type": "o3", "performance_config": { "cu_num": 4,
             "ddr_perf_level": 6, "npu_perf_level": 5 }, ... }
"kvcache":  { "type": "o3_kvcache" }
"llm_info": { max_context 4096, vocab 151936, head_dim 128, embedding_dim 1024 }
"tokenizer": HF tokenizer.json + ChatML prefix/suffix；sampler=argmax；rope_theta=1e6
```

`lora_base_vlm/task.json` 的 `general_model` 指向 `vit/vit.plan`，backend 同为 `o3`。
**说明 LLM 主体（不只 ViT）都跑在 Xring O3 NPU 上**，`cu_num/ddr/npu_perf_level`
是 NPU/DDR 调频参数，`cpu_mask 0x0f0` 只是辅助 CPU 线程。

### 19.3 完整执行链

```text
BaseVlmPlugin (插件 APK, Java)
  → libMI_Genai_Runtime.so (JNI, dlopen libmomi.so, 符号 Momi_LLM_create/prefill/generate)
    → libmomi.so（工程名 xring_llm_engine/lexi/momi，还有 F3Executor/NNExecutor/BitaEngine）
      → libxnlm.so（读 XNLM 权重；midaDevice/Graph/DirectIO 符号来自 libxnumd.so）
      → libxnnl.so（xnnlUmdWrapper：Conv/Matmul/TMA/L2B 节点，构造 NPU graph）
        → libxnumd.so（UMD 封装，未在包内，应在 vendor 分区）
          → libNPURuntime.so → /dev/hpc-*
```

### 19.4 对“任意第三方模型”的最终判定

- 请求侧复核：`AIHeaders`/`AIBody` 没有模型路径/ID 字段；`sendTask` 只取 `ai_api`；
  `DownloadProvider` 的 insert/update/delete 全是空操作。NOTES §13 结论成立。
- 即使 root / 配置注入：权重必须是 `XNLM0.5.0` 私有格式（无公开转换器），
  多模态还需 `YMCA` 魔法的 NPU 编译 plan（Xiaomi Franklin 工具链产物），
  `base.json` 架构参数写死。**“任意第三方模型（GGUF/ONNX/HF）”在这套栈上
  连理论路径都没有，只有小米工具链转换出的同格式模型才可能被加载。**
- `libmomi.so` 还引用 `libxnn_runtime.so`/`libxnlm_plus.so`（其他后端/SoC 家族），
  说明引擎族有 CPU 变体，但同样不对外分发。
- tokenizer 是标准 HF JSON——整条链里唯一通用格式的部件。


## 20. 假设有 Momi/XNLM 转换器：注入自转换模型的路径设计（2026-10-26，纯静态）

前提：转换器只解决"文件合法"。宿主问题独立存在——`libmomi→libxnumd→
libNPURuntime→/dev/hpc-*`，SELinux 只允许 platform_app/priv_app 等域访问
`npu_device`，所以合法文件在自己 App（untrusted_app）进程里照样跑不了，
**必须注入 AIService，或者自己做平台/priv 签名 App 当宿主**。

### 20.1 持久化与校验事实（决定注入点）

```text
Room DB（aiservice 私有 databases/，WAL）三张表：
  ai_config       (api,name) PK  jsonSource = AIConfig JSON（api→plugin+models+param）
  local_resource  (id,version) PK  jsonSource = 资源 JSON，含绝对 path/size、apkPath/libDir
  remote_resource (id,version) PK  服务端资源元数据（fileUrl/fileHash/encrypted）

加载链无哈希/签名校验：
  fileSize/fileHash 只在下载完成时校验（x3/e）；taiyi_hash.json 只在解压时校验
  （m3.a.o）；运行时 BaseVlmPlugin→LLM_Init 按 local_resource 的 path 直接打开文件。
  插件 dex 与 libmomi/libxnlm 均无 sha/verify/signature 字符串。
  vlml 模型 CDN ZIP 为明文（remote_resource.encrypted 列存在但该模型未启用加密链）。
插件目录发现式定位：apkPath=目录里第一个 *.apk，libDir=<目录>/lib/<abi>（y1.d）。
路径拼接硬编码：模型目录 + "/lora_base_llm"、"/lora_base_vlm"（插件 dex 字符串）。
```

### 20.2 三种注入设计（均需 root）

| 设计 | 改动面 | 说明 |
|---|---|---|
| A 原地换文件 | 只动文件系统 | 覆盖 `taiyi/model/vlm_model_qwen3_0_7b_f3/15/` 下同名文件，DB 零改动，最稳 |
| B 注册新资源 | 文件系统 + Room DB | 新 id 新 api（如 myllm），官方模型原样保留，可并存 |
| C 走官方下载链 | system CA + config MITM | 复用下载/解压/注册全链路，验证 §18 威胁模型 |

**A（最小手术）**：`am force-stop com.xiaomi.aiservice` → 用转换器产物按原布局
覆盖同名文件（llm/llava_qwen3.bin、embedding/、tokenizer/、base.json、
lora_base_*/task.json、vit.plan；.so 不动或一并替换）→ `chown u0_a106:u0_a106`、
`restorecon`（/data/user_de 为 DE 存储）→ 重启后 sendTask(ai_api=vlm)。
注意 base.json 的架构参数与权重必须自洽（连 base.json 一起换即可）。

**B（干净注册）**：
1. 解压包放 `taiyi/model/<my_model>/<v>/`；
2. `local_resource` INSERT：type=MODEL、scheme=SANDBOX、unwrapped=1、
   subType=BASE_LLM、jsonSource 含绝对 path 与 size；
3. `ai_config` INSERT：克隆 vlm 行 jsonSource，api 改 "myllm"，
   models[0].id 改 <my_model>，plugin 仍指向已安装 vlm 插件（BaseVlmPlugin 复用）；
4. force-stop 重启（w1.a 内存缓存需重载），sendTask(ai_api="myllm")。
   Room WAL 先 checkpoint；jsonSource 建议先 pull 真实行做模板。

**C（官方链路）**：root 装 system CA + hosts 把 ai.engine.miui.com 指向本地代理，
改 `deploy/list` 响应插入 configList/resourceList 条目（fileUrl 指向自有服务器；
下载代码允许 http 等明文 scheme，fileHash 自带即可匹配）。配置无 pinning（§16），
触发一次配置刷新后 DownloadEngine 自动完成下载→校验→解压→注册。

### 20.3 无 root 仍不可行 + 平台签名反向方案

- 无 root 注入无入口：config 只信系统 CA（targetSdk 36 无 NSC、无 pinning 但
  user CA 不被信任）、资源 URL 只来自服务端配置、DownloadProvider 只读、
  安装目录私有。§18 威胁表不变。
- **反向方案**：平台/priv 签名 App 自带宿主。整套 native 栈随模型包分发
  （libMI_Genai_Runtime/libmomi/libxnlm/libxnnl/libtokenizers 可自由提取），
  platform_app 在 sepolicy 中本就允许 npu_device。缺 libxnumd.so（vendor 分区，
  加载可行性需设备实验验证 same_process_hal_file/public.libraries 约束）。

### 20.4 设备实验清单（注入相关）

1. root 后 pull `databases/` 确认 ai_config/local_resource 真实行格式与 DB 文件名；
2. `ls /data/user_de/0/com.xiaomi.aiservice/taiyi/model/vlm_model_qwen3_0_7b_f3/15/`
   核对 21 个条目的实际落盘布局（含 .so 与 .DS_Store）；
3. platform 签名路线：验证 vendor 库（libxnumd/libNPURuntime）对 platform_app
   的可见性（public.libraries.txt / same_process_hal_file）；
4. 设计 A 前先做阴性对照：仅 touch 目录观察是否有完整性巡检（预期没有）。

## 21. 平台签名路线核实 + 离线进度（2026-10-26，未碰设备）

### 21.1 签名事实（从已拉取 APK 的 v1 签名提取，keytool）

```text
voiceassist  META-INF/PLATFORM.RSA  CN=MIUI, Xiaomi（自签名）
             SHA256: C9:00:9D:01:EB:F9:F5:D0:30:2B:C7:1B:2F:E9:AA:9A:47:A4:32:BB:A1:73:08:A3:11:1B:75:D7:B2:14:90:25   ← 真平台密钥
aiservice    META-INF/RELEASEK.RSA  CN=MIUI, Xiaomi（自签名）
             SHA256: D4:5F:07:6F:...:72:CC:6F  ← 与 PoC 日志 ai_app_signature 完全一致
```

结论修正：**aiservice 不是平台签名**。它的 NPU 权限来自 priv-app 安装位置
（priv_app 域），平台签名的是小爱。因此"自造宿主"有两条等价门槛：
平台私钥（正常 adb install 即可获得 platform_app 域）或 priv-app 分区安装
（需要 root）。公开检索未发现该 MIUI 平台证书指纹在已知泄露集合中；
AOSP default-key 问题不适用（那要求 CN=Android 测试密钥）。
**零售 user build + 无小米私钥 + 无 root ⇒ 平台签名路线实际不可行。**

### 21.2 独立宿主的依赖闭包（host 侧已确认）

```text
已有（模型包/插件内，可自由分发）：
  libMI_Genai_Runtime.so  libmomi.so  libxnlm.so  libxnnl.so
  libtokenizers.so  libc++_shared.so  libc++.so
缺一：libxnumd.so（vendor 分区；mida* ABI 全部 dlsym 动态解析，
      libxnnl/libxnlm 动态导入表 mida* 数为 0，无法由现有库替代实现；
      libNPURuntime 导出的是另一层 npu::umd::* C++ API，两者不同）
momi ABI（libmomi 导出，1310 符号）：Momi_LLM_create/createTask/free/
      prefill(WithModal)/generate(WithModal)/setLoraStrength/
      setPerformance/getStatistics
```

### 21.3 离线可做 / 必须设备

离线已完成：证书指纹提取、依赖闭包分析、mida/momi ABI 面确认。
离线可继续：Momi_LLM_* 反编译还原签名、base.json 全字段 schema、
XNLM 头部布局（Range 拉头部）、standalone harness APK 构建到"只差 libxnumd"。
必须设备：adb pull libxnumd.so（无需 root）；完整 vendor_sepolicy.cil 存档
（上次只摘录未保存）；platform_app 对 /dev/hpc-*、dma_heap 等的完整允许集；
（若有 key）user-installed 平台签名 App 是否获得 platform_app 域的 seapp 验证。

## 22. Root 路径调查（2026-10-26，host 侧检索）

设备：小米 18 Fold（`lhasa`），HyperOS `OS4.0.26.0.XPNCNXM`，内核 `6.18.21`。

### 22.1 GhostLock（CVE-2026-43499，futex PI UAF）

- 影响范围 5.7~7.1；2026-07 前后修复进 stable（6.1.175 / 6.6.140 / 6.12.86 等）。
  本机 6.18.21 + 2026-10 安全补丁，**大概率已含修复**。
- 可行性与内核版本无关，取决于该内核 PGO/LTO 栈布局（waiter word 落点）；
  官方支持表最高覆盖 6.12，无 6.18 waiter 布局宏；Xring O3 的
  kernel_phys_load 无人发布。已验证的小米设备只有 Xiaomi 17（pudding/SM8850/6.12）。
- 离线确证方法（无需设备）：下载 lhasa 完整 OTA ZIP → ghostlock-extract
  生成 profile；若 `remove_waiter()` 已修复，提取器以退出码 6 拒绝。

### 22.2 更优路径：o3root（CVE-2026-98260）

`zsw125/o3root` v3.0，专为本机型系列制作：

- 目标：yingtian（Pad 9 Pro Max）/ **lhasa（Xiaomi 18 Fold）**，
  HyperOS 4 / Android 17 / 内核 **6.18.21**；lhasa 与 yingtian 的 boot 内核
  逐位相同（sha256 cde87cd7…），同一载荷适用——**但作者未在 lhasa 实机测试**。
- 链路：Shizuku shell 模式（uid=2000）→ CVE-2026-98260（CLOCK_THREAD_CPUTIME_ID
  + execve/de_thread 竞态 → posix_timers_cache UAF → 任意内核 8 字节写）→
  uid0 + full caps → 加载 KernelSU/KernelSU-Next/SukiSU LKM → rootd。
- 特性：无需电脑（无线调试 + Shizuku shell）、不解锁 BL、不改分区、
  **重启即失效**（per-boot）；竞态概率性，1–6 次尝试；终态 SELinux Enforcing、
  policycap 干净；LKM 为 6.18.21 重链接，内核 taint。
- 与 Download 里已有的 `ReSukiSU_v4.2.0-rc3` 正好配套（SukiSU LKM 支持）。

### 22.3 持久 root 路径

- 非官方 BL 解锁：2026-10 出现 Xiaomi 18 Pro/18 Fold 的非官方解锁（第三方）。
  可刷 init_boot 装持久 KSU/Magisk；代价：改分区、保修、OTA、来源可信度风险。
- 国行官方解锁通道 HyperOS 时代近乎关闭（社区等级/名额制），不指望。

### 22.4 与本课题衔接

临时 root（o3root）已足够完成 §20 设计 A（换文件）/ 设计 B（Room DB 编辑）
与 §21 的设备侧清单（pull libxnumd.so、存档完整 vendor_sepolicy.cil）；
设计 C 的 system CA + 配置 MITM 也可在单次 boot 内完成注入后不再依赖 root。
建议顺序：o3root 实测（lhasa 首测，注意软砖风险=重启即恢复）→ 设备侧清单 →
设计 A 阴性对照 → 设计 A/B 注入。GhostLock 作为备选研究线离线推进（OTA→extractor）。

## 23. 设计 B 离线准备完成（2026-10-26，工具就绪待上机）

### 23.1 上机前敲定的机制（本轮离线核实）

- **路由**：`ai_process_list.json`（APK assets，version 16）里 vlm 进程的
  apiList 是固定清单；未知 api `myllm` 不命中 → fallback stub0
  （`multiprocess:stub0`，通用插件宿主）或 `plugin` 进程。全部 aiservice
  进程同属 priv_app 域，NPU 权限不受影响。llm_runtime 插件 dex 无 isLLM
  门禁，ServiceInfo 直接透传。
- **选择器**：`g(api, name, scope)` 的 name 分支走 `w1.a.e(name)` **全局名
  查找、忽略 api**。因此最终方案：`ai_config(api=myllm, name=myllm_f3)` +
  请求头 `ai_name=myllm_f3`。name 全局唯一 → 对官方 vlm/无名调用零遮蔽。
- **资源解析**：`e.c()` 按 `(id, version)` 查 `local_resource`；`unwrapped`
  须 1；**不插 remote_resource 行** → 无 need-update 标记、不会被更新覆盖。
- **行存活**：服务端同步是纯 INSERT OR REPLACE；唯一批量删除
  `DELETE FROM ai_config WHERE origin=?`（OTA 清系统分区行）。元数据列
  （origin/appVersion*/otaVersion）全部克隆真实 vlm 行。
- **DB**：`ai_db`（Room，identity `ebfee399…`），WAL 模式；推回后必须删
  `-wal/-shm`。
- **JSON 手术**：config 的 `name` 与 `plugin.id` 是同一字符串，必须
  JSON-aware 修改（inject.js 按字段改，非字符串替换）。

### 23.2 交付物（tools/designb/）

```text
inject.js    node:sqlite 注入器：克隆真实行→改 api/name/model id/版本/路径；
             --remove 回滚；INSERT OR REPLACE 与 Room 生成 SQL 一致
selftest.js  副本 DB 自测（精确建表语句+真实 vlm jsonSource）：
             注入/重注入/回滚/字段校验 12 项全过
run.sh       上机编排：预检(root)→备份→force-stop→克隆模型目录→拉 DB→
             注入→推回(修 owner/mode/context)→PoC --es api/name→logcat 留证
rollback.sh  按 TS 恢复 DB + 删克隆目录 + 校验行已清除
RUNBOOK.md   上机手册：步骤、成功判据（getAIConfig/applyLoRA path/
             generate success/status=7）、失败排查、进程进 :vlm 的
             process_list 补丁（可选）、换自转换模型的操作
```

PoC APK 已重编（`poc/taiyi-authorize-poc.apk`）：MainActivity 支持
`--es api` / `--es name` extras（默认 vlm，行为不变）。

### 23.3 上机清单（root 就绪后 ~15 分钟）

1. `nix develop && cd tools/designb && ./run.sh`
2. 看证据行：`getAIConfig myllm:myllm_f3` / `applyLoRA success, path=…/myllm_model/1/…` /
   `generate success` / `CALLBACK status=7` / `RESP_BODY ans=`
3. 成功后立即补齐 §21 设备清单（pull libxnumd.so、存档 vendor_sepolicy.cil）
4. 需要回滚：`./rollback.sh <TS>`

## 24. Mimo 切换实验与失败根因（2026-10-26/27，实机）

### 24.1 路由与资源

| 项 | 值 |
|---|---|
| `ai_api` | `mimo_dev` |
| config | `AI(mimo_dev, mimo_dev_f3, 33, false)` |
| 进程 / Service | `aiservice:mimoVlm` / `MimoVlmService` |
| 插件 | `BaseMimoVlmPlugin` |
| plugin resource | `mimo_dev_f3@25` |
| model resource | `mimo_model_f3@19` |
| param resource | `mimo_param@4` |

Mimo 与 VLM 共用同一套 Binder 入口（`ILLMManager.sendTask`、`AIRequest`），
但 `ai_api` 会把请求路由到不同进程、插件和 native engine。VLM 是
`BaseVlmPlugin` + Qwen3 0.7B；Mimo 是 `BaseMimoVlmPlugin` +
`mimo_model_f3` + LoRA base。

复现入口（`com.example.taiyi.chat`）：

```bash
adb shell am start -W -n com.example.taiyi.chat/.ChatActivity \
  --ez autotest true --ez textonly true --es model mimo
# 显式 init 实验
adb shell am start -W -n com.example.taiyi.chat/.ChatActivity \
  --ez autotest true --ez textonly true --es model mimo --ez init true
# postChat 实验
adb shell am start -W -n com.example.taiyi.chat/.ChatActivity \
  --ez autotest true --ez textonly true --es model mimo --ez chat true
```

### 24.2 实验 1：显式 `ai:method:llm_init` 能加载模型

日志（冷进程）：

```text
INIT_MODE api=mimo_dev
BaseMimoVlmPlugin [mimo_dev_f3:25] onCreate
LLM_Path: .../model/mimo_model_f3/19
Lib_Path: .../model/mimo_model_f3/19
dlopen handle_c_ / libc / xnnl / xnlm / tokenizers  (全部成功)
VLM_SDK version: 2026.0914
LLM init : .../model/mimo_model_f3/19/base.json
LLM init in: .../model/mimo_model_f3/19/base.json
llm load succeed!
LLM init time 347~484 ms
CALLBACK status=7 err=0
```

结论：`mimo_model_f3@19` 的 `base.json` 存在，权重可加载，模型包不是
“目录缺失/so 缺失”这一类问题。

### 24.3 实验 2：init 之后 `postTask` 仍然 prefill 失败

```text
CALLBACK status=7 err=0          # init 成功
SENDTASK api=mimo_dev chat=false # 正式 postTask
BaseMimoVlmPlugin: VLMEngine not ready! initialized=true isApplyLoRA=false
MI_Genai_Runtime: LLMInputParam.enableThinking 1
MI_Genai_Runtime: LLM prefill fail: status=-1
BaseMimoVlmPlugin: generate failed: code=-1, state=[0], ans=[]
CALLBACK status=6 err=10803 len=0
```

`10803` = `ErrorCode.ERROR_PROCESS_FAILED`。说明：

- 预热 init 成功，但后续任务没有进入可推理状态；
- 日志里 `isApplyLoRA=false`，且 `enableThinking=1`，这两个状态都不像
  一次正常的 Mimo chat 推理；
- 失败发生在 native prefill，耗时 0~3 ms，不是超时或内存不足。

### 24.4 实验 3：直连 `postChat` 拿不到 `jsonSource`

`BaseMimoVlmPlugin.process` 对 `postChat`/`executeChat` 读取：

```java
String jsonSource = aIRequest.getBody().getJsonSource();
if (jsonSource.length() == 0) {
    // "postChat jsonSource is empty"
    // callback code 10803, ans="chat request is empty"
}
```

我们通过 Binder 直连投递 `json_source`，无论放在 `AIBody.bundle` 第一层
还是第二层，插件都打印：

```text
BaseMimoVlmPlugin: postChat jsonSource is empty
CALLBACK status=6 err=10803 len=21 FINAL=chat request is empty
```

原因方向：真实 `AIBody.jsonSource` 是对象字段；`AIBody.readFromParcel`
只从第二个 Bundle 恢复 `frameMap`，`jsonSource` 的 bundle 回填发生在
`collect()`，而插件 `process` 在 `collect()` 之前执行。官方 SDK 走
`LLMKit.postChat` 时由 `AIBody.setJsonSource()` / `LLMKit.a(...)` 填充；
当前手写 Binder 路径没有完整复刻这套序列化。

### 24.5 结论

**Mimo 与 VLM 的外层接口相同，后端不同；Mimo 目前不可用不是单一
“接口写错”问题。** 已确认的阻塞点有两个，彼此独立：

1. 插件懒加载 init 路径只走到 so/dlopen，未完成可推理所需的
   `LLM_Init(base.json)`/LoRA 状态；显式 `llm_init` 能加载权重，但之后
   `postTask` 仍 prefill `status=-1`（`isApplyLoRA=false`、
   `enableThinking=1`）。
2. 直连 Binder 的 `postChat` 无法让服务端在 `process` 前拿到
   `AIBody.jsonSource`，插件直接返回 `chat request is empty`。

排除项：

- 不是 `ai_api` 名字错误：`getAIConfig mimo_dev` 能命中；
- 不是 Binder/auth 失败：`sendTask` 返回 true，进程/插件正常创建；
- 不是图片 Binder：文本和图片都失败；
- 不是模型目录缺失：显式 init 已加载 `base.json`；
- 不能仅凭 `10803` 判断为模型损坏，它是通用 `ERROR_PROCESS_FAILED`。

VLM 一直可用，日志显示其 `base.json` 早已完成 init；推测是相册索引等
系统功能提前预热了 VLM，所以没有暴露 Mimo 的冷启动缺陷。此“由哪个
系统功能预热”尚未实测确认。

### 24.6 后续可选路线

1. **完整复刻官方 `LLMKit.postChat`**：实现真实 `AIBody` 的
   `jsonSource` 序列化/反序列化，并确保 `applyLoRA` 在 generate 前完成；
   工作量较大，仍可能受插件 init 状态限制。
2. **找官方已预热 Mimo 的入口**：从超级小爱某个实际使用 Mimo 的功能
   触发，再复用其进程/会话；需要 UI/调用链分析。
3. **保持 VLM**：当前 demo 的默认路径；Mimo 失败自动回退。

当前 `chat` 代码保留了实验开关（`--es model mimo`、`--ez init true`、
`--ez chat true`），但默认仍是 VLM；Mimo 返回 `10803` 时自动切回 VLM，
不会自动重发。

## 24. o3root 集成 + Mimo init 补完 + git 归档（2026-10-27，离线，设备未连）

### 24.1 o3root 整合进我们的 App（已完成离线部分）

- payload 取自 o3root v3.0 release APK 的 `lib/arm64-v8a/libo3p.so`，
  sha256 `e64217c8…87dfa` 与上游文档一致 → 放到 `chat/jniLibs/arm64-v8a/`，
  以 `android:extractNativeLibs="true"` 打进 APK，运行时从
  `nativeLibraryDir/libo3p.so` 拷贝到 `/data/local/tmp/o3p_chat.bin` 执行。
- 依赖：Shizuku **13.1.5** 的 `api/provider/aidl/shared` classes.jar
  （`tools/shizuku/`）；manifest 增 `moe.shizuku.manager.permission.API_V23`、
  `moe.shizuku.client.V3_SUPPORT` meta-data、`rikka.shizuku.ShizukuProvider`
  （authority `com.example.taiyi.chat.shizuku`）；d8 需把这些 jar 一起编译。
- 代码：`chat/src/.../O3Root.java`
  （`runPayload` 走 Shizuku 隐藏 `newProcess`、`rootShell` 用 payload 自带
  `--shell` 客户端连 `@o3p_rootd`、`rooted()` 以 `id` 判定）；
  `ChatActivity` 顶栏新增 **Root** 按钮与 Shizuku 权限回调；
  自动化入口 `--ez acquireRoot true`。
- **权限模型要点（新，需设备验证）**：payload 只做 cred swap（uid0 + full caps），
  **不切换 SELinux 域**，仍在 `u:r:shell:s0`；rootd 的命令继承该域。因此对
  `com.xiaomi.aiservice` 私有数据目录（DB/模型）的访问是否被 shell 域策略放行
  尚未验证。若被拒，两条出路：(a) 在 KSU 管理器里给本 App 授权，走 KSU 的
  su 域；(b) 解锁 bootloader 后用 host `su`。设备侧验证项已记录。
- 构建校验：APK 内 `classes.dex` 位于根、`lib/arm64-v8a/libo3p.so` 哈希一致、
  dex 含 Shizuku 类（40 处引用）、manifest provider/权限/meta-data 均正确。

### 24.2 Mimo init 调查（插件级证据）

反编译 `plugin-mimo_vlm-mimo_vlm_mimo_cloud_F3-1.tapk`
（CDN 下载，sha256 `462c9acc…`，已归档 `artifacts/cdn/`）：

- 分发：`ai:method:llm_init` → `init(task)` → `initModel(task,true)`；
  `postTask` 在卡片内自建 `ChatRequest("device-qwen-4b", [user:prompt])` 再走 postChat。
- 初始化：`vlmEngine.init(path, path, path)`（三次同一路径 = 模型资源目录），
  失败返回 `AIError(10804, code)`；成功打印 `initModel success, engine version=`。
- 生成前必需 `applyLora(<modelDir>/lora_base_llm)`；失败打
  `applyLora failed, path=…, code=…`。
- **prefill 失败来自 native**：`[%s] LLM prefill fail: status=%d`
  （libMI_Genai_Runtime.so），Java 侧拿不到细节。
- mimo 模型包（`mimo_model_f3`，1.67 GB）结构：`base.json`（name `mimoclaw`、
  o3 backend、SWA rope_theta 10000/window 128、max_input 8192、head_dim 192、
  embedding_dim 2048）、`llm/hybrid_swa_compressed.bin`（1.35 GB）、
  `embedding/embedding_fp16.bin`、`utils/tokenizer_config.json`（VLM 包没有）、
  `lora_base_llm/task.json`（**EAGLE 投机解码**：draft `hybrid_swa_compressed_eagle.bin`、
  `trimMap_24k.bin`、draft_vocab 24000）、`libmomi/libxnlm/libxnnl/libtokenizers`。
- 候选根因（待设备实测）：模型未装全（可用 §24.3 预检判定）、缺失
  `lora_base_llm` 内的 eagle/trimMap、或 lazy init 半初始化后直发 generate。

### 24.3 App 侧补完（已实现）

- `TaiyiClient`：`warmMimo()` 加超时看门狗（20 s，最多 2 次尝试）、
  成功/失败时清理计时器、新增 `resetMimoWarm()`；Mimo 失败回退 VLM 时调用它，
  保证下次 Mimo 重新走 `llm_init`。
- `ModelStatus`：通过 `DownloadProvider.call("query")`（authority
  `com.xiaomi.aiservice.authority.download`，仍需 `uid=1000` 头）读取
  `local_size/full_size/download_size` 与 `ai_config_json/ai_resource_json`，
  回答"模型到底装没装全"，无需 root。
- `ChatActivity`：Mimo 发送前自动预检并打 `MIMO_STATUS`；新增 `--ez mimostatus true`。
- 构建通过：`chat/taiyi-chat.apk`（425 KB，含 payload 与 Shizuku 依赖）。

### 24.4 git 归档

- `git init` 于仓库根，96 个文件入库（源码、NOTES、证据、CDN 下载件、payload、Shizuku jar）。
- 忽略：`poc/sdk`（188 MB 工具链）、`artifacts/{pulled,dex,libs,jadx,apktool}`（可再生产物）、
  各 `build/`、`.direnv/`、`*.idsig`。
- 新增 `README.md`（仓库结构与复现方式）与 `SECURITY-NOTICE.md`（授权与风险声明）。
- 归属：payload = o3root（MIT，未修改）；Shizuku jar = Apache-2.0；LKM = GPL-3.0。

### 24.5 Mimo 根因确认（离线代码级，2026-10-27）

上一节的"候选根因"已确证，与设备日志（`initialized=true isApplyLoRA=false`
→ `LLM prefill fail: status=-1`）完全吻合：

1. **插件两条路径行为不同。** `BaseMimoVlmPlugin.process` 的分发：
   - `postTask` → 自建 `ChatRequest("device-qwen-4b")` 并 `setJsonSource()` 后
     调 `postChat` → 进入 `postChat$lambda$37`，该 worker **不 applyLoRA**，
     仅检查 `initialized && isApplyLoRA`；因 `initModel` 在已初始化时早退
     （`init has initialized`），`isApplyLoRA` 永远停在 false → 直接 generate
     → native prefill 失败。
   - `executeChat`/`postChat`（客户端发起）→ `process$lambda$32`：先
     `applyLora(<modelDir>/lora_base_llm)`，失败回 `applyLora error`；
     成功后 `isApplyLoRA=true` 才 generate。**这才是 mimo 的正确入口。**
2. **jsonSource 的传输条件（关键）。** worker 32 要求
   `request.getBody().getJsonSource()` 非空，否则直接
   `chat request is empty`（10803）。而 `AIBody`：
   - `jsonSource/jsonResult` **不参与 parcel**（只 parcel frameMap bundle）；
   - `distribute()`（发送前）与 `collect()`（接收后）**仅在
     `share_memory_enable=true` 时**才 `writeJson/readJson(KEY_JSON_SOURCE)`；
   - 官方 `LLMKit.build()` 每次都 `setShareMemoryState(true)` 并挂
     `ShareMemoryPool`，所以官方路径正常。
   我们的手写客户端设 `share_memory_enable=false` → 服务端 `collect(false)`
   直接 return → 插件读到空 jsonSource。**这就是 §24.2 观察到的空
   jsonSource 的根因。**
   绕过要点：`collect(false)` 在没有 frame 的情况下**不需要真实内存池**，
   只要 `share_memory_enable=true` + bundle 里带 `json_source`（短 JSON 走
   bundle，长 JSON 需 frame），服务端就会回填字段。
3. **本轮已实现**（`chat/`）：Mimo 自动改走 `executeChat` +
   `share_memory_enable=true` + `json_source`（`sendMimoChat()`）；
   `chatMode` 分支同样修正；保留 `llm_init` 预热与超时重试。
   设备侧预期日志：`jsonSource: {...}` → `applyLoRA success, path=…/lora_base_llm`
   → `generate success`；若仍报 applyLora 失败，则指向模型目录内
   `lora_base_llm`（含 `hybrid_swa_compressed_eagle.bin`/`trimMap_24k.bin`）缺失或版本不符。

### 24.6 Mimo 实机验证通过（2026-10-27，无 root）

设备 `lhasa` / `OS4.0.26.0.XPNCNXM`，`com.xiaomi.aiservice` 4.2.12，**全程未 root**。

预检（`--ez mimostatus`，DownloadProvider query）：
`mimo_dev full=1.94 GB left=0 B state=0`，bundle（+lora/llm_runtime/vlm）`full=2.85 GB left=0`
→ 模型早已装全，之前的失败与安装无关。

正确链路（chat app 自动执行）实测：

```text
MIMO_WARMUP sent=true attempt=1
mimoVlm 进程拉起 → LLM init time 337 ms → CALLBACK status=7 err=0
MIMO_CHAT sent=true (executeChat + share_memory_enable=true + json_source)
  applyLora, lora_Root_Path: .../model/mimo_model_f3/19/lora_base_llm
  EAGLE engine initialized: draftLength=3 numBranches=4 maxTargetTokens=16
  The lora init success!                        [applyLora total 217 ms]
  LLM Prefill: 1230.954 ms / Prefill Speed: 100 tokens, 79.98 t/s
  mimo_dev prefillTime: 1230, decodeTime: 590, cost: 2058
  CALLBACK status=7 len=36 FINAL=你好，我是运行在这台手机上的本地 VLM 助手，可以帮你处理文字和图片。
  METRICS ttft=2178 tps=14.9 tok=10 total=2834
```

第二次发送（引擎已热）走新增的"已初始化"分支，**TTFT 从 2178 ms 降到 197 ms**：

```text
CALLBACK status=7 err=10804            ← 插件对重复 llm_init 的"init has initialized"
MIMO_WARMUP already initialized (10804) -> continue
LLM Prefill: 124.343 ms
METRICS ttft=197 tps=15.0 tok=10 total=839
```

VLM 回归同时确认无影响：`applyLoRA success` → `generate success: token=34`，
`METRICS ttft=799 tps=89.1 tok=33`。

新增修复（本轮实机发现）：插件在引擎已初始化时对重复 `llm_init` 回
`status=CODE_RESET_ERROR + err=10804`，这是"已预热"而非失败——客户端已按此处理
（`MIMO_WARMUP already initialized (10804) -> continue`），并抽出
`sendPendingMimo()` 复用。

## 25. 能否用 Shizuku（adb 提权）替代 root？（2026-10-27，实机 + 代码级）

结论：**直接的注入路径不能替代 root；但存在一条"半条"无 root 路径——下载暂存目录的
TOCTOU 替换**。这条路径若成立，能做到"在 aiservice 私有目录写入我们控制的插件/模型"，
即 priv_app 域内代码执行。

### 25.1 shell 的能力边界（实机实测）

| 目标 | shell(uid 2000, u:r:shell:s0) |
|---|---|
| `ls /data/user_de/0/com.xiaomi.aiservice/` | **Permission denied** |
| 读 `databases/`、`head ai_db` | **Permission denied** |
| `touch taiyi/probe_test`（写） | **Permission denied** |
| `am force-stop com.xiaomi.aiservice` | 允许 |
| DownloadProvider `query`（uid=1000 头） | 允许（只读状态/配置） |
| 读/写 `/sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi/...` | **允许**（touch/echo/cat 全部成功） |

所以 §20 的设计 B（改 `ai_db` 行 + 克隆模型目录）**必须 root**，Shizuku 做不到。

### 25.2 那条"半条"路径：下载暂存目录 + 安装链 TOCTOU

代码级证据（`artifacts/jadx/aiservice/sources`）：

1. **下载目标就是 shell 可写目录**：
   `v1.c.f8500w = lazy(n2.a(26))` → `getExternalFilesDir(null) + "/download/taiyi"`；
   `com/xiaomi/aiservice/download/u.java:64-70` 拼出
   `<root>/<type>/<id>/<version>/<fileName>`；设备上该树实际存在
   （`model/…`、`plugin/…`、`param/…`，每个资源目录当前为空骨架）。
2. **下载完成校验不是对文件字节做的**：`x3/e.java` 的 `onFileStateChange`
   在 `SUCCESS` 时比较的是 **配置推导值**（远端行 `b2.a.f2748j/f2749k` vs 下载记录
   `z3.a.f9272h/f9273j`），不哈希磁盘文件；随后才异步排入安装。
3. **安装期校验分两类**（`v2/g.java` ResourceInstaller + `ResType.isZip()`）：
   - `LIBRARY/PARAM/FILE`（非 zip）：拷贝后**真哈希**（`m3.a.n(srcPath)` vs 行内
     `fileHash`）→ 替换无效。
   - `PLUGIN/MODEL`（zip）：解压后读**包内自带** `taiyi_hash.json` 逐文件比对，
     且只校验清单里列出的、确实存在的文件 → **自证，可伪造**。
4. **唯一的空哈希资源**：`bert_model_npu_xring_o3@5`（MODEL，219,358,082 B，
   `fileHash` 为空）→ 连 size 之外的哈希比对都不存在。
5. **拦路的是下载状态**：`download/core/s.java` 只有在
   "记录 state=SUCCESS 且目标文件存在" 时才跳过下载直接校验+安装；而记录存放在
   **私有 Room 库**（`x3/g` → Room DAO），shell 无法伪造。
6. **配置不能伪造**：`AIDetail.parse(api, response)` 只消费响应；无明文 http 端点；
   配置来自 HTTPS 服务端 → 想改 URL 需要可信 CA（=root）。

因此无 root 的可行形态是 **TOCTOU**：在下载完成之后、ResourceInstaller 拷贝/解压之前，
把暂存目录里的包原子替换（`mv`）。目标选 **PLUGIN**（如 `mimo_dev_f3`）：自证校验会放行，
替换后的 APK 会被 AIPluginManager 以 `entryClass` 载入 aiservice（priv_app + NPU 域）
→ 等于无 root 的代码执行。成立的两个前提：(a) 有一个**当前未安装**的资源可触发下载
（用 DownloadProvider 查 `left_download_size > 0`），(b) 赢下那个窗口。

### 25.3 对 root 路线的连带结论

o3root 的 payload 只做 cred swap（uid 0 + full caps）**不切换 SELinux 域**，仍在
`u:r:shell:s0`。uid 0 只绕过 DAC（文件权限），**MAC（SELinux）仍然生效**：本轮的
shell 域拒绝证据意味着 o3root 起 root 后访问 `com.xiaomi.aiservice` 私有目录
**大概率同样被拒**。要让设计 B 落地，需要：
(a) KSU 的 `su` 域（在 KSU 管理器里给本 app 授权后 `su` 会切到 kernel su 域），或
(b) 解锁 bootloader 后 host `su`。
→ 这是设备回来后必须实测的第一项：`o3root` 起 root 后 `id` + 直接 `ls` 私有目录。

### 25.4 本轮未完成的实验（设备掉线中断）

- 用 `poc/download-probe`（已 build，未安装）扫描全部 scene/api，找
  `left_download_size > 0` 的**可触发下载**资源；
- 若有，按 §25.2 尝试 TOCTOU 替换（需用户确认，因为会触发真实下载与安装）。

## 26. 无 root 路径的应用整合 + 测试方案（2026-10-27，设备离线期间完成）

按 §25 的结论，把"暂存目录替换（TOCTOU）"这条无 root 路径**整合进 chat app**，
并在宿主机上对机制做了可复现的验证；设备回来后按
[`tools/TESTPLAN-noRoot.md`](tools/TESTPLAN-noRoot.md) 直接开跑。

### 26.1 新增代码

| 文件 | 作用 |
|---|---|
| `chat/src/.../Shell.java` | 纯 Shizuku shell（uid 2000，无 root）：`exec` / `execAsync` / `writeScript` |
| `chat/src/.../StagingSwap.java` | 暂存树扫描、替换脚本下发、payload 发布/搬运、`arm()` 武装替换、`targetFor(type,id,ver,fileName)` |
| `chat/src/.../DownloadTrigger.java` | 探测资源状态（复用 `ModelStatus.newRequest`）+ 启动官方 `DownloadService` 触发下载 |
| `chat/assets/staging_swap.sh` | 替换循环本体（打进 APK assets，运行时 heredoc 写到 `/data/local/tmp/`） |
| `tools/staging/mkpack.js` | 生成**精确尺寸**的 ZIP（条目在前、零填充居中、中央目录+EOCD 在末尾，兼容 `ZipFile`） |
| `tools/staging/swap-sim.js` | 宿主机仿真：用同一份脚本 + 两种写入风格 + "安装器拷贝"，断言安装器读到我们的包 |

新增自动化入口（`am start` extras）：
`--ez staging`（扫描暂存树）、`--es apis/--es scene`（资源状态探测）、`--ez dlstart`
（触发下载）、`--ez armswap`+`--es swaptarget/swappayload`+`--ez swapesize/swapdur`
（武装替换）、`--ez probeAccess`（三路访问探测：shell / su / 暂存可写）；
Root 芯片长按 = 扫描暂存目录。

### 26.2 离线验证（可复现）

```bash
# 1) 精确尺寸 ZIP：1,000,000 字节，unzip -t 无错，条目可读
node tools/staging/mkpack.js --out /tmp/pkg.zip --size 1000000 --add "mock.apk=..." 
#    min 体积 109 字节；161 字节（vlm_param 练手目标）也可行，已验证
# 2) 替换机制仿真（跑的是设备上同一份 staging_swap.sh）
node tools/staging/swap-sim.js
#   -> PASS: installer read our payload in all styles (fd, reopen)
```

仿真过程中发现并修掉一个真实缺陷：脚本最初用 `ln`（O(1) 硬链接）替换，导致
**替换文件与目标共享 inode**，写者继续 append 会把我们的包一起写大，后续替换全部失效
（`reopen` 风格只成功 1 次）。改为每次 `cp` 后，两种风格都通过。这条已写进脚本注释。

APK 校验：`assets/staging_swap.sh` 进包且 APK 内哈希与工作区一致
（`8e46f0df…b8826`），即"被验证过的脚本 = 设备上运行的脚本"。

### 26.3 现实前提（唯一未决项）

替换窗口必须落在**一次真实下载**上：`DownloadProvider` 只有 `query/pause/cancel`，
下载状态记录在私有 Room 库（`x3/g`），shell 无法伪造"已下载"。而本机 2026-10-27 实测
`mimo_dev` 及 bundle 全部 `left=0`（已装全）。因此设备回来后第一步是**找 `left>0` 的
资源**（`--es apis "<39 个 api>"`，命令写在测试方案 §3），或在资源版本更新后重试；
核选项（`pm uninstall --user 0` 后 `cmd package install-existing`）会清掉 4.6 GB 已装
模型，需明确同意。

### 26.4 风险与回滚

- 替换会改写系统应用 `com.xiaomi.aiservice` 的私有资源目录；无 root 时**无法直接删除**
  私有文件，官方回滚 = 让 AIService 重新下载并安装同一资源（需要 `left>0` 或版本更新），
  否则只能等 OTA 覆盖。
- 因此测试方案给了**低风险练手**步骤：拿 161 字节的 `param/vlm_param/3/vlm_param.json`
  当靶子（非 zip + 带哈希 → 安装期真哈希校验必然失败），成功判据是 AIService 侧出现
  `copy hash not match` —— 这证明我们的包**确实到达了安装步骤**（窗口存在），
  而资源功能不受影响。
- 不碰内核、不需要解锁 BL、失败时资源保持原样或被重新下载。

### 26.5 与 root 路线的关系

`--ez probeAccess` 一次给出三路结论：shell 域能否进私有目录（预期 denied）、设备是否有
`su`（KSU 是否可用）、暂存目录是否可写。若最终判定无 root 路线不可行（例如始终找不到
可触发下载），root 路线的下一步是"在 app 内用 `su` 跑设计 B"（把 `tools/designb/inject.js`
的逻辑用 Android `SQLiteDatabase` 重写：su 备份 DB → app 内改 → su 写回并 restorecon →
克隆模型目录 → 重启 aiservice）；现有 host `tools/designb/run.sh` 仍可直接使用。

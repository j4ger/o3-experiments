# Design B 上机手册（AIService 注入自转换/克隆模型）

目标：在 **root（临时 root 即可）** 条件下，把一个新的 `myllm` 配置注册进
AIService 的 Room DB（`ai_db`），复用已安装的 vlm 插件（BaseVlmPlugin）与
模型目录克隆，用普通第三方 PoC APK 走 `sendTask(ai_api=myllm, ai_name=myllm_f3)`
让模型实际执行。离线部分已全部验证（见 §23）。

## 已离线确认的机制（勿再现场验证）

| 事实 | 出处 |
|---|---|
| 配置/资源存于 Room `ai_db`（`ai_config`/`local_resource`/`remote_resource`） | jadx `x1/f.java`, `h5/c.java`（db 名 `ai_db`） |
| 请求按 `ai_name` 全局查找配置（忽略 api）；无名请求按 api 列表 | `configuration/e.java g()` `w1.a.e()` |
| 资源按 `(id, version)` 查 `local_resource`；`unwrapped` 须为 1 | `e.c()`（statisticsResource） |
| 不插 `remote_resource` → 无"需要更新"标记，不会被自动更新覆盖 | `e.c()` 行 223/233 |
| 服务端同步仅 INSERT OR REPLACE；唯一批量删除按 `origin`（OTA 清系统分区行） | `x1/a.java`, `androidx/room/e0.java` |
| 未知 api 的进程路由 fallback：stub0（通用插件宿主）/plugin 进程，同 priv_app 域 | `ai_process_list.json`, `ProcessInfo.m2default()` |
| 加载链无哈希/签名校验 | §20.1 |

## 上机步骤（root 就绪后）

```bash
nix develop
cd tools/designb
./run.sh          # 全流程：备份→克隆→注入→实测
./rollback.sh <TS>  # 出问题回滚（TS 见 run.sh 输出）
```

`run.sh` 做的事：
1. 预检：设备在线、`su -c id`=uid0、定位 `ai_db`；
2. 备份 DB 到 `/data/local/tmp/designb-backup-<TS>/` + host 侧 pristine 副本；
3. `am force-stop` 停 aiservice（内存缓存 `w1.a` 必须重载）；
4. `cp -a` 官方模型目录 → `taiyi/model/myllm_model/1/`（修 owner/restorecon）；
5. 拉 DB → `inject.js` 插入 `ai_config(myllm, myllm_f3)` +
   `local_resource(myllm_model@1)` → 推回、删 WAL、修 owner/mode/context；
6. 装 PoC，`am start --es api myllm --es name myllm_f3`，logcat 留证。

## 成功判据（按强度排序）

```text
getAIConfig myllm:myllm_f3: -> [AI(myllm, myllm_f3, 24, false)]   # 配置被选中
BaseVlmPlugin ... applyLoRA success, path=.../myllm_model/1/...   # 用了我们的目录
BaseVlmPlugin ... generate success: token=N                       # 引擎真实执行
TaiyiPoc CALLBACK id=... status=7 err=0                           # 任务完成回包
TaiyiPoc RESP_BODY ans=...                                        # 内容返回
```

注意：进程会是 `aiservice:stub0`（或 `:plugin`），**不是** `:vlm`——这是
预期行为（路由 fallback），不影响功能与权限（同 priv_app 域）。若想让任务
进 `:vlm` 进程：root 编辑 aiservice 的 SharedPreferences
（`v1.f` 对应的 prefs 文件，key `process_list`），version 改 99 并把
`"myllm"` 加进 vlm 进程的 `apiList`。仅功能验证时不必做。

## 失败排查

| 现象 | 处置 |
|---|---|
| `getAIConfig myllm:... -> []` + ERROR_CONFIG_EMPTY | 行没读到：确认 DB 推回成功、force-stop 生效；pristine 副本对照列名 |
| `statisticsResource myllm_model ... required but not found` | `local_resource` 行缺/版本不匹配（config.models[0].version 必须=1） |
| detail 可用但 dispatch 后无 plugin 日志 | stub0 进程宿主问题 → 做 process_list 补丁（见上），或检查 `:stub0` 是否起来 |
| `LLM prefill fail` / native 崩溃 | 先对官方 `vlm` 跑一次 PoC 确认基线；再查克隆目录 `base.json` 存在性 |
| aiservice 起不来 | `./rollback.sh <TS>`，或直接恢复备份 DB |

## 参数约定（与 §20.2 一致）

- api=`myllm`，name=`myllm_f3`（全局唯一，无遮蔽风险）
- model id=`myllm_model`，version=1（目录 `taiyi/model/myllm_model/1/`）
- plugin/param 复用官方已装资源（`vlm_qwen3_0_7b_f3@18` / `vlm_param@3`）
- 不插 `remote_resource`；origin/版本等元数据全部克隆真实行

## 换成自转换模型时（转换器就绪后）

只需替换 `taiyi/model/myllm_model/1/` 内容为转换器产物（整包布局：
`base.json` + `llm/` + `embedding/` + `tokenizer/` + `lora_base_*/` + `*.so`），
DB 不动；`size` 字段为 0 时运行时自动按目录计算。架构不同的模型需同步改
`ai_config` jsonSource 的 `llm_info` 与 `models[0]`。

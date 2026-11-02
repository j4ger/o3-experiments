# 设备提取清单（2026-11-03）

设备：`lhasa` 2608BPX34C，serial `08030C6049983540`
提取工具：`tools/device/collect.sh`
原始目录：`artifacts/device/20261103-015444/`（gitignored）
清单：`artifacts/device/20261103-015444/manifest.json`

## 结果

- 27 项成功，8 项失败。
- `libxnnl.so` / `libxnumd.so` 与本地已有副本 sha256 完全一致。
- 模型目录自带的 `libxnnl_model.so` 与 CDN 包内 `libxnnl.so` sha256 完全一致：
  `e8bc1b849fdefad35d6faed29b4aa4f99bb5edae5cd79a2bf54e2ced683961db`。
- `/vendor/etc/npu/**` 已整体取回，包含 AON cipher 模型与 AINR `.bell`；
  这些是相机 NPU 资产，不是 LLM GemmSplitMN 直接需要的文件。

## 已取回的关键项

| 类别 | 路径 |
|---|---|
| vendor 库 | `/vendor/lib64/libxnnl.so`、`libxnumd.so`、`libxnlm.so`、`libxnn*.so`、`libNPURuntime.so`、`libNBGlinker.so`、`libOpenVX.so`、`libovxlib.so`、`libVSC.so`、`libOpenCL.so` |
| system 库 | `/system/lib64/libnpumanager.so`、`libvip_channel.so` |
| NPU 配置 | `/vendor/etc/npu/**` |
| 设备侧临时证据 | `/data/local/tmp/collect.tgz`、`libxnnl_model.so`、`rms.dot`、`cu2.log`、`cu2b.log`、`probe.log`、`rootd.log` |
| 运行快照 | `getprop.txt`、`uname.txt`、`dev-hpc.txt`、`interrupts.txt`、`iomem.txt`、`logcat-all.txt`、`devices.txt` |

## 失败项与原因

| 路径 | 原因 |
|---|---|
| `/vendor/lib64/libNPURuntime_wrapper.so` | Permission denied |
| `/vendor/lib64/libNNVXCBinary.so` | Permission denied |
| `/vendor/lib64/libOpenVXU.so` | Permission denied |
| `/vendor/lib64/libOvx12VXCBinary.so` | Permission denied |
| `/vendor/lib64/libOvxGPUVXCBinary.so` | Permission denied |
| `/vendor/lib64/libnpu_ainr.so` | Permission denied |
| `/data/local/tmp/o3p_rootd.log` | Permission denied |
| `/data/local/tmp/collect` 目录 | 目录内 `o3p_rootd.log` 无法读；其余文件已单独取回 |

说明：`/vendor/lib64` 的目标文件在 `adb shell`（uid shell）和 `o3p_chat.bin`（uid 0 但 SELinux 仍是 `u:r:shell:s0`）下都被拒绝；
需要 KernelSU 的 `su` 域才可能读取，当前没有可靠 headless 入口。

## 受保护 App 数据

尝试读取：

```text
/data/user_de/0/com.xiaomi.aiservice/files/download/taiyi
```

结果：

```text
find: ... Permission denied
```

`/sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi` 下能看到目录名，
但 model/param/plugin 的具体版本目录为空；实际模型已移入内部数据。
因此不把 App 数据提取作为阻塞项；模型包走 CDN 下载，已有 `libxnnl_model.so` 作为模型侧 `libxnnl.so` 的等价副本。

## 复现

```bash
tools/device/collect.sh
# 或指定输出目录
tools/device/collect.sh artifacts/device/20261103-015444
```

脚本不会因单个 pull 失败退出；失败会写入 `manifest.json`，便于下次续传。

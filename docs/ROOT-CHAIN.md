# 应用内的临时 root 链路（o3root 整合）

`apps/o3root` 把上游 [zsw125/o3root](https://github.com/zsw125/o3root)（MIT）的
临时 root 链路整进去了，并额外挂了一个 NPU 探针。**不碰分区、不解锁 bootloader、
不改 vbmeta**；root 只在本次开机有效。

## 1. 组成

| 文件 | 作用 |
|---|---|
| `apps/o3root/jniLibs/arm64-v8a/libo3p.so` | 上游 v3.0 payload，**与官方 release byte-identical**（sha256 `e64217c8…`，1,492,484 B，内含 KernelSU / KernelSU-Next / SukiSU 三个 LKM 容器） |
| `apps/o3root/jniLibs/arm64-v8a/libprobex.so` | 我们自己的 OpenVX/OpenCL 探针（`tools/probe/probe_vx.c`，NDK 29 编译，AArch64 PIE，7,936 B） |
| `apps/o3root/src/.../O3Root.java` | Shizuku 驱动 + payload 执行 + `rootShell()` + `runNpuProbe()` |
| `apps/o3root/src/.../ChatActivity.java` | `acquireRoot()`、`npuProbeFlow()`、`runNpuProbeNow()` |

## 2. 运行链路

```
App ──Shizuku.newProcess──▶ /system/bin/sh (uid 2000, u:r:shell:s0)
   └─ cp libo3p.so /data/local/tmp/o3p_chat.bin && chmod 755 && exec
        └─ CVE-2026-98260 (POSIX timer UAF) → uid 0 + full caps
             ├─ stage 0: 清 selinux_state.enforcing
             ├─ stage 1: task_struct.cred → 伪造 cred 页
             └─ insmod KernelSU LKM（运行时 /proc/kallsyms 重链接）
                  └─ rootd @o3p_rootd（接受 uid 0/2000）
App ──Shizuku shell──▶ o3p_chat.bin --shell '<cmd>'  → rootd 执行
```

NPU 探针就挂在这条链上：

```
npuProbeFlow() → stageProbe()（把 libprobex.so 拷到 /data/local/tmp/o3p_probe）
              → rooted? 否则 acquireRoot()
              → runNpuProbe(): setenforce 0; /data/local/tmp/o3p_probe; setenforce 1
```

探针会 `dlopen("libOpenVX.so")` → `vxCreateContext()`，并枚举 OpenCL 设备；
输出走 logcat 的 `NPU_PROBE` 标签。

## 3. 触发方式

```bash
# 一键（自动提权 → 跑探针）
adb shell am start -n dev.j4ger.o3root/.MainActivity --ez autorun true

# 只提权
adb shell am start -n dev.j4ger.o3root/.MainActivity --ez autorun true
```

## 4. 前置条件

1. 设备上装好 **Shizuku** 并以 **shell 模式**启动（无线调试 / USB 调试，
   `shizuku_plus_server` 必须是 `uid=2000`、`u:r:shell:s0`）。**root 模式不行**——
   这条链需要 shell 域。
2. 我们的 APK 已安装：`adb install -r apps/o3chat/o3chat.apk`。
3. 首次运行需在弹窗里授予 Shizuku 权限。

## 5. 对 payload 的静态审查（我们在本轮做的）

`payload/o3p.c`（1,136 行）审查结论：

- **无网络**：`socket()` 只用于 `AF_UNIX` 的 `@o3p_rootd`，没有 `connect` 到外部。
- **写入面**：仅 `/data/local/tmp/o3p_*`（外加可选的 `/sdcard/o3p_sd.log`）；
  `/data/app` 只读，用于识别装了哪个 KernelSU 管理器。
- **无 mount / 无分区写 / 无 reboot**。
- **内核侧**只有两处写：`selinux_state - 7`（写 0）与 `task_struct.cred`（指向伪造
  cred 页），没有额外的任意写原语。
- 常量与本机内核匹配：`SELINUX_STATE_STATIC 0xffffffc082959630`、
  `INIT_USER_NS_STATIC 0xffffffc0826d10d8`、`cred @ task+0x908`。

## 6. 风险

- **概率性**：UAF 竞态，1–6 次尝试；失败可能崩溃/重启（重启即恢复原状）。
- **内核污染**：LKM 为 6.18.32 构建、重链接到 6.18.21，`/proc/modules` 带 `(O)`。
- **KSU 授权列表**在 `/data/adb` 下跨重启保留，实验完要清理：
  `rm -rf /data/adb/ksu /data/adb/modules`。
- **lhasа 未实机验证**：上游只在 `yingtian`（内核声称逐位相同）验证过。
- 上游明确说："不要当生产级 root 方案"。

## 7. 清理

```bash
adb shell "rm -f /data/local/tmp/o3p*"
# 需要 root：
adb shell "su -c 'rm -rf /data/adb/ksu /data/adb/modules'"
adb uninstall dev.j4ger.o3chat
# 重启：SELinux 回到 Enforcing，root 消失
```

## 8. 实机运行记录（2026-10-28，首次成功）

在 lhasa 上以 Shizuku 13.6.0（`shizuku_server` 以 `uid=shell` 运行）为入口，
一次成功：`KSU_MODULE LIVE`、`ROOTD start uid=0`、`UIDNOW uid=0 euid=0`、
`FULLROOT armed=1`、`rooted probe: uid=0(root)`、`O3ROOT exit=0 rooted=true`。
完整日志：[`artifacts/device/root-run-2026-10-28.log`](../artifacts/device/root-run-2026-10-28.log)。

**但 NPU 探针被 SELinux 拦住**：

```
CapEff: 000001ffffffffff          (full caps，含 CAP_MAC_ADMIN)
context=u:r:shell:s0              (域未改变)
setenforce 0                      -> Permission denied
echo 0 > /sys/fs/selinux/enforce  -> rc=1
exec 3<>/dev/hpc-cdev             -> Permission denied
runcon u:r:platform_app:s0        -> Permission denied
```

即：**提权本身成功，但 shell 域没有 `security:security setenforce`、
没有 `npu_device:chr_file open`、也没有 `process transition`**。
这与第四轮的访问矩阵完全一致。

## 9. 用 KernelSU 自己的策略补丁放行 NPU（保持 Enforcing）

提权后进程仍在 `u:r:shell:s0`，而该域没有 `/dev/hpc-cdev` 的 open 权限。
**不需要把整机设成 permissive**：KernelSU 自带策略补丁工具 `ksud sepolicy`，
加一条定向 allow 即可。

### 规则（从设备策略里逐字推出）

设备上所有 NPU 客户端域（`hal_npu`、`priv_app_36`、`platform_app`、`minetd` …）持有的
权限集合完全一致，且 `npu_device` 上**没有 `allowxperm`**（不需要 ioctl 白名单）：

```
allow shell npu_device:chr_file { append getattr ioctl lock map open read watch watch_reads write };
```

### 执行

```bash
tools/root/npu_enable.sh /path/to/ksu-manager.apk
```

脚本会从 KernelSU 管理器 APK 里取出 `lib/arm64-v8a/libksud.so`（即 `ksud`），
推到 `/data/local/tmp/ksud`，然后依次 `sepolicy check` → `sepolicy patch` → 用
`(exec 3<>/dev/hpc-cdev)` 验证。应用内也内置了同样的流程：
`O3Root.patchSelinux()` / `runNpuProbe()`，探针优先走这条 Enforcing 路径，
失败才回退到 `setenforce 0/1`。

### 性质

- 只给 `shell` 域开这一个字符设备，系统其余部分仍是 **Enforcing**；
- 补丁在**内核内存**里，重启即消失（和 root 一样是 boot-only）；
- 因此可以做成可重复的流水线：CVE 提权 → `ksud sepolicy patch` → 跑探针。

## 10. 两个 app 的拆分（2026-10-28）

特权面与聊天面分开，避免聊天应用继承任何 root 能力：

| app | 包名 | 内容 | 体积 |
|---|---|---|---|
| **O3 Root** | `dev.j4ger.o3root` | o3root payload、`ksud`、OpenVX 探针、Shizuku 驱动、提权/放行/探针 UI | 4.2 MB |
| **O3 Chat** | `dev.j4ger.o3chat` | 仅聊天（本地 VLM），**不含任何 native 库与 exploit 代码** | 62 KB |

接口：聊天页的 `Root` 按钮 / `--ez npuprobe true` 只是 **deep link** 到
`dev.j4ger.o3root/.MainActivity`（带 `autorun`），不做任何特权操作。

### 构建与安装

```bash
./o3root/build.sh      # -> o3root/o3root.apk   (dev.j4ger.o3root)
apps/o3root/build.sh        # -> apps/o3root/o3root.apk   (dev.j4ger.o3root)
apps/o3chat/build.sh        # -> apps/o3chat/o3chat.apk   (dev.j4ger.o3chat)

adb install -r o3root/o3root.apk
adb install -r apps/o3chat/o3chat.apk
# 旧的 com.example.taiyi.chat 可以直接卸载
adb uninstall dev.j4ger.o3chat
```

### 运行

1. Shizuku 以 shell 模式启动；
2. 打开 **O3 Root**，首次授予 Shizuku 权限；
3. `1. 获取临时 root` → `2. 放行 NPU + 跑探针`；
4. 聊天应用不再参与提权，`Root` 按钮仅跳转。

### 说明

- 聊天应用里原有的 `staging`/`probeAccess` 等研究开关仍保留（它们只用到
  adb 级 Shizuku shell，不涉及 root），需要给 `dev.j4ger.o3chat` 单独授予
  Shizuku 权限。
- `dev.j4ger.o3root` 单独签名（同一个 debug keystore），与聊天应用互不依赖。

# KernelSU 的提权/策略机制，以及我们踩的坑

设备上跑通 o3root（CVE-2026-98260 → uid 0 + KernelSU LKM）之后，出现了两个反直觉现象。
本文用 KernelSU 上游源码（`git clone https://github.com/tiann/KernelSU`，17 MB）解释原因，
并给出正确做法。

## 1. 现象

```
uid=0(root)  CapEff=000001ffffffffff   context=u:r:shell:s0
(exec 3</dev/hpc-cdev)   -> OK        # 只读打开成功
(exec 3<>/dev/hpc-cdev)  -> DENIED    # 读写打开失败
touch /data/local/tmp/x  -> DENIED    # 连自己目录里建文件都不行
chmod /data/local/tmp/p  -> EPERM
```

节点本身是 `cr--r--r-- system system u:object_r:npu_device:s0`（**0444 只读**）。

## 2. 根因：SELinux 的 `capability` 类，不是 DAC

原厂策略里 **`shell` 域的 `capability` 类一条 allow 都没有**：

```
$ sesearch -A -s shell -c capability artifacts/device/sepolicy | wc -l
0
```

而其他域都有，例如：

```
allow init init:capability { audit_write chown dac_override dac_read_search fowner ... }
```

所以：

- `CapEff` 是内核 DAC 层的凭据，**SELinux 还会再过一道 `capability` 检查**；
- `shell` 域没有任何 capability 授权 → `dac_override`/`fowner`/`chown` 全部被拒；
- 只读打开 0444 节点不需要任何 capability，所以成功；
- 写打开 / 在 `drwxrwx--x shell shell` 的目录里创建文件（fsuid=0 不是 owner）需要
  `dac_override` → 被 SELinux 拒；
- 这也解释了为什么 `setenforce` 被拒（`security` 类），但 `ksud sepolicy patch` 能成功。

## 3. KernelSU 自己是怎么做的

### 3.1 su 域是 permissive

`kernel/selinux/rules.c` 的 `apply_kernelsu_rules()` 里：

```c
ksu_permissive(db, KERNEL_SU_DOMAIN);
```

`KERNEL_SU_CONTEXT = "u:r:" KERNEL_SU_DOMAIN ":s0"`（`kernel/selinux/selinux.h:11`）。
**KSU 把自己的 `su` 域直接设成 permissive**，所以进入该域后 MAC 不再拦截。

### 3.2 提权用真实 cred，不是伪造

`kernel/policy/app_profile.c:125` `escape_with_root_profile()`：

```c
cred = prepare_creds();               // 完整、合法的 cred
...
cred->uid = cred->suid = cred->euid = cred->fsuid = profile->uid;   // 0
cred->gid = ... = profile->gid;
cred->cap_* = profile->capabilities;  // 完整 cap 集
commit_creds(...);                    // 之后切到 u:r:su:s0
```

对比 o3root payload 的做法：它把 `task_struct.cred` 指针换成一张**伪造页**，只填了
uid/gid 和 cap 位图，**域仍然是 `u:r:shell:s0`**——所以拿不到任何 SELinux 权限。

### 3.3 但 su 路径被访问控制挡住了

`kernel/supercall/perm.c`：

```c
bool allowed_for_su(void) { return is_manager() || ksu_is_allow_uid_for_current(current_uid().val); }
```

`KSU_IOCTL_GRANT_ROOT`（LKM 模式下的提权入口）用的就是它。而
`kernel/policy/allowlist.c`：

```c
if (unlikely(uid == SHELL_UID) && !ksu_shell_su_allowed_for_current()) return false;
return ksu_is_uid_granted_root(uid);
```

即：

- **必须是 KernelSU 管理器 app，或 uid 在允许列表里**；
- **shell (uid 2000) 被显式拒绝**，除非管理器里打开了"允许 shell 提权"；
- `escape_with_root_profile()` 还有一条 `if (cred->euid.val == 0) { "Already root, don't escape!" }`
  —— 从我们已有的 root 进程里再请求会被直接拒。

所以：**没有管理器 + 没有允许列表时，KSU 的 su 用不了**，我们不能借它的 permissive 域。

### 3.4 我们能用的只有 SET_SEPOLICY

`kernel/supercall/dispatch.c:832`：

```c
{ .cmd = KSU_IOCTL_SET_SEPOLICY, .handler = do_set_sepolicy, .perm_check = only_root }
```

**只要 uid 0 就能打策略补丁**——这就是为什么 `ksud sepolicy patch` 能成功，而且
（实测）立即生效：补丁写进内核里已加载的策略库，`/sys/fs/selinux/policy` 可读到变化。

## 4. 结论：正确做法

对"已经通过 CVE 拿到 uid 0"的我们来说，正确路径不是去够 KSU 的 su，而是
**用 `ksud sepolicy patch` 给 shell 域补齐它需要的权限**：

```
# 1) NPU 设备节点（已验证：只读打开成功）
allow shell npu_device chr_file { append getattr ioctl lock map open read watch watch_reads write }

# 2) capability —— 缺的就是这条
allow shell shell capability { dac_override dac_read_search fowner fsetid chown }
```

注意 `ksud` 的语法是 **`allow <源> <目标> <类> { 权限... }`**（空格分隔、无冒号），
写成 `allow shell npu_device:chr_file {...}` 会报 `Failed to parse policy statement`。

加上第 2 条之后，rootd 才是一个"能干活"的 root：可以 chmod/cp、可以以 O_RDWR 打开
0444 的 NPU 节点。

### 如果不加 capability

也可以完全绕开：**文件操作走 uid 2000 的 adb shell**（它能 chmod 自己目录里的文件），
rootd 只负责跑探针。此时只剩一个问题：NPU 驱动是否接受只读 fd 上的 ioctl。
如果接受，连 capability 都不用补。

## 5. 安全提醒

给 `shell` 域加 `dac_override` 意味着 **adb shell 也能绕过 DAC**（不只是我们的 rootd）。
补丁只在内存里，重启即失效；但这确实是一次真实的权限放宽，只应在研究设备上、且
实验结束后重启清理。

## 6. 参考

- `kernel/policy/app_profile.c` — `escape_with_root_profile()`
- `kernel/selinux/rules.c` — `apply_kernelsu_rules()` / `ksu_permissive()`
- `kernel/selinux/selinux.c` — `setup_ksu_cred()` / `setenforce()`
- `kernel/supercall/dispatch.c` — ioctl 表与 `only_root` / `allowed_for_su`
- `kernel/supercall/perm.c`、`kernel/policy/allowlist.c` — 访问控制
- `userspace/ksud/src/su.rs`、`cli.rs` — `su` 是 `ksud` 的软链，按 argv[0] 分发

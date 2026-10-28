# O3 Root (`dev.j4ger.o3root`)

Privileged research tool: it obtains temporary root on a stock, locked,
verified-boot device and then drives the NPU.

It exists as a separate app so that the chat client inherits none of the
exploit code, the kernel module, or the SELinux policy patching.

## What it does

1. starts the [o3root](https://github.com/zsw125/o3root) payload through
   Shizuku in shell mode (`uid=2000`, `u:r:shell:s0`)
2. the payload triggers CVE-2026-98260 and loads a KernelSU LKM at runtime
3. a root daemon (`@o3p_rootd`) accepts uid 0/2000 and runs commands
4. with uid 0, it patches SELinux with six targeted rules (Enforcing stays on)
5. runs the NPU probes and the `o3npu` driver self-test

Root and the policy patch live in memory only — a reboot clears both.

## Build

```bash
export ANDROID_BUILD_TOOLS=/path/to/build-tools
export ANDROID_JAR=/path/to/platforms/android-XX/android.jar
scripts/fetch-ksud.sh /path/to/KernelSU-manager.apk   # ksud is GPL, fetched not shipped
apps/o3root/build.sh                                   # -> apps/o3root/o3root.apk
```

## Run

Prerequisites: Shizuku installed and running in **shell** mode (wireless
debugging), and this app granted the Shizuku permission.

```bash
adb install -r apps/o3root/o3root.apk
adb shell am start -n dev.j4ger.o3root/.MainActivity
# or unattended:
adb shell am start -n dev.j4ger.o3root/.MainActivity --ez autorun true
```

Three buttons: **获取 root**, **放行 NPU + 探针**, **显示状态**.

## Risk

It modifies kernel memory, loads an unsigned module and taints the kernel.
It can crash or soft-brick the device and may void the warranty. Run it only on
hardware you own. The exploit chain is
[zsw125/o3root](https://github.com/zsw125/o3root); see [NOTICE](../../NOTICE).

Details: [docs/ROOT-CHAIN.md](../../docs/ROOT-CHAIN.md).

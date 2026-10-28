# o3-experiments

Reverse engineering and tooling for the NPU in the **Xiaomi 18 Fold** (`lhasa`,
XRing O3, VeriSilicon VIP9000 MP6). Two Android apps, plus the drivers and
notes needed to reproduce them.

> Research on hardware we own. Nothing here is production-grade.

## The two apps

| App | Package | What it does |
|---|---|---|
| **O3 Root** | `dev.j4ger.o3root` | Temporary root on a stock, locked, verified-boot device using CVE-2026-98260, then loads a KernelSU LKM. Also carries the NPU probes and the SELinux policy patcher. |
| **O3 Chat** | `dev.j4ger.o3chat` | Local VLM chat on the phone's own model, through Xiaomi's AI service. No root, no exploit code. |

Planned: agent-style tool calling on top of the chat app.

## Status

| Piece | State |
|---|---|
| Temporary root (no bootloader unlock, reboot clears it) | **works**, verified on `lhasa` / `OS4.0.26.0.XPNCNXM` |
| NPU access from the `shell` domain with six targeted SELinux rules | **works**, `Enforcing` throughout |
| `o3npu` driver (mida device, memory, subgraph) | **works** |
| llama.cpp ggml backend | plumbing works (llama.cpp loads it and offloads `MUL_MAT`); **numerics still wrong** |
| NPU matmul execution | not yet |

## Layout

```
apps/o3root/       privileged app: payload, ksud, probes, policy patcher
apps/o3chat/       chat app
tools/npu/         o3npu driver layer (mida / xnnl)
tools/ggml-o3npu/  llama.cpp backend, loaded as a dynamic backend
tools/probe/       standalone probes (mida, xnnl, OpenVX, device access)
tools/xnlm/        container parsers for the vendor model format
tools/xnnl/        API harvesting helpers
docs/              architecture, NPU stack, root chain, formats, ggml notes
archive/           raw research logs, kept for reference
```

## Building

Two things are needed: an Android SDK build-tools directory and the Shizuku
client jars (already in `tools/shizuku/`).

```bash
export ANDROID_BUILD_TOOLS=/path/to/build-tools    # must contain aapt2, d8, apksigner, zipalign
export ANDROID_JAR=/path/to/platforms/android-XX/android.jar

scripts/fetch-ksud.sh /path/to/KernelSU-manager.apk   # ksud is GPL, fetched not shipped
apps/o3root/build.sh                                  # -> apps/o3root/o3root.apk
apps/o3chat/build.sh                                  # -> apps/o3chat/o3chat.apk
```

Install and run:

```bash
adb install -r apps/o3root/o3root.apk apps/o3chat/o3chat.apk
# start Shizuku in shell mode first (wireless debugging / adb), then:
adb shell am start -n dev.j4ger.o3root/.MainActivity --ez autorun true
```

The root app needs Shizuku running in **shell** mode (`uid=2000`,
`u:r:shell:s0`). Root and the SELinux patch live in memory only: a reboot
clears both.

## Attribution

**The temporary root chain is [`zsw125/o3root`](https://github.com/zsw125/o3root)**
(MIT). The payload in `apps/o3root/jniLibs/arm64-v8a/libo3p.so` is their v3.0
build, unmodified; the CVE-2026-98260 exploit and the KernelSU runtime loader
are entirely their work. This project drives it, packages it, and adds the NPU
side. See [NOTICE](NOTICE) for KernelSU, Shizuku and llama.cpp.

## Security and disclosure

See [docs/SECURITY.md](docs/SECURITY.md) for scope, the responsible-disclosure stance and the risks of the root path.

## Legal

Only run this on hardware you own. It modifies kernel memory, loads an unsigned
module, taints the kernel, and can crash or soft-brick a device. Vendor
libraries and model files are not redistributed here.

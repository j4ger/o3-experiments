# Architecture

## Device

| | |
|---|---|
| Model | Xiaomi 18 Fold, codename `lhasa` (2608BPX34C) |
| SoC | XRing O3 (`xring_o3_asic`), NPU: VeriSilicon VIP9000 series, MP6 |
| OS | HyperOS 4 / Android 17, `OS4.0.26.0.XPNCNXM` |
| Kernel | `6.18.21-android17-5-g805aa6adea19-abogki565292102-4k`, 10 cores, 4 KB pages |
| SELinux | Enforcing; bootloader locked, verified boot green |

The NPU is identified by
`/vendor/etc/npu/compiler/VIP9000NANOQI_MP6_PID0X1000002B.config`
(`chipModel 0x9000`, `productID 0x0509000b`).

## Software stack

```
apps/o3chat  ── AI_ABILITY ──►  com.xiaomi.aiservice (priv_app_36) ──► minetd ──► NPU
    │                                                                  (official path)
    └── deep link ──► apps/o3root ── Shizuku ──► CVE-2026-98260 ──► uid 0
                                                        │
                                                        └── ksud sepolicy patch (6 rules)

tools/npu (o3npu) ──► libxnnl  (matmul / RMSNorm / quantize graph builders)
                        └──► mida  (subgraph, instantiate, stream launch, memory)
                               └──► libxnumd (UMD)
                                      └──► /dev/hpc-cdev

tools/ggml-o3npu  ──► llama.cpp dynamic backend ──► o3npu ──► NPU
```

There is **no NNAPI driver** on this device: the AOSP NNAPI runtime is present
but `lshal` reports zero neuralnetworks services. Xiaomi's own stack
(`minet` AIDL HAL + `minetd` + `libxnumd`/`mida`) is the only path to the NPU.

## Why root is needed

`/dev/hpc-cdev` is `npu_device` and only `system_server`, platform/privileged
apps, `minetd` and a few HALs may open it. `shell` and `untrusted_app` may not.
The minet HAL is restricted to `system_app`/`system_server`. So running our own
code on the NPU requires either a privileged context or temporary root.

`apps/o3root` obtains uid 0 without unlocking the bootloader, then keeps SELinux
in Enforcing and adds exactly six allow rules (see [NPU-STACK.md](NPU-STACK.md)).

## The apps

- **O3 Root** — privileged tool. Drives the o3root payload through Shizuku
  (shell mode), stages `ksud`, patches SELinux, runs the NPU probes. All
  privileged code lives here so the chat app inherits none of it.
- **O3 Chat** — the local VLM chat client. Talks to `com.xiaomi.aiservice`
  through `AI_ABILITY`; its `Root` button is a plain deep link into O3 Root.

## Reading order

1. [NPU-STACK.md](NPU-STACK.md) — the access matrix, the rules, the driver.
2. [ROOT-CHAIN.md](ROOT-CHAIN.md) — how the root is obtained and wired.
3. [GGML-BACKEND.md](GGML-BACKEND.md) — the llama.cpp integration state.
4. [MODEL-FORMATS.md](MODEL-FORMATS.md) — which model formats make sense.
5. [FORMAT-XNLM.md](FORMAT-XNLM.md) — the vendor container, decoded.

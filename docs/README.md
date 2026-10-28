# Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | device facts, the NPU software stack, both apps, where to start |
| [NPU-STACK.md](NPU-STACK.md) | who may touch the NPU (SELinux), the exact rules that work, the `o3npu` driver, probe results |
| [ROOT-CHAIN.md](ROOT-CHAIN.md) | how the temporary root works, how it is wired into the app, risks and cleanup |
| [KERNELSU-NOTES.md](KERNELSU-NOTES.md) | why a uid-0 process still cannot write, and how KernelSU's own su differs |
| [GGML-BACKEND.md](GGML-BACKEND.md) | the llama.cpp backend, CPU baseline, offline vs on-device split |
| [MODEL-FORMATS.md](MODEL-FORMATS.md) | ONNX vs GGUF vs vendor formats, and which path fits this NPU |
| [FORMAT-XNLM.md](FORMAT-XNLM.md) | the vendor model container, decoded |
| [SECURITY.md](SECURITY.md) | scope, disclosure stance, risks |

Raw research logs live in [`../archive/`](../archive/).

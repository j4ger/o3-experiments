# NPU 栈：访问矩阵、SELinux 规则与驱动

本文合并了设备的访问控制事实（第一段）与我们实际打通 NPU 所需的最小规则集和驱动层（第二段）。

---

## 第一部分：谁能访问 NPU（实测）

设备：Xiaomi 18 Fold `lhasa`，`xring_o3_asic`，HyperOS `OS4.0.26.0.XPNCNXM`，SELinux **Enforcing**。
证据来源：`/sys/fs/selinux/policy`（world-readable，已拉取 `artifacts/device/sepolicy`）+
`sesearch`/`seinfo` 分析 + `ls -Z` 实测。

## 1. NPU 是什么

`/vendor/etc/npu/compiler/VIP9000NANOQI_MP6_PID0X1000002B.config`：

```
chipModel = 0x00009000;  chipRevision = 0x00009200;
productID = 0x0509000b;  customerID   = 0x1000002b;
```

→ **VeriSilicon VIP9000 系列（Vivante），MP6 六核**。编译产物为 `.bell`
（FlatBuffers，文件标识 `OMDL`，编译器版本 `6.0-3.0`），与本项目里的 `vit.plan`
（标识 `YMCA`，`6.2-3.2`）同属一套 FlatBuffers 容器族。

## 2. 没有 NNAPI 驱动

- 框架侧存在：`/system/apex/com.android.neuralnetworks.capex`（AOSP 标准 NNAPI）。
- `lshal` 里 `neuralnetworks` 服务数 = **0**；`/vendor/bin/hw`、`/vendor/lib64/hw` 无 NNAPI 实现；
  vintf 清单无相关条目。
- 只有 `persist.device_config.nnapi_native.current_feature_level = 7`（AOSP 参考实现配置）。

→ **NNAPI 到不了这颗 NPU**。VeriSilicon 通常以 NNAPI HAL 交付，小米这里改用了私有栈。

## 3. 存在两套 Vivante 接口

| 库 | 用途 | 公共库白名单 |
|---|---|---|
| `libOpenCL.so` + `libVSC.so`/`libCLC.so`/`libSPIRV_viv.so` | OpenCL（通常是 **GPU**） | **在 `/vendor/etc/public.libraries.txt`** |
| `libOpenVX.so`(2784 个 `vx*` 导出) + `libovxlib.so`(`vxGenerateNBG`/`vxVerifyGraph`/`vxProcessGraph`) + `libNBGlinker.so` | OpenVX（**NPU** 的正规接口） | 不在白名单 |

`same_process_hal_file` 类型：`libOpenVX.so`、`libxnumd.so` 均为此类。

## 4. 谁能真正打开 NPU（SELinux 实测）

`/dev/hpc-cdev`、`/dev/hpc-heap` = `u:object_r:npu_device:s0`；
`/dev/hpc-mdr` = `u:object_r:npu_service_device:s0`。

**能 open/write `npu_device` 的域**：

```
system_server, platform_app, platform_app_36, priv_app_36, minetd,
hal_camera_default, hal_audio_default, hal_c2_x, hal_seaaudio_default,
vendor_xring_vpp_service, theme
```

**不能**：`untrusted_app`（普通应用）、`shell`。
`npu_service_device` 仅 `hal_npu` 可 open。

## 5. 小米自己的 AI 服务

```
com.xiaomi.aiservice            (u0_a106)   应用进程
minetd                          (root)      AI 守护进程
vendor.xiaomi.aidl.minet@1.0-service (root) AIDL HAL
```

SELinux 属性 `hal_minetaidlservice_client` 展开只有 **`system_app`、`system_server`**：

```
Type Attributes: 1
   attribute hal_minetaidlservice_client;
        system_app
        system_server
```

→ 普通应用、shell、platform_app **都调不到 minet**。

## 6. 结论

| 路径 | 无 root 可行性 |
|---|---|
| NNAPI | ✗ 没有驱动 |
| OpenCL（GPU） | 可加载，但**不是 NPU** |
| OpenVX/NBG 直连 NPU | ✗ 需要 open `/dev/hpc-cdev`，仅 system/platform/priv/minetd |
| minet AIDL HAL | ✗ 仅 system_app / system_server |
| **经官方引擎做模型注入** | ✓ 唯一无需特权代码的路径 |

即：**要跑我们自己的代码（自定义算子/OpenVX 图），必须 root**；不 root 只能走
"把权重换成我们自己的" —— 而这条路的格式问题已经全部解完（见 FORMAT-xnlm.md §13–14）。


---

## 第二部分：从 shell 域驱动 NPU（已验证）

**结论：可以。** 在 `u:r:shell:s0` 域、`uid=0`、**SELinux 保持 Enforcing** 的前提下，
加载 `libxnumd.so` 并 `midaDeviceGet()` 成功，能分配/释放 NPU 内存。不需要 root 之外的
任何特权域，也不需要 permissive。

## 1. 需要的规则（全部实测）

ksud 语法：`allow <源> <目标> <类> { 权限 }` —— **空格分隔、无冒号**。

```
allow shell npu_device chr_file { append getattr ioctl lock map open read watch watch_reads write }
allow shell dmabuf_system_heap_device chr_file { getattr ioctl lock map open read watch watch_reads }
allow shell timestamp_device chr_file { open read ioctl map getattr }
allow shell xr_dmabuf_helper chr_file { open read ioctl map getattr }
allow shell vendor_npu_prop file { open read getattr map }
allow shell shell capability { dac_override dac_read_search fowner fsetid chown }
```

每条都有明确的证据来源：

| 规则 | 为什么 |
|---|---|
| `npu_device chr_file` | `/dev/hpc-cdev`、`/dev/hpc-heap`、`/dev/hpc-rpmsg` 都是这个类型 |
| `dmabuf_system_heap_device` | `libxnumd` 从 `/dev/dma_heap/xring_cpa`、`/dev/dma_heap/system` 取 NPU 内存 |
| `timestamp_device` | AVC：`avc: denied { read } ... tcontext=u:object_r:timestamp_device:s0` |
| `xr_dmabuf_helper` | AVC：`avc: denied { read } ... tcontext=u:object_r:xr_dmabuf_helper:s0` |
| `vendor_npu_prop file` | AVC：`avc: denied { read } ... tcontext=u:object_r:vendor_npu_prop:s0` |
| `capability { dac_override ... }` | 原厂策略给 `shell` 域的 capability 是 **0 条**，uid 0 也无法 `dac_override`（见 `KERNELSU-NOTES.md`） |

## 2. 实测结果

```
dlopen(/vendor/lib64/libxnumd.so) -> ok
dlsym: get release usage -> ok
sizeof(midaDeviceConfig)=28 (0x1c)
midaDeviceGet(type=0) -> err=0 dev=0xb40000773a406540
midaGetNPUMemoryUsage -> err=0
midaMemoryAlloc(4096) -> err=0
  midaMemorySize -> 4096
  usage+0x08 = 4096, usage+0x10 = 4096
midaMemoryFree -> err=0
midaDeviceRelease -> err=0
Enforcing
```

`midaDeviceGet` 成功意味着 `DeviceImpl::Init` 走完：打开 `/dev/hpc-cdev`、
引导 NPU 固件（库内字符串 `hpc-cdev boot success!!!!!!!`）、建立与 npu 侧的通道。

## 3. 还没打通的一点：CPU 侧指针

`midaMemoryAlloc` 分配的是 **NPU 侧内存**：`midaMemoryGetPtr()` 和
`midaMemoryGetDeviceVa()` 都返回 0。CPU 可访问的地址需要再走
`midaMemoryMap`（签名里有 `int fd`）/`midaMemoryGetFd` / `midaMemoryAddMap`，
或者用 `midaMemoryCacheStart/End` 做缓存同步。这是下一步。

## 4. OpenVX 是死路（重要）

`libOpenVX.so` 在这台机器上是 **Vivante GPU（GC）的 OpenVX**，不是 NPU：

```
vxCreateContext() -> SIGSEGV
#00 libGAL.so (gcoHAL_DeviceControl_NoKernel+1368)
#01 libGAL.so (_QueryHardwareIdAndOptions+216)
#02 libGAL.so (gcoHARDWARE_ConstructEx+3112)
#04 libGAL.so (gcoVX_Construct+460)
#06 libOpenVX.so (vxCreateContext+328)
```

`gcoVX_*` 是 GPU 路径。NPU 的正规门是 `libxnumd`（mida），或者更上层的
`libxnnl`/`.bell` 模型执行栈。

## 5. 复现

```bash
# 1) 提权（o3root app 按钮 1，或 --ez autorun true）
# 2) 打规则（按钮 2 会自动做，或手动）
/data/local/tmp/ksud sepolicy patch 'allow shell npu_device chr_file { ... }'
...
# 3) 跑探针
/data/local/tmp/o3p_chat.bin --shell '/data/local/tmp/probe_mida'
```

源码：`tools/probe/probe_mida.c`；规则常量：`o3root/src/.../RootChain.java` 的 `RULES`。

## 6. 成体系的驱动层：o3npu

把整条栈封装成 `tools/npu/o3npu.{h,c}`，调用方不再需要关心 vendor 分层：

```
你的代码
  → libxnnl   (算子图构建：matmul / RMSNorm / quantize，带 groupwise 量化参数)
    → mida    (subgraph / instantiate / stream launch / memory)
      → libxnumd (UMD)
        → /dev/hpc-cdev
```

### API

```c
o3npu_t *n = o3npu_open();                       // 打开设备 + 建 xnnl handle
void *m = o3npu_alloc(n, 4096, /*cpu*/1, &sz);   // flag=1 → CPU 可映射
void *p = o3npu_cpu_ptr(n, m);                   // 真实 CPU 指针
uint64_t va = o3npu_dev_va(n, m);                // NPU 侧 VA（host 内存为 0）
int fd = o3npu_mem_fd(n, m);                     // dmabuf fd
o3npu_free(n, m);
o3npu_close(n);
```

### 关键发现：两种内存

| `midaMemoryAlloc` 第 4 参 | `midaMemoryGetPtr` | 用途 |
|---|---|---|
| `1` | 返回真实 CPU 指针 | 上传权重、读写张量 |
| `0` | 0 | NPU 侧分配 |

### 实测（自检程序）

```
o3npu_open -> ok  device=0x… xnnl=0x…
alloc(cpu_visible) -> size=4096  cpu_ptr=0x7d6e407000  fd=28079  write/readback OK
alloc(npu_only)    -> ok
subgraph_create    -> 0
RESULT ok
```

### 下一步：第一个算子

执行链已确认（`midaSubgraphInstantiate` → `midaStreamLaunchSubgraph` →
`midaStreamSynchronize`），但空子图无法 instantiate（返回 1，符合预期）。
需要先用 `xnnlSetTensorDescriptor` + `xnnlRMSNormalizationCreateGraph`
（或 `xnnlMatmulCreateGraph`）往子图里放一个真实算子，再做数值对拍。


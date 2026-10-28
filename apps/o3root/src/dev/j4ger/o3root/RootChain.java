package dev.j4ger.o3root;

import android.content.Context;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.lang.reflect.Method;

import rikka.shizuku.Shizuku;

/**
 * In-app integration of the o3root chain (github.com/zsw125/o3root, MIT).
 *
 * The payload is a self-contained executable ELF shipped as
 * {@code lib/arm64-v8a/libo3p.so}; it must run in the {@code shell} SELinux
 * domain with uid 2000, which is what Shizuku (started in shell mode) provides.
 * On success the payload swaps creds to uid 0 (staying in the shell domain),
 * loads a KernelSU LKM and leaves a root daemon on the abstract socket
 * {@code @o3p_rootd}.  Root is per-boot only.
 *
 * Deliberately no Shizuku *root* mode: the chain needs shell mode.
 */
public final class RootChain {
    private static final String TAG = "O3RootChat";
    private static final String BIN = "/data/local/tmp/o3p_chat.bin";

    /** Payload argument vector, identical to o3root v3.0. */
    private static final String ARGS = "10000 14000 300 1 6 1 8000 800";

    public interface Sink {
        void onLine(String line);

        void onExit(int code);
    }

    private RootChain() {
    }

    public static String payloadPath(Context ctx) {
        return ctx.getApplicationInfo().nativeLibraryDir + "/libo3p.so";
    }

    /** Native OpenVX/OpenCL probe bundled as lib/arm64-v8a/libprobex.so. */
    public static String probePath(Context ctx) {
        return ctx.getApplicationInfo().nativeLibraryDir + "/libprobex.so";
    }

    /** Stage the probe into /data/local/tmp (uid 2000 can write there). */
    public static String stageProbe(Context ctx) {
        String src = probePath(ctx);
        return shell("cp '" + src + "' /data/local/tmp/o3p_probe"
                + " && chmod 755 /data/local/tmp/o3p_probe"
                + " && ls -l /data/local/tmp/o3p_probe");
    }

    /**
     * The single SELinux rule that lets the shell domain drive the NPU.
     * Derived from the device policy: every NPU client domain (hal_npu,
     * priv_app_36, platform_app, minetd, ...) holds exactly
     * {@code npu_device:chr_file { append getattr ioctl lock map open read watch
     * watch_reads write }} and there is no allowxperm on npu_device, so no
     * ioctl whitelist is needed.
     */
    /**
     * The exact set of policy statements needed to drive the NPU from the shell
     * domain. Derived from the device policy and verified on-device: with these
     * applied, libxnumd midaDeviceGet() returns 0 and NPU memory can be allocated,
     * all while SELinux stays Enforcing.
     *
     * ksud syntax: "allow <src> <tgt> <class> { perms }" - space separated, no
     * colon. A colon (npu_device:chr_file) fails to parse.
     */
    public static final String[] RULES = {
            // the NPU compute/heap/rpmsg character devices
            "allow shell npu_device chr_file { append getattr ioctl lock map open read watch watch_reads write }",
            // NPU memory comes from the CMA/dma heaps
            "allow shell dmabuf_system_heap_device chr_file { getattr ioctl lock map open read watch watch_reads }",
            // libxnumd timestamps work and maps dma-bufs too
            "allow shell timestamp_device chr_file { open read ioctl map getattr }",
            "allow shell xr_dmabuf_helper chr_file { open read ioctl map getattr }",
            // and reads vendor NPU properties
            "allow shell vendor_npu_prop file { open read getattr map }",
            // the stock policy gives shell no capabilities at all, so uid 0 still
            // cannot dac_override: no chmod, no O_RDWR on the 0444 npu node
            "allow shell shell capability { dac_override dac_read_search fowner fsetid chown }",
    };

    private static final String KSUD = "/data/local/tmp/ksud";

    /** KernelSU userspace CLI, bundled as lib/arm64-v8a/libksud.so. */
    public static String ksudPath(Context ctx) {
        return ctx.getApplicationInfo().nativeLibraryDir + "/libksud.so";
    }

    /** Stage probe and ksud into /data/local/tmp (uid 2000 can write there). */
    public static String prepare(Context ctx) {
        String a = shell("cp '" + probePath(ctx) + "' /data/local/tmp/o3p_probe && chmod 755 /data/local/tmp/o3p_probe");
        String b = shell("cp '" + ksudPath(ctx) + "' " + KSUD + " && chmod 755 " + KSUD);
        String c = shell("ls -l /data/local/tmp/o3p_probe " + KSUD + " 2>&1");
        return a + "\n" + b + "\n" + c;
    }

    /**
     * Grant the shell domain access to the NPU through KernelSU's own policy
     * patcher, which keeps the device in Enforcing. The patch lives in memory,
     * so it disappears on reboot together with the temporary root.
     */
    public static String patchSelinux() {
        StringBuilder sb = new StringBuilder();
        sb.append("test -x ").append(KSUD).append(" || { echo NO_KSUD; exit 2; }; ");
        for (String r : RULES) {
            sb.append(KSUD).append(" sepolicy patch '").append(r).append("' 2>&1; ");
        }
        sb.append("echo -n 'open_test='; (exec 3<>/dev/hpc-cdev) 2>&1 && echo OK || echo DENIED; ");
        sb.append("getenforce");
        return rootShell(sb.toString());
    }

    /**
     * Run the NPU probe as root. Preferred path: patch one SELinux rule with
     * ksud and stay in Enforcing. Fallback: switch to permissive for the
     * duration of the call (needs CAP_MAC_ADMIN, which the fake cred provides).
     */
    public static String runNpuProbe() {
        String patched = patchSelinux();
        // The mida probe is the real NPU door; the OpenVX one only proves that
        // libOpenVX.so is the Vivante GPU path (it faults inside libGAL).
        String cmd = "[ -x /data/local/tmp/probe_mida ] && /data/local/tmp/probe_mida 2>&1; "
                + "[ -x /data/local/tmp/o3p_probe ] && /data/local/tmp/o3p_probe 2>&1; "
                + "echo PROBE_DONE";
        if (patched.contains("open_test=OK")) {
            return patched + "\n" + rootShell(cmd);
        }
        return patched + "\nfallback: setenforce\n"
                + rootShell("setenforce 0 2>&1; " + cmd + "; setenforce 1 2>&1;"
                + " echo \"getenforce=$(getenforce)\"");
    }

    public static boolean shizukuAlive() {
        try {
            return Shizuku.pingBinder();
        } catch (Throwable t) {
            return false;
        }
    }

    public static boolean hasPermission() {
        try {
            return Shizuku.checkSelfPermission() == 0;
        } catch (Throwable t) {
            return false;
        }
    }

    public static void requestPermission(int requestCode) {
        try {
            Shizuku.requestPermission(requestCode);
        } catch (Throwable t) {
            Log.w(TAG, "requestPermission: " + t);
        }
    }

    /** Run a command through Shizuku's shell process and return combined output. */
    public static String shell(String cmd) {
        try {
            Process p = newProcess(new String[]{"/system/bin/sh", "-c", cmd},
                    new String[]{}, "/data/local/tmp");
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = p.getInputStream().read(buf)) > 0) {
                bos.write(buf, 0, n);
            }
            p.waitFor();
            return new String(bos.toByteArray(), "UTF-8");
        } catch (Throwable t) {
            Throwable c = t;
            while (c.getCause() != null) {
                c = c.getCause();
            }
            return "ERR " + t.getClass().getSimpleName() + ": " + c;
        }
    }

    /**
     * Stage and execute the exploit payload in the shell domain.
     * Everything runs on an own thread; progress is reported through {@code sink}.
     */
    public static Thread runPayload(Context ctx, Sink sink, String fork) {
        Thread t = new Thread(() -> {
            int exit = -1;
            try {
                String src = payloadPath(ctx);
                String env = fork == null ? "" : ("export O3P_KSU_FORK=" + fork + "; ");
                String cmd = env
                        + "cp '" + src + "' " + BIN
                        + " && chmod 755 " + BIN
                        + " && cd /data/local/tmp"
                        + " && " + BIN + " " + ARGS;
                sink.onLine("payload=" + src + " fork=" + (fork == null ? "auto" : fork));
                Process p = newProcess(new String[]{"/system/bin/sh", "-c", cmd},
                        new String[]{}, "/data/local/tmp");
                InputStream in = p.getInputStream();
                BufferedReaderPump pump = new BufferedReaderPump(in, sink);
                pump.start();
                exit = p.waitFor();
                pump.join(3000);
            } catch (Throwable th) {
                sink.onLine("RUNNER " + th);
            } finally {
                sink.onExit(exit);
            }
        }, "o3root-payload");
        t.start();
        return t;
    }

    /** Run a command as root through the payload's own client/daemon. */
    public static String rootShell(String cmd) {
        String quoted = cmd.replace("'", "'\\''");
        return shell(BIN + " --shell '" + quoted + "'");
    }

    public static boolean rooted() {
        String out = rootShell("id");
        Log.i(TAG, "rooted probe: " + (out == null ? "null" : out.trim()));
        return out != null && out.contains("uid=0");
    }

    private static Process newProcess(String[] cmd, String[] env, String dir) throws Exception {
        Method m = Shizuku.class.getDeclaredMethod("newProcess",
                String[].class, String[].class, String.class);
        m.setAccessible(true);
        Object o = m.invoke(null, cmd, env, dir);
        return (Process) o;
    }

    private static final class BufferedReaderPump extends Thread {
        private final InputStream in;
        private final Sink sink;

        BufferedReaderPump(InputStream in, Sink sink) {
            this.in = in;
            this.sink = sink;
            setDaemon(true);
        }

        @Override
        public void run() {
            try {
                byte[] buf = new byte[4096];
                StringBuilder line = new StringBuilder();
                int n;
                while ((n = in.read(buf)) > 0) {
                    String chunk = new String(buf, 0, n, "UTF-8");
                    for (int i = 0; i < chunk.length(); i++) {
                        char c = chunk.charAt(i);
                        if (c == '\n') {
                            String s = line.toString();
                            line.setLength(0);
                            if (!s.isEmpty()) {
                                sink.onLine(s);
                            }
                        } else {
                            line.append(c);
                        }
                    }
                }
                if (line.length() > 0) {
                    sink.onLine(line.toString());
                }
            } catch (Throwable t) {
                sink.onLine("PUMP " + t);
            }
        }
    }
}
package dev.j4ger.o3chat;

import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.lang.reflect.Method;

import rikka.shizuku.Shizuku;

/**
 * Plain Shizuku shell access (uid 2000, u:r:shell:s0) — no root, no payload.
 *
 * Everything this class runs has exactly adb-shell rights: reachable:
 *   - am/pm/dumpsys, settings, appops
 *   - /sdcard/Android/data/<other app>/... (incl. aiservice's download staging)
 * not reachable:
 *   - another app's private data dirs (verified: Permission denied)
 *
 * The staging swap path only ever needs the first group.
 */
public final class Shell {
    private static final String TAG = "TaiyiShell";

    public interface Sink {
        void onLine(String line);

        void onExit(int code);
    }

    private Shell() {
    }

    public static boolean available() {
        try {
            return Shizuku.pingBinder() && Shizuku.checkSelfPermission() == 0;
        } catch (Throwable t) {
            return false;
        }
    }

    public static String state() {
        try {
            return "binder=" + Shizuku.pingBinder() + " perm=" + Shizuku.checkSelfPermission();
        } catch (Throwable t) {
            return "unavailable: " + t;
        }
    }

    /** Run a shell command synchronously and return its combined output. */
    public static String exec(String cmd) {
        try {
            Process p = newProcess(new String[]{"/system/bin/sh", "-c", cmd});
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = p.getInputStream().read(buf)) > 0) {
                bos.write(buf, 0, n);
            }
            p.waitFor();
            String out = new String(bos.toByteArray(), "UTF-8");
            Log.i(TAG, "exec(" + brief(cmd) + ") -> " + brief(out));
            return out;
        } catch (Throwable t) {
            Log.w(TAG, "exec failed", t);
            return "ERR " + t;
        }
    }

    /** Start a long-running shell command; output is streamed to the sink. */
    public static Process execAsync(String cmd, Sink sink) throws Exception {
        Log.i(TAG, "execAsync(" + brief(cmd) + ")");
        Process p = newProcess(new String[]{"/system/bin/sh", "-c", cmd});
        final InputStream in = p.getInputStream();
        Thread pump = new Thread(() -> {
            StringBuilder line = new StringBuilder();
            try {
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) > 0) {
                    String chunk = new String(buf, 0, n, "UTF-8");
                    for (int i = 0; i < chunk.length(); i++) {
                        char c = chunk.charAt(i);
                        if (c == '\n') {
                            String s = line.toString();
                            line.setLength(0);
                            if (sink != null && !s.isEmpty()) {
                                sink.onLine(s);
                            }
                        } else {
                            line.append(c);
                        }
                    }
                }
                if (sink != null && line.length() > 0) {
                    sink.onLine(line.toString());
                }
            } catch (Throwable t) {
                if (sink != null) {
                    sink.onLine("PUMP " + t);
                }
            }
        }, "shell-pump");
        pump.setDaemon(true);
        pump.start();
        if (sink != null) {
            new Thread(() -> {
                int code = -1;
                try {
                    code = p.waitFor();
                } catch (Throwable ignored) {
                }
                sink.onExit(code);
            }, "shell-wait").start();
        }
        return p;
    }

    private static Process newProcess(String[] cmd) throws Exception {
        Method m = Shizuku.class.getDeclaredMethod("newProcess",
                String[].class, String[].class, String.class);
        m.setAccessible(true);
        Object o = m.invoke(null, cmd, new String[]{}, "/data/local/tmp");
        return (Process) o;
    }

    /** Write a shell script to a shell-writable path using a heredoc. */
    public static String writeScript(String path, String script) {
        String cmd = "cat > '" + path + "' <<'OSTAGE_EOF'\n" + script + "\nOSTAGE_EOF\n"
                + "chmod 755 '" + path + "' && echo SCRIPT_OK " + path;
        return exec(cmd);
    }

    private static String brief(String s) {
        if (s == null) {
            return "null";
        }
        String one = s.replace('\n', ' ');
        return one.length() > 160 ? one.substring(0, 160) + "…" : one;
    }
}
package dev.j4ger.o3chat;

import android.content.Context;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Locale;

/**
 * Resource-staging helpers for the no-root path.
 *
 * AIService downloads every resource package into
 *   /sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi/<type>/<id>/<version>/<fileName>
 * (v1.c.f8500w -> n2.a(26), download/u.java).  That tree is readable AND
 * writable by the adb-shell identity (uid 2000), which is what Shizuku gives us
 * — so no root is needed to touch it.  The resource installer then copies that
 * path and, for PLUGIN/MODEL packages (ResType.isZip() == true), only verifies
 * the unzipped files against the package's own taiyi_hash.json, i.e. a package
 * we supply verifies itself.
 *
 * The one thing shell cannot do is fabricate the private Room "download record"
 * that marks a resource as already downloaded, so the window has to be caught
 * while a real download runs: keep our payload at the path (see
 * assets/staging_swap.sh) until the installer has copied it.
 */
public final class StagingSwap {
    private static final String TAG = "TaiyiStaging";

    /** staging root AIService downloads into (shell-writable). */
    public static final String ROOT =
            "/sdcard/Android/data/com.xiaomi.aiservice/files/download/taiyi";

    /** where the swap script is unpacked for the shell to run. */
    public static final String SCRIPT = "/data/local/tmp/staging_swap.sh";

    /** default payload path, shell-visible (push with adb or copy from app dir). */
    public static final String DEFAULT_PAYLOAD = "/data/local/tmp/repl.taiyi";

    private StagingSwap() {
    }

    /** type dir names come from ResType.dirName() = lowercase enum name. */
    public static String typeDir(String type) {
        return type == null ? "" : type.toLowerCase(Locale.ROOT);
    }

    public static String targetFor(String type, String id, long version, String fileName) {
        return ROOT + "/" + typeDir(type) + "/" + id + "/" + version + "/" + fileName;
    }

    /** read the swap script that ships in the APK assets */
    public static String readScript(Context ctx) {
        try (InputStream in = ctx.getAssets().open("staging_swap.sh")) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) > 0) {
                bos.write(buf, 0, n);
            }
            return new String(bos.toByteArray(), "UTF-8");
        } catch (Throwable t) {
            Log.e(TAG, "readScript failed", t);
            return null;
        }
    }

    public static boolean installScript(Context ctx) {
        String script = readScript(ctx);
        if (script == null) {
            return false;
        }
        String out = Shell.writeScript(SCRIPT, script);
        boolean ok = out != null && out.contains("SCRIPT_OK");
        Log.i(TAG, "installScript ok=" + ok + " out=" + out);
        return ok;
    }

    /** list files currently present in the staging tree (shell side). */
    public static String scan() {
        String cmd = "if [ -d '" + ROOT + "' ]; then "
                + "find '" + ROOT + "' -type f 2>/dev/null | while read f; do "
                + "echo \"$(stat -c%s \"$f\") $f\"; done | sort -k2; "
                + "echo STAGING_FILES_DONE; else echo STAGING_ROOT_MISSING; fi";
        return Shell.exec(cmd);
    }

    /** make a payload visible to the shell: copy from an app-readable path. */
    public static boolean stagePayload(Context ctx, String sourcePath, String destPath) {
        String dest = destPath == null ? DEFAULT_PAYLOAD : destPath;
        if (sourcePath != null && !sourcePath.isEmpty()) {
            String out = Shell.exec("cp -f '" + sourcePath + "' '" + dest
                    + "' && chmod 644 '" + dest + "' && stat -c%s '" + dest + "'");
            Log.i(TAG, "stagePayload " + dest + " -> " + out);
            return out != null && !out.contains("ERR") && !out.trim().isEmpty();
        }
        // fall back: publish the app's own external file if it exists
        try {
            java.io.File ext = new java.io.File(ctx.getExternalFilesDir(null), "repl.taiyi");
            if (!ext.exists()) {
                Log.w(TAG, "no payload at " + ext);
                return false;
            }
            String out = Shell.exec("cp -f '" + ext.getAbsolutePath() + "' '" + dest + "' && stat -c%s '" + dest + "'");
            Log.i(TAG, "stagePayload(appdir) -> " + out);
            return true;
        } catch (Throwable t) {
            Log.w(TAG, "stagePayload failed", t);
            return false;
        }
    }

    /** write a payload the app itself holds into its external dir (shell can read it). */
    public static String publishPayload(Context ctx, byte[] bytes) {
        try {
            java.io.File ext = new java.io.File(ctx.getExternalFilesDir(null), "repl.taiyi");
            try (FileOutputStream fos = new FileOutputStream(ext)) {
                fos.write(bytes);
            }
            Log.i(TAG, "publishPayload " + ext.getAbsolutePath() + " bytes=" + bytes.length);
            return ext.getAbsolutePath();
        } catch (Throwable t) {
            Log.w(TAG, "publishPayload failed", t);
            return null;
        }
    }

    /**
     * Arm the swap: keep {@code payload} at {@code target} for {@code durationMs}.
     * Runs the same script validated by tools/staging/swap-sim.js.
     */
    public static boolean arm(String target, String payload, long expectedSize,
                             long durationMs, Shell.Sink sink) {
        String sleep = "0.02";
        String cmd = "sh " + SCRIPT + " '" + target + "' '" + payload + "' "
                + expectedSize + " " + durationMs + " " + sleep;
        try {
            Shell.execAsync(cmd, sink);
            Log.i(TAG, "ARMED target=" + target + " payload=" + payload
                    + " size=" + expectedSize + " dur=" + durationMs);
            return true;
        } catch (Throwable t) {
            Log.e(TAG, "arm failed", t);
            return false;
        }
    }

    /** one-shot summary line for logging */
    public static String describe(String target, String payload, long size, long dur) {
        return "target=" + target + " payload=" + payload + " expected=" + size + " dur=" + dur;
    }
}
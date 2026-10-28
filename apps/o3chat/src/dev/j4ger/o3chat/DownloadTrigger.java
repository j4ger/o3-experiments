package dev.j4ger.o3chat;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;

import java.util.Arrays;
import java.util.List;

/**
 * Trigger an official resource download.
 *
 * AIService's DownloadProvider only serves query/pause/cancel, but its
 * DownloadService is exported with the AI_ABILITY permission (which this app
 * declares) and accepts the same AIRequest parcelable:
 *
 *   <service android:name="com.xiaomi.aiservice.download.DownloadService"
 *            android:exported="true"
 *            android:permission="com.xiaomi.aiservice.permission.AI_ABILITY">
 *     <intent-filter><action android:name="com.xiaomi.aiservice.DOWNLOAD"/></intent-filter>
 *   </service>
 *   DownloadService reads intent.getParcelableExtra("request") (AIKeys.KEY_REQUEST).
 *
 * Starting it for an api whose resources are not installed yet makes AIService
 * fetch those packages into the shell-writable staging tree — that is the
 * window StagingSwap needs.  The same request shape carries the uid=1000 header
 * used by the rest of this research client.
 */
public final class DownloadTrigger {
    private static final String TAG = "TaiyiDownload";

    public static final String PACKAGE = "com.xiaomi.aiservice";
    public static final String ACTION = "com.xiaomi.aiservice.DOWNLOAD";
    public static final String SERVICE = "com.xiaomi.aiservice.download.DownloadService";

    private DownloadTrigger() {
    }

    /** Probe which of the listed apis still need bytes (left_download_size > 0). */
    public static List<ModelStatus.Info> probe(Context ctx, String scene, List<String> apis) {
        ModelStatus.Info info = ModelStatus.query(ctx, scene, apis);
        Log.i(TAG, "probe scene=" + scene + " apis=" + apis + " -> " + info
                + (info.rawKeys != null ? " keys=[" + info.rawKeys + "]" : ""));
        return Arrays.asList(info);
    }

    /** Start the exported download service for the given api set. */
    public static boolean start(Context ctx, String scene, List<String> apis) {
        try {
            Intent intent = new Intent(ACTION);
            intent.setComponent(new ComponentName(PACKAGE, SERVICE));
            intent.putExtra("request", ModelStatus.newRequest(ctx, scene, apis, "dltrigger"));
            if (Build.VERSION.SDK_INT >= 26) {
                ctx.startForegroundService(intent);
            } else {
                ctx.startService(intent);
            }
            Log.i(TAG, "DOWNLOAD_STARTED scene=" + scene + " apis=" + apis);
            return true;
        } catch (Throwable t) {
            Log.e(TAG, "download start failed scene=" + scene + " apis=" + apis, t);
            return false;
        }
    }
}
package dev.j4ger.o3chat;

import android.content.ContentResolver;
import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;

import com.xiaomi.taiyi.sdk.transfer.data.AIBody;
import com.xiaomi.taiyi.sdk.transfer.data.AIHeaders;
import com.xiaomi.taiyi.sdk.transfer.data.AIRequest;
import com.xiaomi.taiyi.sdk.transfer.data.AIResponse;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Read-only AIService download/install status probe.
 *
 * The authoritative answer to "is the Mimo model actually installed?" comes
 * from AIService itself: {@code DownloadProvider.call("query")} returns
 * local_size / full_size / download_size plus the config and resource JSON for
 * the requested api set.  No file system access and no root required.
 */
public final class ModelStatus {
    private static final String TAG = "ModelStatus";
    private static final String AUTHORITY = "com.xiaomi.aiservice.authority.download";

    public static final class Info {
        /** full_size: total bytes of the requested resource set (body). */
        public long fullSize = -1;
        /** left_download_size: bytes still pending; 0 == already installed. */
        public long leftDownload = -1;
        /** total_download_size: bytes moved in this session (body). */
        public long totalDownload = -1;
        public int downloadState = -1;
        public String configJson;
        public String resourceJson;
        public String rawKeys;
        public String error;

        public boolean installedFully() {
            return fullSize > 0 && leftDownload == 0;
        }

        public boolean hasAnything() {
            return fullSize > 0;
        }

        @Override
        public String toString() {
            if (error != null) {
                return "error=" + error;
            }
            return "full=" + human(fullSize)
                    + " left=" + human(leftDownload)
                    + " session=" + human(totalDownload)
                    + " state=" + downloadState;
        }
    }

    private ModelStatus() {
    }

    public static String human(long bytes) {
        if (bytes < 0) {
            return "?";
        }
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024L * 1024) {
            return String.format(java.util.Locale.US, "%.1f KB", bytes / 1024.0);
        }
        if (bytes < 1024L * 1024 * 1024) {
            return String.format(java.util.Locale.US, "%.1f MB", bytes / 1048576.0);
        }
        return String.format(java.util.Locale.US, "%.2f GB", bytes / 1073741824.0);
    }

    /** Build the AIService request shape used by download/config queries. */
    public static AIRequest newRequest(Context ctx, String scene, List<String> apis, String tag) {
        AIHeaders headers = new AIHeaders();
        headers.bundle.putString("ai_scene", scene);
        headers.bundle.putStringArrayList("ai_api_list", new ArrayList<>(apis));
        headers.bundle.putStringArrayList("ai_resource_list", new ArrayList<String>());
        headers.bundle.putBoolean("is_llm", true);
        headers.bundle.putInt("download_ui_flag", -1);
        headers.bundle.putString("ai_ability_name", "");
        headers.bundle.putString("package_name", ctx.getPackageName());
        headers.bundle.putString("app_signature", "");
        headers.bundle.putString("app_code", "");
        headers.bundle.putInt("uid", 1000);
        headers.bundle.putInt("pid", Process.myPid());
        headers.bundle.putString("process_name", ctx.getPackageName());
        headers.bundle.putInt("ai_sdk_code", 200);
        headers.bundle.putString("ai_sdk_name", tag == null ? "modelstatus" : tag);
        headers.bundle.putString("request_id", "ms-" + System.currentTimeMillis());
        return new AIRequest(headers, new AIBody());
    }

    /** scene.txt_llm is the full official set; api list narrows it. */
    public static Info query(Context ctx, String scene, List<String> apis) {
        Info info = new Info();
        try {
            AIRequest request = newRequest(ctx, scene, apis, "modelstatus");
            Bundle extras = new Bundle();
            extras.putParcelable("request", request);

            ContentResolver resolver = ctx.getContentResolver();
            Bundle result = resolver.call(
                    Uri.parse("content://" + AUTHORITY + "/call"), "query", null, extras);
            if (result == null) {
                info.error = "null bundle";
                return info;
            }
            result.setClassLoader(AIResponse.class.getClassLoader());
            AIResponse response = result.getParcelable("response", AIResponse.class);
            if (response == null) {
                info.error = "null response";
                return info;
            }
            if (response.headers != null) {
                // sizes live in the body on this build; keep the header probe
                // as a fallback for other versions.
                info.fullSize = response.headers.bundle.getLong("full_size", info.fullSize);
                info.leftDownload = response.headers.bundle.getLong("left_download_size", info.leftDownload);
            }
            if (response.body != null) {
                info.fullSize = response.body.bundle.getLong("full_size", info.fullSize);
                info.leftDownload = response.body.bundle.getLong("left_download_size", info.leftDownload);
                info.totalDownload = response.body.bundle.getLong("total_download_size", -1);
                info.downloadState = response.body.bundle.getInt("download_state", -1);
                info.configJson = response.body.bundle.getString("ai_config_json", null);
                info.resourceJson = response.body.bundle.getString("ai_resource_json", null);
                if (info.configJson == null) {
                    info.configJson = response.body.bundle.getString("ai_config_json_list", null);
                }
                if (info.resourceJson == null) {
                    info.resourceJson = response.body.bundle.getString("ai_resource_json_list", null);
                }
                StringBuilder bkeys = new StringBuilder();
                for (String k : response.body.bundle.keySet()) {
                    bkeys.append(k).append(' ');
                }
                info.rawKeys = bkeys.toString().trim();
            }
            Log.i(TAG, "query scene=" + scene + " apis=" + apis + " -> " + info);
        } catch (Throwable t) {
            info.error = String.valueOf(t);
            Log.w(TAG, "query failed", t);
        }
        return info;
    }

    public static Info mimo(Context ctx) {
        return query(ctx, "text_llm", Arrays.asList("mimo_dev"));
    }

    public static Info mimoBundle(Context ctx) {
        return query(ctx, "text_llm",
                Arrays.asList("mimo_dev", "mimo_lora_offline_xiaoai", "llm_runtime", "vlm"));
    }
}
package dev.j4ger.o3chat;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.graphics.Bitmap;
import android.os.Binder;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.Parcel;
import android.os.Process;
import android.os.RemoteException;
import android.os.SystemClock;
import android.util.Log;

import com.xiaomi.taiyi.sdk.transfer.data.AIBody;
import com.xiaomi.taiyi.sdk.transfer.data.AIHeaders;
import com.xiaomi.taiyi.sdk.transfer.data.AIRequest;

import java.util.concurrent.atomic.AtomicInteger;

/** Minimal client for the AIService local VLM service. */
public final class TaiyiClient {
    private static final String TAG = "TaiyiChat";
    private static final String SERVICE_PACKAGE = "com.xiaomi.aiservice";
    private static final String SERVICE_ACTION = "com.xiaomi.aiservice.LLM_MANAGER";
    private static final String MANAGER_DESCRIPTOR =
            "com.xiaomi.taiyi.sdk.llm.ipc.ILLMManager";
    private static final String CLIENT_DESCRIPTOR =
            "com.xiaomi.taiyi.sdk.llm.ipc.ILLMClient";
    private static final AtomicInteger SEQ = new AtomicInteger();
    private static final String MIMO_API = "mimo_dev";

    public interface Listener {
        void onReady();

        void onDelta(String text, int status, int error);

        void onError(String message);
    }

    private final Context context;
    private final Listener listener;
    private IBinder binder;
    private boolean bound;
    private boolean ready;
    private BitmapBinder lastBitmap;
    private Callback lastCallback;
    private Callback initCallback;
    private boolean mimoWarmed;
    private long mimoInitDeadline;
    private int mimoInitAttempts;
    private static final int MIMO_INIT_TIMEOUT_MS = 20000;
    private static final int MIMO_INIT_MAX_ATTEMPTS = 2;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable mimoWatchdog;

    private String pendingPrompt;
    private Bitmap pendingImage;
    private String pendingApi;
    private boolean pendingChatMode;

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            binder = service;
            bound = true;
            Log.i(TAG, "SERVICE_CONNECTED " + name);
            new Thread(() -> {
                int code = authorize(makeHeaders("vlm"), new AIBody());
                Log.i(TAG, "AUTHORIZE code=" + code);
                ready = true;
                listener.onReady();
            }, "taiyi-authorize").start();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.i(TAG, "SERVICE_DISCONNECTED " + name);
            binder = null;
            bound = false;
            ready = false;
            listener.onError("AIService 已断开");
        }
    };

    public TaiyiClient(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;
    }

    public void bind() {
        if (bound) {
            return;
        }
        Intent intent = new Intent(SERVICE_ACTION);
        intent.setPackage(SERVICE_PACKAGE);
        intent.putExtra("request", new AIRequest(makeHeaders("vlm"), new AIBody()));
        boolean ok = context.bindService(intent, connection, Context.BIND_AUTO_CREATE);
        Log.i(TAG, "BIND_RESULT " + ok);
        if (!ok) {
            listener.onError("无法绑定 AIService");
        }
    }

    public boolean isReady() {
        return ready && binder != null;
    }

    /**
     * @param prompt user prompt, may contain prior turns
     * @param image  optional image; sent through IBitmap Binder, not raw Parcelable
     */
    public boolean send(String prompt, Bitmap image, String api) {
        return send(prompt, image, api, false);
    }

    /**
     * @param chatMode experiment: use the LLMKit "postChat" method with a
     *                 ChatRequest-shaped json_source instead of "postTask"
     */
    public boolean send(String prompt, Bitmap image, String api, boolean chatMode) {
        return send(prompt, image, api, chatMode, false);
    }

    public boolean send(String prompt, Bitmap image, String api, boolean chatMode,
                        boolean initMode) {
        if (!isReady()) {
            listener.onError("本地模型还没就绪");
            return false;
        }

        // Mimo is only fully loaded by an explicit ai:method:llm_init call.
        // The plugin's lazy init path marks itself initialized without running
        // LLM_Init(base.json), so every generate() would fail with status -1.
        if (!initMode && MIMO_API.equals(api) && !mimoWarmed) {
            pendingPrompt = prompt;
            pendingImage = image;
            pendingApi = api;
            pendingChatMode = chatMode;
            return warmMimo();
        }

        return sendOnce(prompt, image, api, chatMode, initMode);
    }

    private boolean warmMimo() {
        AIHeaders headers = makeHeaders(MIMO_API);
        headers.bundle.putString("ai_method", "ai:method:llm_init");
        AIBody body = new AIBody();
        body.bundle.putString("prompt", "");
        body.bundle.putBoolean("share_memory_enable", false);
        body.bundle.putInt("task_status", 0);

        initCallback = new Callback();
        body.bundle.putBinder("llm_client", initCallback);

        mimoInitAttempts++;
        mimoInitDeadline = SystemClock.elapsedRealtime() + MIMO_INIT_TIMEOUT_MS;
        armMimoWatchdog();

        boolean sent = sendTask(headers, body);
        Log.i(TAG, "MIMO_WARMUP sent=" + sent + " attempt=" + mimoInitAttempts
                + " deadline_ms=" + MIMO_INIT_TIMEOUT_MS);
        if (!sent) {
            handler.removeCallbacks(mimoWatchdog);
            mimoInitAttempts = 0;
            pendingPrompt = null;
            pendingImage = null;
            listener.onError("Mimo 预热失败");
        }
        return sent;
    }

    /**
     * The Mimo engine only loads through an explicit {@code ai:method:llm_init}
     * task; its lazy path can leave the engine half-initialized and every
     * generate then fails with a native prefill error.  A missing completion
     * callback used to hang the flow forever, so retry once and then give up
     * with an explicit error.
     */
    private void armMimoWatchdog() {
        if (mimoWatchdog == null) {
            mimoWatchdog = () -> {
                if (mimoWarmed) {
                    return;
                }
                Log.w(TAG, "MIMO_WARMUP timeout ms=" + MIMO_INIT_TIMEOUT_MS
                        + " attempt=" + mimoInitAttempts);
                initCallback = null;
                if (mimoInitAttempts < MIMO_INIT_MAX_ATTEMPTS) {
                    Log.i(TAG, "MIMO_WARMUP retry");
                    warmMimo();
                } else {
                    Log.e(TAG, "MIMO_WARMUP giving up");
                    mimoWarmed = false;
                    mimoInitAttempts = 0;
                    pendingPrompt = null;
                    pendingImage = null;
                    listener.onError("Mimo 初始化超时");
                }
            };
        }
        handler.removeCallbacks(mimoWatchdog);
        handler.postDelayed(mimoWatchdog, MIMO_INIT_TIMEOUT_MS);
    }

    /** Resume the request that was parked while Mimo warm-up was running. */
    private void sendPendingMimo() {
        String prompt = pendingPrompt;
        Bitmap image = pendingImage;
        String api = pendingApi;
        boolean chat = pendingChatMode;
        pendingPrompt = null;
        pendingImage = null;
        if (prompt != null) {
            sendOnce(prompt, image, api, chat, false);
        }
    }

    /** Drop the warm flag so the next Mimo send runs a fresh llm_init. */
    public void resetMimoWarm() {
        handler.removeCallbacks(mimoWatchdog);
        mimoWarmed = false;
        mimoInitAttempts = 0;
        initCallback = null;
        Log.i(TAG, "MIMO_WARMUP reset");
    }

    private boolean sendOnce(String prompt, Bitmap image, String api, boolean chatMode,
                             boolean initMode) {
        // Mimo's postTask path never runs applyLoRA on this plugin build, so the
        // engine stays without a base LoRA and the native prefill fails with
        // status=-1.  The executeChat path does apply it, but only when
        // jsonSource is present -> route Mimo through chat mode.
        if (MIMO_API.equals(api) && !chatMode && !initMode) {
            return sendMimoChat(prompt, image);
        }
        AIHeaders headers = makeHeaders(api);
        AIBody body = new AIBody();
        body.bundle.putString("prompt", prompt);
        body.bundle.putBoolean("share_memory_enable", false);
        body.bundle.putInt("task_status", 0);
        body.bundle.putInt("bitmap_count", image == null ? 0 : 1);
        if (image != null) {
            lastBitmap = new BitmapBinder(image);
            body.bundle.putBinder("bitmap", lastBitmap);
            body.bundle.putBinder("bitmap_0", lastBitmap);
        } else {
            lastBitmap = null;
        }

        if (chatMode) {
            // jsonSource only crosses the Binder inside the bundle when share
            // memory is enabled (AIBody.distribute/collect are no-ops otherwise),
            // and the Mimo postChat worker requires a non-empty jsonSource to
            // reach its LoRA-apply step.
            body.bundle.putBoolean("share_memory_enable", true);
            headers.bundle.putString("ai_method", "executeChat");
            String json = buildChatJson(prompt, false);
            body.bundle.putString("json_source", json);
            Log.i(TAG, "CHAT_MODE json=" + json);
        }

        if (initMode) {
            headers.bundle.putString("ai_method", "ai:method:llm_init");
            Log.i(TAG, "INIT_MODE api=" + api);
        }

        lastCallback = new Callback();
        body.bundle.putBinder("llm_client", lastCallback);

        boolean sent = sendTask(headers, body);
        Log.i(TAG, "SENDTASK sent=" + sent + " image=" + (image != null)
                + " api=" + api + " chat=" + chatMode
                + " prompt=" + prompt.length());
        if (!sent) {
            listener.onError("发送任务失败");
        }
        return sent;
    }

    /**
     * Mimo through the executeChat path: jsonSource must be present and the
     * share-memory flag must be on for it to survive the Binder hop, otherwise
     * the plugin logs "jsonSource is empty" and no LoRA is ever applied.
     */
    private boolean sendMimoChat(String prompt, Bitmap image) {
        AIHeaders headers = makeHeaders(MIMO_API);
        headers.bundle.putString("ai_method", "executeChat");
        AIBody body = new AIBody();
        body.bundle.putString("prompt", prompt);
        body.bundle.putBoolean("share_memory_enable", true);
        body.bundle.putInt("task_status", 0);
        body.bundle.putInt("bitmap_count", image == null ? 0 : 1);
        if (image != null) {
            lastBitmap = new BitmapBinder(image);
            body.bundle.putBinder("bitmap", lastBitmap);
            body.bundle.putBinder("bitmap_0", lastBitmap);
        } else {
            lastBitmap = null;
        }
        String json = buildChatJson(prompt, false);
        body.bundle.putString("json_source", json);
        lastCallback = new Callback();
        body.bundle.putBinder("llm_client", lastCallback);
        boolean sent = sendTask(headers, body);
        Log.i(TAG, "MIMO_CHAT sent=" + sent + " image=" + (image != null) + " json=" + json);
        if (!sent) {
            listener.onError("Mimo 任务发送失败");
        }
        return sent;
    }

    private static String buildChatJson(String prompt, boolean thinking) {
        try {
            org.json.JSONObject msg = new org.json.JSONObject();
            msg.put("role", "user");
            msg.put("content", prompt);
            org.json.JSONArray messages = new org.json.JSONArray();
            messages.put(msg);

            org.json.JSONObject root = new org.json.JSONObject();
            root.put("model", "mimo_dev_f3");
            root.put("messages", messages);
            root.put("stream", Boolean.TRUE);
            root.put("enable_thinking", thinking ? Boolean.TRUE : Boolean.FALSE);
            root.put("max_tokens", Integer.valueOf(256));
            root.put("sampler_temp", Double.valueOf(0.3));
            root.put("sampler_top_k", Integer.valueOf(50));
            root.put("sampler_top_p", Double.valueOf(1.0));
            root.put("prompt_cache_enabled", Boolean.FALSE);
            return root.toString();
        } catch (Throwable t) {
            Log.e(TAG, "CHAT_JSON_ERROR", t);
            return "";
        }
    }

    private AIHeaders makeHeaders(String api) {
        AIHeaders h = new AIHeaders();
        // The current AIService build authorizes on the caller-supplied header
        // UID; this is the same local test path as the PoC.
        h.bundle.putString("ai_api", api);
        h.bundle.putString("ai_method", "postTask");
        h.bundle.putString("package_name", context.getPackageName());
        h.bundle.putString("app_signature", "");
        h.bundle.putString("app_code", "");
        h.bundle.putInt("uid", 1000);
        h.bundle.putInt("pid", Process.myPid());
        h.bundle.putString("process_name", context.getPackageName());
        h.bundle.putInt("ai_sdk_code", 200);
        h.bundle.putString("ai_sdk_name", "taiyi-chat");
        h.bundle.putLong("task_protocol_code", 2053L);
        h.bundle.putInt("ai_api_level", 1);
        h.bundle.putLong("timestamp", System.currentTimeMillis());
        h.bundle.putString("request_id",
                "chat-" + System.currentTimeMillis() + "-" + SEQ.incrementAndGet());
        return h;
    }

    private boolean sendTask(AIHeaders headers, AIBody body) {
        if (binder == null) {
            return false;
        }
        Parcel data = Parcel.obtain();
        try {
            data.writeInterfaceToken(MANAGER_DESCRIPTOR);
            data.writeInt(1);
            data.writeParcelable(headers, 0);
            data.writeParcelable(body, 0);
            return binder.transact(2, data, null, 0);
        } catch (Throwable t) {
            Log.e(TAG, "SENDTASK_ERROR", t);
            return false;
        } finally {
            data.recycle();
        }
    }

    private int authorize(AIHeaders headers, AIBody body) {
        if (binder == null) {
            return Integer.MIN_VALUE;
        }
        Parcel data = Parcel.obtain();
        Parcel reply = Parcel.obtain();
        try {
            data.writeInterfaceToken(MANAGER_DESCRIPTOR);
            data.writeInt(1);
            data.writeParcelable(headers, 0);
            data.writeParcelable(body, 0);
            boolean ok = binder.transact(1, data, reply, 0);
            if (!ok) {
                return Integer.MIN_VALUE;
            }
            reply.readException();
            return reply.readInt();
        } catch (Throwable t) {
            Log.e(TAG, "AUTHORIZE_ERROR", t);
            return Integer.MIN_VALUE;
        } finally {
            reply.recycle();
            data.recycle();
        }
    }

    public void close() {
        if (bound) {
            try {
                context.unbindService(connection);
            } catch (Throwable ignored) {
            }
            bound = false;
        }
        binder = null;
        ready = false;
        lastBitmap = null;
        lastCallback = null;
    }

    private final class Callback extends Binder {
        @Override
        protected boolean onTransact(int code, Parcel data, Parcel reply, int flags)
                throws RemoteException {
            if (code == IBinder.INTERFACE_TRANSACTION) {
                if (reply != null) {
                    reply.writeString(CLIENT_DESCRIPTOR);
                }
                return true;
            }
            if (code == 1) { // free(List<String>)
                data.enforceInterface(CLIENT_DESCRIPTOR);
                data.createStringArrayList();
                return true;
            }
            if (code != 2) {
                return super.onTransact(code, data, reply, flags);
            }

            data.enforceInterface(CLIENT_DESCRIPTOR);
            String id = data.readString();
            int status = data.readInt();
            int error = data.readInt();

            String ans = "";
            int present = data.readInt();
            if (present != 0) {
                AIHeaders headers = data.readParcelable(AIHeaders.class.getClassLoader());
                AIBody body = data.readParcelable(AIBody.class.getClassLoader());
                if (body != null) {
                    ans = body.bundle.getString("ans", "");
                }
                if (headers != null) {
                    // no-op; useful while debugging
                }
            }
            if (status == 7 || error != 0) {
                String preview = ans == null ? "null" : ans.replace('\n', ' ');
                if (preview.length() > 240) {
                    preview = preview.substring(0, 240) + "…";
                }
                Log.i(TAG, "CALLBACK id=" + id + " status=" + status
                        + " err=" + error + " len=" + (ans == null ? 0 : ans.length())
                        + " FINAL=" + preview);
            } else {
                Log.i(TAG, "CALLBACK id=" + id + " status=" + status
                        + " err=" + error + " ans=" + (ans == null ? "null" : ans.length()));
            }
            if (error != 0) {
                if (this == initCallback && error == 10804) {
                    // The plugin answers a repeated llm_init on an already
                    // initialized engine with CODE_RESET_ERROR + 10804
                    // ("init has initialized").  That is not a failure: the
                    // engine is warm, so continue with the pending prompt.
                    handler.removeCallbacks(mimoWatchdog);
                    initCallback = null;
                    mimoInitAttempts = 0;
                    mimoWarmed = true;
                    Log.i(TAG, "MIMO_WARMUP already initialized (10804) -> continue");
                    sendPendingMimo();
                    return true;
                }
                if (this == initCallback) {
                    handler.removeCallbacks(mimoWatchdog);
                    initCallback = null;
                    mimoWarmed = false;
                    mimoInitAttempts = 0;
                    pendingPrompt = null;
                    pendingImage = null;
                    getLoggerError("MIMO_WARMUP failed err=" + error);
                }
                listener.onError("模型错误 err=" + error);
                return true;
            }

            if (this == initCallback) {
                // status 2/3/4 are progress; only 7 means init actually finished.
                if (status != 7) {
                    return true;
                }
                handler.removeCallbacks(mimoWatchdog);
                initCallback = null;
                mimoInitAttempts = 0;
                mimoWarmed = true;
                Log.i(TAG, "MIMO_WARMUP done status=" + status);
                sendPendingMimo();
                return true;
            }

            listener.onDelta(ans == null ? "" : ans, status, error);
            return true;
        }
    }

    private void getLoggerError(String message) {
        Log.e(TAG, message);
    }
}

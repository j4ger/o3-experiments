package dev.j4ger.o3chat;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

/**
 * Headless CLI entry point for the local Taiyi models.
 *
 * Usage:
 *   adb shell am start -n dev.j4ger.o3chat/.CliActivity \
 *       --es prompt "用一句中文介绍你自己" --es model vlm
 *   adb shell am start -n dev.j4ger.o3chat/.CliActivity \
 *       --es prompt "你好" --es model mimo
 *
 * Output is written to logcat under the tag O3CHAT_CLI:
 *   START / READY / SENT / DELTA status=.. err=.. text=.. / RESULT / ERROR
 */
public class CliActivity extends Activity implements TaiyiClient.Listener {
    private static final String TAG = "O3CHAT_CLI";

    private TaiyiClient client;
    private String prompt;
    private String model;
    private boolean chatMode;
    private boolean initMode;
    private boolean started;
    private boolean finished;
    private final StringBuilder out = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prompt = getIntent().getStringExtra("prompt");
        if (prompt == null || prompt.isEmpty()) {
            prompt = "用一句中文介绍你自己。";
        }
        model = getIntent().getStringExtra("model");
        if (model == null || model.isEmpty()) {
            model = "vlm";
        }
        if ("mimo".equalsIgnoreCase(model)) {
            model = "mimo_dev";
        }
        chatMode = getIntent().getBooleanExtra("chat", false);
        initMode = getIntent().getBooleanExtra("init", false);
        Log.i(TAG, "START model=" + model + " chat=" + chatMode + " init=" + initMode
                + " prompt=" + prompt);
        client = new TaiyiClient(this, this);
        client.bind();
    }

    @Override
    public void onReady() {
        if (started) {
            return;
        }
        started = true;
        Log.i(TAG, "READY");
        boolean ok = client.send(prompt, null, model, chatMode, initMode);
        Log.i(TAG, "SENT ok=" + ok);
        if (!ok) {
            Log.e(TAG, "SEND_FAILED");
            finishSoon();
        }
    }

    @Override
    public void onDelta(String text, int status, int error) {
        if (text != null && text.length() > 0) {
            out.append(text);
        }
        Log.i(TAG, "DELTA status=" + status + " err=" + error
                + " len=" + (text == null ? 0 : text.length()) + " text=" + text);
        if (status == 7 || error != 0) {
            Log.i(TAG, "RESULT status=" + status + " err=" + error + " text=" + out);
            finishSoon();
        }
    }

    @Override
    public void onError(String message) {
        Log.e(TAG, "ERROR " + message);
        finishSoon();
    }

    private void finishSoon() {
        runOnUiThread(() -> {
            if (finished) {
                return;
            }
            finished = true;
            try {
                if (client != null) {
                    client.close();
                }
            } catch (Throwable ignored) {
            }
            finish();
        });
    }
}

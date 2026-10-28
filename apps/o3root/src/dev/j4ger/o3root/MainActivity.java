package dev.j4ger.o3root;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * Standalone root/NPU tool for the Xiaomi 18 Fold (lhasa), kept separate from the
 * chat app on purpose: everything privileged (CVE-2026-98260 payload, KernelSU
 * LKM, ksud policy patching, the OpenVX probe) lives here, and the chat app
 * cannot inherit any of it.
 */
public class MainActivity extends Activity {
    private static final String TAG = "O3Root";
    private static final int REQ_SHIZUKU = 0x61;

    private TextView status;
    private TextView log;
    private final StringBuilder history = new StringBuilder();
    private boolean running;
    private boolean probeAfterRoot;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        rikka.shizuku.Shizuku.addRequestPermissionResultListener((code, result) -> {
            if (code == REQ_SHIZUKU && result == 0) {
                setStatus("已授权，开始提权…");
                runPayload();
            } else if (code == REQ_SHIZUKU) {
                running = false;
                setStatus("Shizuku 未授权");
            }
        });
        setStatus(RootChain.shizukuAlive()
                ? (RootChain.hasPermission() ? "Shizuku 就绪" : "需要授权 Shizuku")
                : "需要 Shizuku（shell 模式）");
        if (getIntent().getBooleanExtra("autorun", false)) {
            status.postDelayed(this::acquireRoot, 800);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#101014"));
        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("O3 Root · NPU");
        title.setTextColor(Color.parseColor("#E8E8F0"));
        title.setTextSize(20);
        root.addView(title);

        status = new TextView(this);
        status.setTextColor(Color.parseColor("#9AA0B4"));
        status.setTextSize(13);
        status.setPadding(0, pad / 2, 0, pad);
        root.addView(status);

        root.addView(button("1. 获取临时 root", v -> acquireRoot()));
        root.addView(button("2. 放行 NPU + 跑探针", v -> npuFlow()));
        root.addView(button("3. 显示状态", v -> setLog(dumpState())));

        ScrollView scroll = new ScrollView(this);
        log = new TextView(this);
        log.setTextColor(Color.parseColor("#B9C0D4"));
        log.setTextSize(11);
        log.setPadding(0, pad, 0, pad);
        log.setTextIsSelectable(true);
        scroll.addView(log);
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        setContentView(root);
    }

    private Button button(String label, android.view.View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setGravity(Gravity.CENTER);
        b.setOnClickListener(l);
        return b;
    }

    private void setStatus(String s) {
        Log.i(TAG, "STATUS " + s);
        runOnUiThread(() -> status.setText(s));
    }

    private void appendLog(String line) {
        Log.i(TAG, line);
        runOnUiThread(() -> {
            history.append(line).append('\n');
            if (history.length() > 20000) {
                history.delete(0, history.length() - 20000);
            }
            log.setText(history.toString());
        });
    }

    private void setLog(String s) {
        history.setLength(0);
        appendLog(s);
    }

    private String dumpState() {
        return "shizuku=" + RootChain.shizukuAlive()
                + " perm=" + RootChain.hasPermission()
                + "\nrooted=" + RootChain.rooted();
    }

    private void acquireRoot() {
        if (running) {
            return;
        }
        if (!RootChain.shizukuAlive()) {
            setStatus("需要 Shizuku（shell 模式）");
            appendLog("Shizuku binder 不存在：请先以 shell 模式启动 Shizuku");
            return;
        }
        if (!RootChain.hasPermission()) {
            setStatus("请求 Shizuku 权限…");
            RootChain.requestPermission(REQ_SHIZUKU);
            return;
        }
        runPayload();
    }

    private void runPayload() {
        running = true;
        setStatus("获取临时 root…（可能需要几次尝试）");
        RootChain.runPayload(this, new RootChain.Sink() {
            @Override
            public void onLine(String line) {
                appendLog(line);
            }

            @Override
            public void onExit(int code) {
                boolean ok = RootChain.rooted();
                running = false;
                setStatus(ok ? "root 就绪（当次开机有效）" : "提权失败 code=" + code);
                appendLog("exit=" + code + " rooted=" + ok);
                if (ok && probeAfterRoot) {
                    probeAfterRoot = false;
                    runProbe();
                }
            }
        }, null);
    }

    private void npuFlow() {
        setStatus("准备探针与 ksud…");
        new Thread(() -> {
            appendLog(RootChain.prepare(this));
            if (RootChain.rooted()) {
                runProbe();
            } else {
                probeAfterRoot = true;
                runOnUiThread(this::acquireRoot);
            }
        }, "prepare").start();
    }

    private void runProbe() {
        setStatus("放行 NPU 并运行探针…");
        new Thread(() -> {
            String out = RootChain.runNpuProbe();
            appendLog(out);
            setStatus("探针结束（见下方输出）");
        }, "probe").start();
    }
}

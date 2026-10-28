package dev.j4ger.o3chat;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.InputType;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Local VLM chat demo backed by com.xiaomi.aiservice. */
public final class ChatActivity extends Activity implements TaiyiClient.Listener {
    private static final int REQ_PICK = 1001;
    private static final int ICON_IMAGE = 1;
    private static final int ICON_SAMPLE = 2;
    private static final int BG = Color.parseColor("#0B0F14");
    private static final int PANEL = Color.parseColor("#121820");
    private static final int CARD = Color.parseColor("#1B2430");
    private static final int CARD_2 = Color.parseColor("#202C3A");
    private static final int ACCENT = Color.parseColor("#2563EB");
    private static final int TEXT = Color.parseColor("#E6EDF3");
    private static final int MUTED = Color.parseColor("#8FA3B8");
    private static final int GREEN = Color.parseColor("#34D399");

    private TaiyiClient client;
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private TextView sendButton;
    private View attachButton;
    private View sampleButton;
    private TextView statusView;
    private TextView subtitleView;
    private TextView modelButton;
    private String selectedModel = "vlm";
    private LinearLayout previewRow;
    private ImageView previewImage;
    private Bitmap pendingImage;
    private boolean generating;
    private Msg currentAssistant;
    private final List<Msg> conversation = new ArrayList<>();
    private boolean autoTest;
    private boolean autoTextOnly;
    private boolean chatMode;
    private boolean initMode;
    private boolean autoStarted;
    private long requestStartedMs;
    private long firstTokenMs;
    private long lastTokenMs;
    private int tokenCount;

    private TextView rootButton;
    private boolean mimoStatusChecked;
    private boolean autoMimoStatus;
    private boolean autoAcquireRoot;
    private boolean autoStaging;
    private boolean autoAccess;
    private boolean autoArmSwap;
    private boolean autoDlStart;
    private boolean autoNpuProbe;
    private String probeScene = "text_llm";
    private String probeApis;
    private String swapTarget;
    private String swapPayload;
    private long swapSize;
    private long swapDur = 30000L;
    private static final String TAG = "TaiyiChat";

    private static final class Msg {
        boolean user;
        String text;
        Bitmap image;
        TextView textView;
        TextView metaView;
        boolean streaming;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setDecorFitsSystemWindows(false);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        autoTest = getIntent().getBooleanExtra("autotest", false);
        autoTextOnly = getIntent().getBooleanExtra("textonly", false);
        chatMode = getIntent().getBooleanExtra("chat", false);
        initMode = getIntent().getBooleanExtra("init", false);
        autoMimoStatus = getIntent().getBooleanExtra("mimostatus", false);
        autoAcquireRoot = getIntent().getBooleanExtra("acquireRoot", false);
        autoStaging = getIntent().getBooleanExtra("staging", false);
        autoAccess = getIntent().getBooleanExtra("probeAccess", false);
        autoArmSwap = getIntent().getBooleanExtra("armswap", false);
        autoDlStart = getIntent().getBooleanExtra("dlstart", false);
        autoNpuProbe = getIntent().getBooleanExtra("npuprobe", false);
        if (getIntent().hasExtra("scene")) {
            probeScene = getIntent().getStringExtra("scene");
        }
        probeApis = getIntent().getStringExtra("apis");
        swapTarget = getIntent().getStringExtra("swaptarget");
        swapPayload = getIntent().getStringExtra("swappayload");
        swapSize = getIntent().getLongExtra("swapesize", 0L);
        swapDur = getIntent().getLongExtra("swapdur", 30000L);
        if ("mimo".equalsIgnoreCase(getIntent().getStringExtra("model"))) {
            selectedModel = "mimo_dev";
        }
        buildUi();
        applyModelUi();
        addAssistant("你好，我是运行在这台手机上的本地 VLM 助手。\n可以发文字，也可以点左下角的图片按钮选图，或点星形示例按钮快速试一张。");
        client = new TaiyiClient(this, this);
        client.bind();
        if (autoMimoStatus || autoAcquireRoot || autoStaging || autoAccess
                || autoArmSwap || autoDlStart || autoNpuProbe) {
            statusView.postDelayed(() -> {
                if (autoNpuProbe) {
                    launchO3Root(true);
                } else if (autoAccess) {
                    accessProbe();
                } else if (autoStaging) {
                    stagingScan();
                } else if (autoArmSwap) {
                    armSwap();
                } else if (autoDlStart) {
                    downloadProbe(true);
                } else if (autoAcquireRoot) {
                    launchO3Root(false);
                } else {
                    probeMimoStatus();
                }
            }, 1500);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            int top = insets.getInsets(WindowInsets.Type.statusBars()).top;
            int bottom = Math.max(
                    insets.getInsets(WindowInsets.Type.navigationBars()).bottom,
                    insets.getInsets(WindowInsets.Type.ime()).bottom);
            v.setPadding(0, top, 0, bottom);
            return insets;
        });

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(20), dp(14), dp(20), dp(12));
        bar.setBackgroundColor(PANEL);

        LinearLayout titles = new LinearLayout(this);
        titles.setOrientation(LinearLayout.VERTICAL);
        TextView title = new TextView(this);
        title.setText("Taiyi Chat");
        title.setTextColor(TEXT);
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 21);
        title.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        TextView subtitle = new TextView(this);
        subtitleView = subtitle;
        subtitle.setText("本地 VLM · Qwen3 0.7B · on-device");
        subtitle.setTextColor(MUTED);
        subtitle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        titles.addView(title);
        titles.addView(subtitle);
        bar.addView(titles, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        modelButton = pill("VLM", CARD_2, Color.parseColor("#7DD3FC"), dp(64));
        modelButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        modelButton.setOnClickListener(v -> toggleModel());
        bar.addView(modelButton);

        rootButton = pill("Root", CARD_2, Color.parseColor("#FCA5A5"), dp(56));
        rootButton.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        rootButton.setContentDescription("打开 O3 Root（提权/NPU 工具）；长按扫描资源暂存目录");
        rootButton.setOnClickListener(v -> launchO3Root(false));
        rootButton.setOnLongClickListener(v -> {
            stagingScan();
            return true;
        });
        bar.addView(rootButton);

        statusView = new TextView(this);
        statusView.setText("连接中…");
        statusView.setTextColor(MUTED);
        statusView.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        statusView.setGravity(Gravity.CENTER);
        statusView.setPadding(dp(10), dp(5), dp(10), dp(5));
        statusView.setBackground(round(Color.parseColor("#1B2430"), 20));
        bar.addView(statusView);
        root.addView(bar);

        View divider = new View(this);
        divider.setBackgroundColor(Color.parseColor("#1E2A38"));
        root.addView(divider, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(1)));

        scroll = new ScrollView(this);
        scroll.setFillViewport(false);
        scroll.setClipToPadding(false);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(12), dp(10), dp(12), dp(12));
        scroll.addView(messages, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        LinearLayout bottom = new LinearLayout(this);
        bottom.setOrientation(LinearLayout.VERTICAL);
        bottom.setPadding(dp(12), dp(8), dp(12), dp(12));
        bottom.setBackgroundColor(PANEL);

        previewRow = new LinearLayout(this);
        previewRow.setOrientation(LinearLayout.HORIZONTAL);
        previewRow.setGravity(Gravity.CENTER_VERTICAL);
        previewRow.setVisibility(View.GONE);
        previewRow.setPadding(dp(8), dp(6), dp(8), dp(6));
        previewRow.setBackground(round(CARD, 14));

        previewImage = new ImageView(this);
        previewImage.setScaleType(ImageView.ScaleType.CENTER_CROP);
        previewImage.setClipToOutline(true);
        previewImage.setBackground(round(CARD_2, 10));
        previewRow.addView(previewImage, new LinearLayout.LayoutParams(dp(52), dp(52)));

        TextView previewLabel = new TextView(this);
        previewLabel.setText("已选择图片");
        previewLabel.setTextColor(TEXT);
        previewLabel.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        previewLabel.setPadding(dp(10), 0, 0, 0);
        previewRow.addView(previewLabel, new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        TextView remove = new TextView(this);
        remove.setText("移除");
        remove.setTextColor(MUTED);
        remove.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        remove.setPadding(dp(10), dp(6), dp(10), dp(6));
        remove.setOnClickListener(v -> {
            pendingImage = null;
            updatePreview();
        });
        previewRow.addView(remove);
        bottom.addView(previewRow, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.BOTTOM);
        row.setPadding(0, dp(8), 0, 0);

        attachButton = iconButton(ICON_IMAGE, "选择图片");
        attachButton.setOnClickListener(v -> pickImage());
        row.addView(attachButton);

        sampleButton = iconButton(ICON_SAMPLE, "示例图片");
        sampleButton.setOnClickListener(v -> loadSample());
        row.addView(sampleButton);

        input = new EditText(this);
        input.setHint("输入消息…");
        input.setHintTextColor(Color.parseColor("#6B7C90"));
        input.setTextColor(TEXT);
        input.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
        input.setInputType(InputType.TYPE_CLASS_TEXT
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        input.setMinHeight(dp(46));
        input.setMaxLines(4);
        input.setPadding(dp(14), dp(10), dp(14), dp(10));
        input.setBackground(round(CARD, 22));
        LinearLayout.LayoutParams inputParams = new LinearLayout.LayoutParams(0,
                ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        inputParams.setMargins(dp(8), 0, dp(8), 0);
        row.addView(input, inputParams);

        sendButton = pill("发送", ACCENT, Color.WHITE, dp(68));
        sendButton.setOnClickListener(v -> sendMessage());
        row.addView(sendButton);
        bottom.addView(row);
        root.addView(bottom);

        setContentView(root);
        root.requestApplyInsets();
        updatePreview();
    }

    private ImageView iconButton(int kind, String description) {
        ImageView v = new ImageView(this);
        v.setImageBitmap(makeIcon(kind));
        v.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
        v.setPadding(dp(11), dp(11), dp(11), dp(11));
        v.setBackground(round(CARD_2, 22));
        v.setClickable(true);
        v.setFocusable(true);
        v.setContentDescription(description);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(dp(44), dp(44));
        p.setMargins(0, 0, dp(6), 0);
        v.setLayoutParams(p);
        return v;
    }

    private Bitmap makeIcon(int kind) {
        int size = 64;
        Bitmap b = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(b);
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(TEXT);
        p.setStrokeWidth(4);
        p.setStrokeCap(Paint.Cap.ROUND);
        p.setStrokeJoin(Paint.Join.ROUND);
        if (kind == ICON_IMAGE) {
            p.setStyle(Paint.Style.STROKE);
            c.drawRoundRect(10, 12, 54, 50, 7, 7, p);
            p.setStyle(Paint.Style.FILL);
            c.drawCircle(23, 25, 4, p);
            p.setStyle(Paint.Style.STROKE);
            Path mountain = new Path();
            mountain.moveTo(13, 45);
            mountain.lineTo(27, 31);
            mountain.lineTo(36, 40);
            mountain.lineTo(44, 30);
            mountain.lineTo(52, 45);
            c.drawPath(mountain, p);
        } else {
            p.setColor(Color.parseColor("#7DD3FC"));
            p.setStyle(Paint.Style.FILL);
            Path star = new Path();
            star.moveTo(32, 8);
            star.lineTo(38, 25);
            star.lineTo(56, 32);
            star.lineTo(38, 39);
            star.lineTo(32, 56);
            star.lineTo(26, 39);
            star.lineTo(8, 32);
            star.lineTo(26, 25);
            star.close();
            c.drawPath(star, p);
            p.setColor(TEXT);
            p.setStrokeWidth(3);
            p.setStyle(Paint.Style.STROKE);
            c.drawCircle(49, 15, 4, p);
        }
        return b;
    }

    private TextView pill(String text, int color, int textColor, int width) {
        TextView v = new TextView(this);
        v.setText(text);
        v.setTextColor(textColor);
        v.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14);
        v.setGravity(Gravity.CENTER);
        v.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        v.setBackground(round(color, 22));
        v.setClickable(true);
        v.setFocusable(true);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(width, dp(44));
        p.setMargins(0, 0, dp(6), 0);
        v.setLayoutParams(p);
        return v;
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, REQ_PICK);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQ_PICK || resultCode != RESULT_OK || data == null) {
            return;
        }
        Uri uri = data.getData();
        if (uri == null) {
            return;
        }
        try {
            Bitmap bmp = decodeSampled(uri);
            if (bmp == null) {
                Toast.makeText(this, "无法读取这张图片", Toast.LENGTH_SHORT).show();
                return;
            }
            pendingImage = scaleToMax(bmp, 448);
            if (pendingImage != bmp) {
                bmp.recycle();
            }
            updatePreview();
        } catch (Throwable t) {
            Toast.makeText(this, "读取图片失败：" + t.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private Bitmap decodeSampled(Uri uri) throws Exception {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }
        int sample = 1;
        int w = Math.max(1, bounds.outWidth);
        int h = Math.max(1, bounds.outHeight);
        while (w / sample > 896 || h / sample > 896) {
            sample *= 2;
        }
        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(in, null, opts);
        }
    }

    private void loadSample() {
        pendingImage = scaleToMax(makeSampleBitmap(), 448);
        input.setText("请描述这张图片里的形状、颜色和文字。");
        updatePreview();
    }

    private void updatePreview() {
        if (pendingImage == null) {
            previewRow.setVisibility(View.GONE);
            previewImage.setImageDrawable(null);
        } else {
            previewRow.setVisibility(View.VISIBLE);
            previewImage.setImageBitmap(pendingImage);
        }
    }

    private void sendMessage() {
        if (generating) {
            return;
        }
        if (!client.isReady()) {
            Toast.makeText(this, "本地模型还在连接…", Toast.LENGTH_SHORT).show();
            return;
        }
        String text = input.getText().toString().trim();
        Bitmap image = pendingImage;
        if (text.isEmpty() && image == null) {
            return;
        }
        String userText = text;
        if (userText.isEmpty()) {
            userText = "请描述这张图片。";
        }
        String prompt = buildPrompt(userText, image);

        input.setText("");
        pendingImage = null;
        updatePreview();

        Msg user = addMessage(true, userText, image);
        Msg assistant = addMessage(false, "…", null);
        assistant.streaming = true;
        currentAssistant = assistant;
        conversation.add(user);
        conversation.add(assistant);

        requestStartedMs = SystemClock.elapsedRealtime();
        firstTokenMs = 0L;
        lastTokenMs = 0L;
        tokenCount = 0;

        setGenerating(true);
        if ("mimo_dev".equals(selectedModel) && !mimoStatusChecked) {
            // Preflight: ask AIService whether the Mimo bundle is actually installed.
            probeMimoStatus();
        }
        if (!client.send(prompt, image, selectedModel, chatMode, initMode)) {
            assistant.text = "调用本地模型失败。";
            assistant.textView.setText(assistant.text);
            assistant.streaming = false;
            currentAssistant = null;
            setGenerating(false);
        }
    }

    private String buildPrompt(String userText, Bitmap image) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是一个运行在手机本地的中文助手。回答简洁、准确，优先使用中文。\n");
        int start = Math.max(0, conversation.size() - 6);
        for (int i = start; i < conversation.size(); i++) {
            Msg m = conversation.get(i);
            if (m.text == null || m.text.isEmpty() || "…".equals(m.text)) {
                continue;
            }
            sb.append(m.user ? "用户: " : "助手: ");
            sb.append(trim(m.text, 500)).append('\n');
        }
        sb.append("用户: ").append(userText).append('\n');
        if (image != null) {
            sb.append("[用户附带了一张图片，请结合图片内容回答。]\n");
        }
        sb.append("助手:");
        return sb.toString();
    }

    private Msg addMessage(boolean user, String text, Bitmap image) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(user ? Gravity.END : Gravity.START);
        row.setPadding(0, dp(4), 0, dp(4));

        LinearLayout bubble = new LinearLayout(this);
        bubble.setOrientation(LinearLayout.VERTICAL);
        bubble.setPadding(dp(14), dp(10), dp(14), dp(10));
        bubble.setBackground(round(user ? ACCENT : CARD, 18));

        Msg msg = new Msg();
        msg.user = user;
        msg.text = text;
        msg.image = image;

        if (image != null) {
            ImageView iv = new ImageView(this);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            iv.setClipToOutline(true);
            iv.setBackground(round(CARD_2, 12));
            iv.setImageBitmap(image);
            LinearLayout.LayoutParams ip = new LinearLayout.LayoutParams(dp(220), dp(150));
            ip.setMargins(0, 0, 0, dp(8));
            bubble.addView(iv, ip);
        }

        if (!user) {
            TextView label = new TextView(this);
            label.setText("本地 VLM");
            label.setTextColor(Color.parseColor("#7DD3FC"));
            label.setTextSize(TypedValue.COMPLEX_UNIT_SP, 10);
            label.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
            label.setLetterSpacing(0.08f);
            label.setPadding(0, 0, 0, dp(4));
            bubble.addView(label);
        }

        if (text != null && !text.isEmpty()) {
            TextView tv = new TextView(this);
            tv.setText(text);
            tv.setTextColor(user ? Color.WHITE : TEXT);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
            tv.setLineSpacing(dp(3), 1f);
            tv.setTextIsSelectable(true);
            tv.setSingleLine(false);
            tv.setMaxWidth(dp(300));
            if (user && image != null) {
                bubble.addView(tv, new LinearLayout.LayoutParams(dp(220),
                        ViewGroup.LayoutParams.WRAP_CONTENT));
            } else {
                bubble.addView(tv);
            }
            msg.textView = tv;
        } else if (user) {
            // image-only user message
        } else {
            TextView tv = new TextView(this);
            tv.setText(text == null ? "" : text);
            tv.setTextColor(TEXT);
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15);
            tv.setTextIsSelectable(true);
            tv.setMaxWidth(dp(300));
            bubble.addView(tv);
            msg.textView = tv;
        }

        if (!user) {
            TextView meta = new TextView(this);
            meta.setTextSize(TypedValue.COMPLEX_UNIT_SP, 10);
            meta.setTextColor(Color.parseColor("#6B7C90"));
            meta.setLetterSpacing(0.02f);
            meta.setPadding(0, dp(5), 0, 0);
            meta.setVisibility(View.GONE);
            bubble.addView(meta);
            msg.metaView = meta;
        }

        row.addView(bubble, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        messages.addView(row, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        scrollToBottom();
        return msg;
    }

    private void addAssistant(String text) {
        Msg m = addMessage(false, text, null);
        conversation.add(m);
    }

    private void setGenerating(boolean value) {
        generating = value;
        sendButton.setEnabled(!value);
        attachButton.setEnabled(!value);
        sampleButton.setEnabled(!value);
        sendButton.setAlpha(value ? 0.5f : 1f);
        statusView.setText(value ? "生成中…" : (client != null && client.isReady() ? "就绪" : "连接中…"));
        statusView.setTextColor(value ? MUTED : (client != null && client.isReady() ? GREEN : MUTED));
    }

    private void scrollToBottom() {
        scroll.post(() -> scroll.scrollTo(0, messages.getHeight()));
        scroll.postDelayed(() -> scroll.scrollTo(0, messages.getHeight()), 150);
    }

    @Override
    public void onReady() {
        runOnUiThread(() -> {
            statusView.setText("就绪");
            statusView.setTextColor(GREEN);
            if (autoTest && !autoStarted) {
                autoStarted = true;
                statusView.postDelayed(this::runAutoTest, 600);
            }
        });
    }

    private void runAutoTest() {
        if (autoTextOnly) {
            input.setText("用一句中文介绍你自己。");
        } else {
            loadSample();
        }
        sendMessage();
    }

    private void toggleModel() {
        selectedModel = "mimo_dev".equals(selectedModel) ? "vlm" : "mimo_dev";
        applyModelUi();
    }

    private String modelLabel() {
        return "mimo_dev".equals(selectedModel) ? "Mimo" : "VLM";
    }

    private void applyModelUi() {
        if (modelButton != null) {
            modelButton.setText(modelLabel());
            modelButton.setContentDescription("当前模型 " + modelLabel() + "，点按切换");
        }
        if (subtitleView != null) {
            subtitleView.setText("mimo_dev".equals(selectedModel)
                    ? "本地 Mimo · on-device"
                    : "本地 VLM · Qwen3 0.7B · on-device");
        }
    }

    private void setStatusText(String text) {
        if (statusView != null) {
            statusView.setText(text);
        }
    }

    /** Query AIService for the real install state of the Mimo bundle. */
    private void probeMimoStatus() {
        mimoStatusChecked = true;
        new Thread(() -> {
            ModelStatus.Info mimo = ModelStatus.mimo(this);
            ModelStatus.Info bundle = ModelStatus.mimoBundle(this);
            Log.i(TAG, "MIMO_STATUS mimo_dev " + mimo
                    + (mimo.rawKeys != null ? " keys=[" + mimo.rawKeys + "]" : ""));
            Log.i(TAG, "MIMO_STATUS bundle  " + bundle
                    + (bundle.rawKeys != null ? " keys=[" + bundle.rawKeys + "]" : ""));
            if (bundle.configJson != null) {
                Log.i(TAG, "MIMO_STATUS config=" + bundle.configJson);
            }
            if (bundle.resourceJson != null) {
                Log.i(TAG, "MIMO_STATUS resources=" + bundle.resourceJson);
            }
            runOnUiThread(() -> setStatusText("Mimo full="
                    + ModelStatus.human(mimo.fullSize)
                    + " left=" + ModelStatus.human(mimo.leftDownload)));
        }, "mimo-status").start();
    }

    /** shell-side view of the AIService resource staging tree. */
    private void stagingScan() {
        new Thread(() -> {
            Log.i(TAG, "STAGING_SHELL " + Shell.state());
            if (!Shell.available()) {
                runOnUiThread(() -> setStatusText("需要 Shizuku"));
                return;
            }
            boolean installed = StagingSwap.installScript(this);
            Log.i(TAG, "STAGING_SCRIPT installed=" + installed + " path=" + StagingSwap.SCRIPT);
            String listing = StagingSwap.scan();
            int files = 0;
            for (String line : listing.split("\n")) {
                String t = line.trim();
                if (t.isEmpty()) {
                    continue;
                }
                Log.i(TAG, "STAGING_FILE " + t);
                if (!t.startsWith("STAGING_")) {
                    files++;
                }
            }
            final int shown = files;
            runOnUiThread(() -> setStatusText("staging 文件 " + shown));
        }, "staging-scan").start();
    }

    /** three-way access probe: bare shell / hosted root / KSU su. */
    private void accessProbe() {
        new Thread(() -> {
            String cmd = "echo ACCESS_ID uid=$(id -u) ctx=$(id -Z 2>/dev/null); "
                    + "echo ACCESS_SHELL_LS; ls /data/user_de/0/com.xiaomi.aiservice 2>&1 | head -3; "
                    + "echo ACCESS_SU_BIN=$(command -v su || echo none); "
                    + "echo ACCESS_SU_ID; timeout 5 su -c id 2>&1 | head -2; "
                    + "echo ACCESS_STAGING; touch '" + StagingSwap.ROOT + "/.probe' 2>&1 "
                    + "&& echo WRITE_OK && rm -f '" + StagingSwap.ROOT + "/.probe' || echo WRITE_DENIED; "
                    + "echo ACCESS_DONE";
            String out = Shell.exec(cmd);
            for (String line : out.split("\n")) {
                if (!line.trim().isEmpty()) {
                    Log.i(TAG, "ACCESS " + line.trim());
                }
            }
            runOnUiThread(() -> setStatusText("访问探测完成"));
        }, "access-probe").start();
    }

    private static java.util.List<String> parseApis(String csv) {
        java.util.List<String> out = new java.util.ArrayList<>();
        if (csv != null) {
            for (String p : csv.split(",")) {
                String t = p.trim();
                if (!t.isEmpty()) {
                    out.add(t);
                }
            }
        }
        return out;
    }

    /** Query the official resource state for an api set (and optionally start it). */
    private void downloadProbe(final boolean start) {
        final String scene = probeScene;
        final java.util.List<String> apis = parseApis(probeApis);
        new Thread(() -> {
            ModelStatus.Info info = ModelStatus.query(this, scene, apis);
            Log.i(TAG, "DL_PROBE scene=" + scene + " apis=" + apis + " -> " + info
                    + (info.rawKeys != null ? " keys=[" + info.rawKeys + "]" : ""));
            if (start && info.leftDownload > 0) {
                DownloadTrigger.start(this, scene, apis);
                Log.i(TAG, "DL_PROBE started download for scene=" + scene);
            } else if (start) {
                Log.i(TAG, "DL_PROBE nothing to download (left=0)");
            }
            runOnUiThread(() -> setStatusText("left=" + ModelStatus.human(info.leftDownload)
                    + " full=" + ModelStatus.human(info.fullSize)));
        }, "dl-probe").start();
    }

    /** Arm the staging swap for the window during which AIService installs. */
    private void armSwap() {
        final String target = swapTarget;
        final String payload = (swapPayload == null || swapPayload.isEmpty())
                ? StagingSwap.DEFAULT_PAYLOAD : swapPayload;
        final long size = swapSize;
        final long dur = swapDur;
        if (target == null || target.isEmpty()) {
            Log.e(TAG, "ARM_SWAP missing --es swaptarget");
            setStatusText("缺少 swaptarget");
            return;
        }
        Log.i(TAG, "ARM_SWAP " + StagingSwap.describe(target, payload, size, dur));
        new Thread(() -> {
            if (!Shell.available()) {
                Log.w(TAG, "ARM_SWAP no shizuku");
                runOnUiThread(() -> setStatusText("需要 Shizuku"));
                return;
            }
            StagingSwap.installScript(this);
            StagingSwap.arm(target, payload, size, dur, new Shell.Sink() {
                @Override
                public void onLine(String line) {
                    Log.i(TAG, "SWAP " + line);
                    runOnUiThread(() -> setStatusText(line.length() > 24
                            ? line.substring(0, 24) : line));
                }

                @Override
                public void onExit(int code) {
                    Log.i(TAG, "SWAP_EXIT code=" + code);
                    runOnUiThread(() -> setStatusText(code == 0 ? "替换窗口结束" : "替换异常 " + code));
                }
            });
        }, "arm-swap").start();
    }

    /**
     * Everything privileged lives in dev.j4ger.o3root: the chat app only hands
     * over. Deep link with autorun so the tool goes straight to work.
     */
    private void launchO3Root(boolean autorun) {
        android.content.Intent i = new android.content.Intent();
        i.setClassName("dev.j4ger.o3root", "dev.j4ger.o3root.MainActivity");
        if (autorun) {
            i.putExtra("autorun", true);
        }
        i.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            startActivity(i);
            setStatusText(autorun ? "已交给 O3 Root（提权+探针）" : "已打开 O3 Root");
        } catch (Throwable t) {
            Log.w(TAG, "O3ROOT launch failed: " + t);
            setStatusText("未安装 O3 Root");
        }
    }

    @Override
    public void onDelta(String text, int status, int error) {
        runOnUiThread(() -> {
            if (currentAssistant == null) {
                return;
            }
            long now = SystemClock.elapsedRealtime();
            if (status == 7) {
                if (firstTokenMs == 0L) {
                    firstTokenMs = now;
                    tokenCount = Math.max(1, tokenCount);
                }
                if (lastTokenMs == 0L) {
                    lastTokenMs = now;
                }
                if (text != null && !text.isEmpty()) {
                    currentAssistant.text = text;
                } else if ("…".equals(currentAssistant.text)) {
                    currentAssistant.text = "（没有返回内容）";
                }
                currentAssistant.streaming = false;
                if (currentAssistant.textView != null) {
                    currentAssistant.textView.setText(currentAssistant.text);
                }
                showMetrics(currentAssistant, now);
                scrollToBottom();
                currentAssistant = null;
                setGenerating(false);
            } else if (status == 4) {
                if (text == null || text.isEmpty()) {
                    return;
                }
                if (firstTokenMs == 0L) {
                    firstTokenMs = now;
                }
                lastTokenMs = now;
                tokenCount++;
                String old = currentAssistant.text == null || "…".equals(currentAssistant.text)
                        ? "" : currentAssistant.text;
                if (!old.isEmpty() && text.startsWith(old)) {
                    currentAssistant.text = text;
                } else {
                    currentAssistant.text = old + text;
                }
                if (currentAssistant.textView != null) {
                    currentAssistant.textView.setText(currentAssistant.text);
                }
                scrollToBottom();
            }
        });
    }

    private void showMetrics(Msg msg, long finishedAt) {
        if (msg.metaView == null) {
            return;
        }
        long ttft = Math.max(0L, firstTokenMs - requestStartedMs);
        long total = Math.max(0L, finishedAt - requestStartedMs);
        long span = Math.max(0L, lastTokenMs - firstTokenMs);
        double tps = 0.0;
        if (span > 0 && tokenCount > 1) {
            tps = (tokenCount - 1) * 1000.0 / span;
        }
        String tpsText = tps > 0 ? String.format(Locale.US, "%.1f", tps) : "--";
        String metrics = String.format(Locale.US,
                "TTFT %s  ·  TPS %s  ·  %d tok  ·  %s",
                formatDuration(ttft), tpsText, tokenCount, formatDuration(total));
        msg.metaView.setText(metrics);
        msg.metaView.setVisibility(View.VISIBLE);
        Log.i("TaiyiChat", "METRICS ttft=" + ttft + " tps=" + tpsText
                + " tokens=" + tokenCount + " total=" + total);
    }

    private static String formatDuration(long ms) {
        if (ms < 1000L) {
            return ms + " ms";
        }
        return String.format(Locale.US, "%.2f s", ms / 1000.0);
    }

    @Override
    public void onError(String message) {
        runOnUiThread(() -> {
            boolean wasMimo = "mimo_dev".equals(selectedModel);
            if (wasMimo) {
                selectedModel = "vlm";
                applyModelUi();
                if (client != null) {
                    // Next Mimo attempt must run a fresh llm_init.
                    client.resetMimoWarm();
                }
            }
            int color = wasMimo ? Color.parseColor("#FBBF24")
                    : Color.parseColor("#F87171");
            statusView.setText(wasMimo ? "已切回 VLM" : "错误");
            statusView.setTextColor(color);
            Toast.makeText(this, wasMimo ? "Mimo 不可用，已切回 VLM" : message,
                    Toast.LENGTH_LONG).show();
            if (currentAssistant != null) {
                currentAssistant.text = wasMimo
                        ? "Mimo 当前不可用（" + message + "），已切回 VLM。可以直接重试。"
                        : "出错了：" + message;
                if (currentAssistant.textView != null) {
                    currentAssistant.textView.setText(currentAssistant.text);
                }
                currentAssistant = null;
            }
            setGenerating(false);
            statusView.setText(wasMimo ? "已切回 VLM" : "错误");
            statusView.setTextColor(color);
            Log.i("TaiyiChat", "MODEL_FALLBACK wasMimo=" + wasMimo + " message=" + message);
        });
    }

    private static Bitmap scaleToMax(Bitmap src, int max) {
        int w = src.getWidth();
        int h = src.getHeight();
        int side = Math.max(w, h);
        if (side <= max) {
            return src;
        }
        float scale = max / (float) side;
        return Bitmap.createScaledBitmap(src, Math.max(1, Math.round(w * scale)),
                Math.max(1, Math.round(h * scale)), true);
    }

    private static Bitmap makeSampleBitmap() {
        int size = 512;
        Bitmap b = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(b);
        c.drawColor(Color.WHITE);
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.parseColor("#E11D48"));
        c.drawCircle(150, 150, 82, p);
        p.setColor(Color.parseColor("#2563EB"));
        c.drawRect(300, 90, 450, 240, p);
        p.setColor(Color.parseColor("#16A34A"));
        Path triangle = new Path();
        triangle.moveTo(120, 380);
        triangle.lineTo(220, 380);
        triangle.lineTo(170, 290);
        triangle.close();
        c.drawPath(triangle, p);
        p.setColor(Color.BLACK);
        p.setTextSize(42);
        p.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        c.drawText("HELLO", 70, 470, p);
        c.drawText("42", 360, 420, p);
        return b;
    }

    private static String trim(String s, int max) {
        if (s == null) {
            return "";
        }
        return s.length() <= max ? s : s.substring(0, max) + "…";
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private GradientDrawable round(int color, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radiusDp));
        return d;
    }

    @Override
    protected void onDestroy() {
        if (client != null) {
            client.close();
        }
        super.onDestroy();
    }
}

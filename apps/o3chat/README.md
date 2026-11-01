# O3 Chat (`dev.j4ger.o3chat`)

Local VLM chat on the phone's own model. It talks to
`com.xiaomi.aiservice` through the `AI_ABILITY` interface and calls the model
already installed on the device (`vlm_qwen3_0_7b_f3`); it does not bundle or
load any third-party model.

No root, no exploit code: all privileged code lives in
[`../o3root`](../o3root). The `Root` button is a plain deep link into that app.

## Build

```bash
export ANDROID_BUILD_TOOLS=/path/to/build-tools
export ANDROID_JAR=/path/to/platforms/android-XX/android.jar
apps/o3chat/build.sh          # -> apps/o3chat/o3chat.apk
```

## Install and run

```bash
adb install -r apps/o3chat/o3chat.apk
adb shell am start -n dev.j4ger.o3chat/.ChatActivity
```

## Features

- text chat with the local VLM
- image attach through the system picker, delivered via the `IBitmap` Binder
  (`imageCount` becomes 1 in the plugin)
- streaming tokens, short conversation context
- footer metrics: TTFT, tokens/s, token count, total time
- `示例` button with a built-in image for quick VLM checks

## Requirements

- `com.xiaomi.aiservice` with the local model installed
- the app uses AIService's current header-UID path for local research; if Xiaomi
  changes that authorization check, a normal `authorize()` flow will be needed

## Caveats

Tested against `com.xiaomi.aiservice` 4.2.12 on HyperOS 4 / Android 17.
The Mimo model path (`ai_api = mimo_dev`) needs a different entry sequence; the
research notes are in
[`../../archive/o3chat-mimo-notes.md`](../../archive/o3chat-mimo-notes.md).

## CLI mode

`CliActivity` is a headless entry point that binds the same `TaiyiClient` as the UI
and prints every callback to logcat under the `O3CHAT_CLI` tag.

```bash
# Qwen3 0.7B VLM
adb shell am start -n dev.j4ger.o3chat/.CliActivity \
    --es prompt "用一句中文介绍你自己" --es model vlm

# MiMo (auto llm_init + executeChat)
adb shell am start -n dev.j4ger.o3chat/.CliActivity \
    --es prompt "用一句中文介绍你自己" --es model mimo

# read the result
adb logcat -d -s O3CHAT_CLI
```

Optional extras: `--ez chat true` (executeChat + json_source), `--ez init true`
(only send `ai:method:llm_init`).

Verified 2026-11-01 on the O3 device:

- VLM: `我是一个AI语言模型，能够理解并回答中文问题，提供各种信息和帮助。`
- MiMo: `我是 MiMo，一个由小米创造的 AI 助手。……`

#!/usr/bin/env node
// Reconstruct the signed AIService deploy-list request.
// Usage:
//   AISERVICE_SECRET='<secret from R.string.secret_key>' node tools/query-ai-config.mjs --stage 1 --out artifacts/ai-config.json
// The secret is intentionally not hard-coded here.
import crypto from 'node:crypto';
import fs from 'node:fs';

const secret = process.env.AISERVICE_SECRET;
if (!secret) {
  console.error('AISERVICE_SECRET is required');
  process.exit(2);
}

const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};

const stage = Number(getArg('--stage', '1'));
const out = getArg('--out', 'artifacts/ai-config.json');
const base = 'https://ai.engine.miui.com/api/backend/deploy/list';

function sign(map) {
  const s = Object.keys(map).sort().map(k => `${k}=${map[k]}&`).join('') + secret;
  const b64 = Buffer.from(s, 'utf8').toString('base64');
  return crypto.createHash('md5').update(b64, 'utf8').digest('hex').toUpperCase();
}

const now = Date.now();
const conditions = {
  hyperOs: 7,
  androidOs: 37,
  model: '2608BPX34C',
  product: 'lhasa',
  protocol: 200,
  appVersion: 402012,
  country: 'CN',
  otaVersion: 'OS4.0.26.0.XPNCNXM',
  soc: 'O3',
  deviceType: 0,
  brand: 'xiaomi',
  deviceLevel: 1,
  rom: 7,
  ram: 12,
  stage,
  gray: 1234,
  businessLine: 'default',
};

const query = { ...conditions, timestamp: now };
query.sign = sign(query);

const headers = {
  model: conditions.model,
  product: conditions.product,
  protocol: 200,
  region: conditions.country,
  hyper_os: conditions.hyperOs,
  android_os: conditions.androidOs,
  ota_version: conditions.otaVersion,
  package_name: 'com.xiaomi.aiservice',
  version_code: 402012,
  version_name: '4.2.12_8560455_260910_cn',
  caller_package_name: 'com.xiaomi.aiservice',
  caller_version_code: 402012,
  caller_version_name: '4.2.12_8560455_260910_cn',
  timestamp: now,
};
headers.sign = sign(headers);

const url = new URL(base);
for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));

const res = await fetch(url, {
  method: 'POST',
  headers: { ...headers, 'Content-Type': 'application/json' },
  body: JSON.stringify({ conditions }),
});
const text = await res.text();
fs.mkdirSync(out.substring(0, out.lastIndexOf('/')) || '.', { recursive: true });
fs.writeFileSync(out, text);
const json = JSON.parse(text);
const configs = json.data?.configList || [];
const resources = json.data?.resourceList || [];
console.log(JSON.stringify({
  status: res.status,
  code: json.code,
  configs: configs.length,
  resources: resources.length,
  llm: configs.filter(c => c.isLLM).map(c => `${c.api}/${c.name}`),
  schemes: [...new Set(resources.map(r => {
    try { return new URL(r.fileUrl).protocol; } catch { return 'invalid'; }
  }))],
  out,
}, null, 2));

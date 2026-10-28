#!/usr/bin/env node
/**
 * Design B injector for com.xiaomi.aiservice (NOTES.md §20.2 / §23).
 *
 * Runs on the HOST against a pulled copy of the Room database `ai_db`
 * (tables: ai_config, local_resource, remote_resource).  It clones the
 * live `vlm` config row and its model local_resource row into a new
 * api/name/id so the official model files stay untouched.
 *
 * Usage:
 *   node inject.js --db ai_db [--remove]
 *                 [--src-api vlm] [--src-name vlm_qwen3_0_7b_f3]
 *                 [--api myllm] [--name myllm_f3] [--model-id myllm_model]
 *                 [--model-version 1] [--model-dir <abs path on device>]
 *                 [--size 0]
 *
 * Rollback (remove injected rows):
 *   node inject.js --db ai_db --remove --api myllm --name myllm_f3 --model-id myllm_model --model-version 1
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) throw new Error("bad arg " + a);
    if (a === "--remove") { out.remove = true; continue; }
    const key = a.slice(2);
    const val = argv[++i];
    if (val === undefined) throw new Error("missing value for " + a);
    out[key] = val;
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
if (!args.db) { console.error("--db required"); process.exit(2); }

const SRC_API = args["src-api"] || "vlm";
const SRC_NAME = args["src-name"] || "vlm_qwen3_0_7b_f3";
const API = args.api || "myllm";
const NAME = args.name || "myllm_f3";
const MODEL_ID = args["model-id"] || "myllm_model";
const MODEL_VER = Number(args["model-version"] || 1);
const MODEL_DIR = args["model-dir"] ||
  "/data/user_de/0/com.xiaomi.aiservice/taiyi/model/" + MODEL_ID + "/" + MODEL_VER;
const SIZE = Number(args.size || 0);

if (!fs.existsSync(args.db)) { console.error("db not found: " + args.db); process.exit(2); }
const db = new DatabaseSync(args.db);

function rows(sql, ...params) {
  return db.prepare(sql).all(...params);
}
function run(sql, ...params) {
  return db.prepare(sql).run(...params);
}

function dumpConfigs() {
  return rows("SELECT api, name, version, disabled, origin FROM ai_config ORDER BY api, name");
}

if (args.remove) {
  const before = rows("SELECT api,name FROM ai_config WHERE api=? AND name=?", API, NAME);
  const beforeRes = rows("SELECT id,version FROM local_resource WHERE id=? AND version=?", MODEL_ID, MODEL_VER);
  run("DELETE FROM ai_config WHERE api=? AND name=?", API, NAME);
  run("DELETE FROM local_resource WHERE id=? AND version=?", MODEL_ID, MODEL_VER);
  console.log("[remove] ai_config deleted:", JSON.stringify(before));
  console.log("[remove] local_resource deleted:", JSON.stringify(beforeRes));
  finish();
  process.exit(0);
}

// ---- 1. locate the source config row --------------------------------------
const src = rows("SELECT * FROM ai_config WHERE api=? AND name=? LIMIT 1", SRC_API, SRC_NAME);
if (src.length === 0) {
  console.error("source config not found: api=" + SRC_API + " name=" + SRC_NAME);
  console.error("available rows:");
  for (const r of dumpConfigs()) console.error("  ", JSON.stringify(r));
  process.exit(1);
}
const srcRow = src[0];

// ---- 2. clone + mutate the config jsonSource -------------------------------
const cfg = JSON.parse(srcRow.jsonSource);
if (cfg.plugin && cfg.plugin.id === cfg.name) {
  // plugin.id happens to equal the old name; it must keep pointing at the
  // installed plugin resource, so rename the config name first and only
  // then set a fresh plugin object.
}
const newCfg = JSON.parse(JSON.stringify(cfg));
newCfg.api = API;
newCfg.name = NAME;
if (Array.isArray(newCfg.models) && newCfg.models.length > 0) {
  newCfg.models[0].id = MODEL_ID;
  newCfg.models[0].version = MODEL_VER;
} else {
  console.error("source config has no models[]; refusing to continue");
  process.exit(1);
}
const newJson = JSON.stringify(newCfg);
JSON.parse(newJson); // self-check: must round-trip

// ---- 3. clone the model local_resource row --------------------------------
const srcRes = rows("SELECT * FROM local_resource WHERE id=? AND version=? LIMIT 1",
  cfg.models[0].id, cfg.models[0].version);
if (srcRes.length === 0) {
  console.error("source local_resource not found: id=" + cfg.models[0].id +
    " version=" + cfg.models[0].version);
  process.exit(1);
}
const resRow = srcRes[0];
const srcResJson = JSON.parse(resRow.jsonSource);

const newResJson = JSON.stringify({
  id: MODEL_ID,
  version: MODEL_VER,
  type: srcResJson.type || "MODEL",
  scheme: srcResJson.scheme || "SANDBOX",
  unwrapped: srcResJson.unwrapped !== false,
  subType: srcResJson.subType || "BASE_LLM",
  size: SIZE,
  path: MODEL_DIR,
});

const now = Date.now();
const collisions = [
  rows("SELECT api,name FROM ai_config WHERE api=? AND name=?", API, NAME),
  rows("SELECT id,version FROM local_resource WHERE id=? AND version=?", MODEL_ID, MODEL_VER),
];
if (collisions[0].length || collisions[1].length) {
  console.error("target rows already exist (will REPLACE):", JSON.stringify(collisions));
}

run("INSERT OR REPLACE INTO ai_config " +
  "(`api`,`name`,`version`,`disabled`,`origin`,`jsonSource`,`modifyTime`," +
  "`appVersionName`,`appVersionCode`,`osVersion`,`androidVersion`,`otaVersion`) " +
  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
  API, NAME, srcRow.version, srcRow.disabled, srcRow.origin, newJson,
  now, srcRow.appVersionName, srcRow.appVersionCode, srcRow.osVersion,
  srcRow.androidVersion, srcRow.otaVersion);

run("INSERT OR REPLACE INTO local_resource " +
  "(`id`,`version`,`type`,`scheme`,`unwrapped`,`subType`,`jsonSource`,`processTime`," +
  "`modifyTime`,`appVersionName`,`appVersionCode`,`osVersion`,`androidVersion`,`otaVersion`) " +
  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
  MODEL_ID, MODEL_VER, srcResJson.type || "MODEL", srcResJson.scheme || "SANDBOX",
  srcResJson.unwrapped === false ? 0 : 1, srcResJson.subType || "BASE_LLM", newResJson,
  now, now, resRow.appVersionName, resRow.appVersionCode, resRow.osVersion,
  resRow.androidVersion, resRow.otaVersion);

// ---- 4. verify + report ----------------------------------------------------
const chkCfg = rows("SELECT * FROM ai_config WHERE api=? AND name=?", API, NAME)[0];
const chkRes = rows("SELECT * FROM local_resource WHERE id=? AND version=?", MODEL_ID, MODEL_VER)[0];
const chk = JSON.parse(chkCfg.jsonSource);
if (chk.api !== API || chk.name !== NAME || chk.models[0].id !== MODEL_ID) {
  console.error("verification FAILED", JSON.stringify(chk));
  process.exit(1);
}
console.log("[ok] ai_config row:");
console.log("      api/name/version:", chkCfg.api, chkCfg.name, chkCfg.version,
  "disabled=" + chkCfg.disabled, "origin=" + chkCfg.origin);
console.log("      plugin kept:", chk.plugin.id + "@" + chk.plugin.version,
  "param kept:", chk.param ? chk.param.id + "@" + chk.param.version : "-");
console.log("      model:", chk.models[0].id + "@" + chk.models[0].version,
  chk.models[0].subType);
console.log("[ok] local_resource row:");
console.log("     ", JSON.parse(chkRes.jsonSource).path);
console.log("      jsonSource:", chkRes.jsonSource);
console.log("[next] PRAGMA checkpoint");
finish();

function finish() {
  try { db.exec("PRAGMA wal_checkpoint(TRUNCATE);"); } catch (e) { /* non-wal db */ }
  db.close();
}

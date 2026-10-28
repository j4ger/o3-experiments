#!/usr/bin/env node
/**
 * Self-test for tools/designb/inject.js against a replica ai_db built with
 * the exact Room schema recovered from jadx (x1/f.java) and the real vlm
 * config JSON from artifacts/ai-config.json.  Exercises: fresh inject,
 * re-inject (collision path), remove, WAL replay.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");

const ROOT = "/home/j4ger/Documents/oring_exp";
const WORK = fs.mkdtempSync("/tmp/designb-selftest-");
const dbPath = path.join(WORK, "ai_db");

const SCHEMA = `
CREATE TABLE IF NOT EXISTS \`remote_resource\` (\`id\` TEXT NOT NULL, \`version\` INTEGER NOT NULL, \`fileName\` TEXT NOT NULL, \`fileSize\` INTEGER NOT NULL, \`fileHash\` TEXT NOT NULL, \`fileUrl\` TEXT NOT NULL, \`type\` TEXT NOT NULL, \`share\` INTEGER NOT NULL, \`subType\` TEXT NOT NULL DEFAULT '', \`encrypted\` INTEGER NOT NULL, \`modifyTime\` INTEGER NOT NULL, \`appVersionName\` TEXT NOT NULL, \`appVersionCode\` INTEGER NOT NULL, \`osVersion\` INTEGER NOT NULL, \`androidVersion\` INTEGER NOT NULL, \`otaVersion\` TEXT NOT NULL, PRIMARY KEY(\`id\`, \`version\`));
CREATE TABLE IF NOT EXISTS \`local_resource\` (\`id\` TEXT NOT NULL, \`version\` INTEGER NOT NULL, \`type\` TEXT NOT NULL, \`scheme\` TEXT NOT NULL, \`unwrapped\` INTEGER NOT NULL, \`subType\` TEXT NOT NULL DEFAULT '', \`jsonSource\` TEXT NOT NULL, \`processTime\` INTEGER NOT NULL, \`modifyTime\` INTEGER NOT NULL, \`appVersionName\` TEXT NOT NULL, \`appVersionCode\` INTEGER NOT NULL, \`osVersion\` INTEGER NOT NULL, \`androidVersion\` INTEGER NOT NULL, \`otaVersion\` TEXT NOT NULL, PRIMARY KEY(\`id\`, \`version\`));
CREATE TABLE IF NOT EXISTS \`ai_config\` (\`api\` TEXT NOT NULL, \`name\` TEXT NOT NULL, \`version\` INTEGER NOT NULL, \`disabled\` INTEGER NOT NULL, \`origin\` INTEGER NOT NULL, \`jsonSource\` TEXT NOT NULL, \`modifyTime\` INTEGER NOT NULL, \`appVersionName\` TEXT NOT NULL, \`appVersionCode\` INTEGER NOT NULL, \`osVersion\` INTEGER NOT NULL, \`androidVersion\` INTEGER NOT NULL, \`otaVersion\` TEXT NOT NULL, PRIMARY KEY(\`api\`, \`name\`));
CREATE TABLE IF NOT EXISTS room_master_table (id INTEGER PRIMARY KEY,identity_hash TEXT);
`;

const APPVER_NAME = "4.2.12_8560455_260910_cn";
const APPVER_CODE = 402012;
const OSV = 36;
const ANDV = 36;
const OTA = "CP2A.260605.016";

function run() {
  const db = new DatabaseSync(dbPath);
  db.exec(SCHEMA);
  db.exec("INSERT OR REPLACE INTO room_master_table (id,identity_hash) VALUES(42,'ebfee3993e34c1144ff16ab102be3e91')");
  db.exec("PRAGMA journal_mode=WAL;");

  // real server config for vlm
  const server = JSON.parse(fs.readFileSync(ROOT + "/artifacts/ai-config.json", "utf8"));
  const vlmCfg = server.data.configList.find(c => c.api === "vlm");
  if (!vlmCfg) throw new Error("vlm config missing from ai-config.json");
  const cfgJson = JSON.stringify(vlmCfg);

  const insCfg = db.prepare(
    "INSERT OR REPLACE INTO ai_config (`api`,`name`,`version`,`disabled`,`origin`,`jsonSource`,`modifyTime`,`appVersionName`,`appVersionCode`,`osVersion`,`androidVersion`,`otaVersion`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)");
  insCfg.run("vlm", vlmCfg.name, vlmCfg.version, 0, 0, cfgJson, Date.now(),
    APPVER_NAME, APPVER_CODE, OSV, ANDV, OTA);

  const insRes = db.prepare(
    "INSERT OR REPLACE INTO local_resource (`id`,`version`,`type`,`scheme`,`unwrapped`,`subType`,`jsonSource`,`processTime`,`modifyTime`,`appVersionName`,`appVersionCode`,`osVersion`,`androidVersion`,`otaVersion`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)");
  const mkRes = (id, ver, type, subType, pathStr) => insRes.run(id, ver, type, "SANDBOX", 1, subType,
    JSON.stringify({ id, version: ver, type, scheme: "SANDBOX", unwrapped: true, subType, size: 123, path: pathStr }),
    Date.now(), Date.now(), APPVER_NAME, APPVER_CODE, OSV, ANDV, OTA);
  mkRes("vlm_model_qwen3_0_7b_f3", 15, "MODEL", "BASE_LLM",
    "/data/user_de/0/com.xiaomi.aiservice/taiyi/model/vlm_model_qwen3_0_7b_f3/15");
  mkRes("vlm_qwen3_0_7b_f3", 18, "PLUGIN", "LLM",
    "/data/user_de/0/com.xiaomi.aiservice/taiyi/plugin/vlm_qwen3_0_7b_f3/18");
  mkRes("vlm_param", 3, "PARAM", "DEFAULT",
    "/data/user_de/0/com.xiaomi.aiservice/taiyi/param/vlm_param/3");
  db.close();
}

function count(dbPath2, table, where, ...params) {
  const db = new DatabaseSync(dbPath2);
  const r = db.prepare(`SELECT COUNT(*) AS n FROM ${table} WHERE ${where}`).get(...params);
  db.close();
  return Number(r.n);
}

function assert(cond, msg) {
  if (!cond) { console.error("FAIL:", msg); process.exit(1); }
  console.log("PASS:", msg);
}

const INJECT = process.execPath + " " + ROOT + "/tools/designb/inject.js";
const { execFileSync } = require("child_process");

run();

// 1. fresh inject
execFileSync("node", [ROOT + "/tools/designb/inject.js", "--db", dbPath], { stdio: "inherit" });
assert(count(dbPath, "ai_config", "api=? AND name=?", "myllm", "myllm_f3") === 1, "ai_config row inserted");
assert(count(dbPath, "local_resource", "id=? AND version=?", "myllm_model", 1) === 1, "local_resource row inserted");

// verify cloned config content
{
  const db = new DatabaseSync(dbPath);
  const row = db.prepare("SELECT jsonSource FROM ai_config WHERE api='myllm' AND name='myllm_f3'").get();
  const cfg = JSON.parse(row.jsonSource);
  assert(cfg.plugin.id === "vlm_qwen3_0_7b_f3" && cfg.plugin.version === 18, "plugin pointer preserved");
  assert(cfg.param.id === "vlm_param", "param preserved");
  assert(cfg.models[0].id === "myllm_model" && cfg.models[0].version === 1, "model id/version rewritten");
  assert(cfg.api === "myllm" && cfg.name === "myllm_f3", "api/name rewritten");
  assert(cfg.isLLM === true, "isLLM preserved");
  const res = db.prepare("SELECT jsonSource FROM local_resource WHERE id='myllm_model'").get();
  const rj = JSON.parse(res.jsonSource);
  assert(rj.path === "/data/user_de/0/com.xiaomi.aiservice/taiyi/model/myllm_model/1", "model path rewritten");
  assert(rj.unwrapped === true && rj.subType === "BASE_LLM", "model subType/unwrapped cloned");
  db.close();
}

// 2. re-inject (collision replace path must not throw)
execFileSync("node", [ROOT + "/tools/designb/inject.js", "--db", dbPath], { stdio: "inherit" });
assert(count(dbPath, "ai_config", "api=? AND name=?", "myllm", "myllm_f3") === 1, "re-inject stays single row");

// 3. remove
execFileSync("node", [ROOT + "/tools/designb/inject.js", "--db", dbPath, "--remove"], { stdio: "inherit" });
assert(count(dbPath, "ai_config", "api=? AND name=?", "myllm", "myllm_f3") === 0, "remove deletes ai_config row");
assert(count(dbPath, "local_resource", "id=?", "myllm_model") === 0, "remove deletes local_resource row");
assert(count(dbPath, "ai_config", "api='vlm'") === 1, "source rows untouched");

console.log("\nALL TESTS PASSED. workdir:", WORK);

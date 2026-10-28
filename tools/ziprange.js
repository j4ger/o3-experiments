#!/usr/bin/env node
/**
 * List entries of a remote ZIP via HTTP Range requests (no full download),
 * and optionally extract a single entry to stdout/file.
 *
 *   node ziprange.js <url> [--extract <entry>] [--out <file>] [--peek <entry>=<n>]
 */
"use strict";
const zlib = require("zlib");
const fs = require("fs");

const url = process.argv[2];
if (!url) { console.error("usage: ziprange.js <url> [--extract <entry>] [--out <file>] [--peek <entry>=<n>]"); process.exit(2); }
const args = process.argv.slice(3);
function opt(name) { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; }
const extractEntry = opt("--extract");
const outFile = opt("--out");
const peek = opt("--peek");

async function head() {
  const r = await fetch(url, { method: "HEAD" });
  const len = Number(r.headers.get("content-length"));
  const ranges = r.headers.get("accept-ranges");
  if (!Number.isFinite(len)) throw new Error("no content-length");
  return { len, ranges };
}
async function rng(a, b) {
  const r = await fetch(url, { headers: { Range: "bytes=" + a + "-" + b } });
  if (r.status !== 206) throw new Error("range unsupported: " + r.status);
  return Buffer.from(await r.arrayBuffer());
}

(async () => {
  const { len, ranges } = await head();
  const tail = Math.min(655360, len);
  const start = len - tail;
  const buf = await rng(start, len - 1);
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error("EOCD not found (not a zip?); accept-ranges=" + ranges);
  let cdEntries = buf.readUInt16LE(eocd + 10);
  const cdSize = buf.readUInt32LE(eocd + 12);
  let cdOfs = buf.readUInt32LE(eocd + 16);
  let cdBuf, cdStart;
  if (cdOfs === 0xffffffff || cdEntries === 0xffff) {
    let loc = -1;
    for (let i = eocd - 20; i >= 0; i--) if (buf.readUInt32LE(i) === 0x07064b50) { loc = i; break; }
    if (loc < 0) throw new Error("zip64 locator missing");
    const z64 = Number(buf.readBigUInt64LE(loc + 8));
    const r = await rng(z64, z64 + 55);
    cdEntries = Number(r.readBigUInt64LE(32));
    cdOfs = Number(r.readBigUInt64LE(48));
  }
  cdStart = cdOfs - start;
  if (cdStart < 0 || cdStart + cdSize > buf.length) cdBuf = await rng(cdOfs, cdOfs + cdSize - 1), cdStart = 0;
  else cdBuf = buf;
  const entries = {};
  let p = cdStart, total = 0, comp = 0;
  for (let i = 0; i < cdEntries; i++) {
    if (cdBuf.readUInt32LE(p) !== 0x02014b50) break;
    const method = cdBuf.readUInt16LE(p + 10);
    const csize = cdBuf.readUInt32LE(p + 20);
    const usize = cdBuf.readUInt32LE(p + 24);
    const nlen = cdBuf.readUInt16LE(p + 28);
    const elen = cdBuf.readUInt16LE(p + 30);
    const clen = cdBuf.readUInt16LE(p + 32);
    const lho = cdBuf.readUInt32LE(p + 42);
    const name = cdBuf.slice(p + 46, p + 46 + nlen).toString();
    entries[name] = { method, csize, usize, lho };
    total += usize; comp += csize;
    p += 46 + nlen + elen + clen;
  }
  console.error("file size: " + len + "  entries: " + cdEntries + "  uncompressed total: " + total + "  compressed: " + comp);
  for (const [name, e] of Object.entries(entries)) {
    console.log(String(e.usize).padStart(12) + "  " + String(e.csize).padStart(12) + "  " +
      (e.method === 0 ? "stored " : e.method === 8 ? "deflate" : "m" + e.method) + "  " + name);
  }
  async function readEntry(name, maxOut) {
    const e = entries[name];
    if (!e) throw new Error("entry not found: " + name);
    const h = await rng(e.lho, e.lho + 30 + 512 - 1);
    const nlen2 = h.readUInt16LE(26), elen2 = h.readUInt16LE(28);
    const dataOfs = e.lho + 30 + nlen2 + elen2;
    const c = await rng(dataOfs, dataOfs + e.csize - 1);
    const out = e.method === 8 ? zlib.inflateRawSync(c) : c;
    return maxOut ? out.subarray(0, maxOut) : out;
  }
  if (peek) {
    const [name, n] = peek.split("=");
    const o = await readEntry(name, Number(n || 256));
    process.stdout.write("\n--- " + name + " (first " + o.length + " bytes) ---\n");
    process.stdout.write(o.toString("hex").match(/../g).join(" ") + "\n");
    process.stdout.write(JSON.stringify(o.toString("latin1").replace(/[^\x20-\x7e]/g, ".")) + "\n");
  }
  if (extractEntry) {
    const o = await readEntry(extractEntry);
    if (outFile) { fs.writeFileSync(outFile, o); console.error("wrote " + outFile + " (" + o.length + " bytes)"); }
    else process.stdout.write(o);
  }
})().catch(e => { console.error("ERR " + e.message); process.exit(1); });
#!/usr/bin/env node
/*
 * Parse an XNLM 0.5.0 layer block into per-tensor records.
 *
 *   node record.js <model.bin> --layer 0
 *
 * Layout recovered from Ghidra's decompilation of WeightImpl::parseLayer and
 * verified by arithmetic closure (payload + size == next record offset):
 *
 *   [u64 nameLen][name ^ magic]
 *   [XnlmQuantParam prefill 0x28][scale bytes][zp bytes]
 *   [XnlmQuantParam decode 0x28]
 *   [per-channel FP32 scales][groupwise scale arrays (UINT8 prefill, INT8 decode)]
 *   [64-byte-header data blocks: {u64 size; u32 kind; ...} + data, 64-aligned]
 *   [64-byte-header weight block {u64 size; u32 kind}]
 *   [4096-byte reservation][payload: size bytes]
 *
 * The parser does not assume how many data blocks precede the weight block; it
 * finds the weight header by the closure payload+size == record end.
 */
'use strict';
const fs = require('fs');

const file = process.argv[2];
if (!file) {
    console.error('usage: record.js <model.bin> [--layer N] [--json]');
    process.exit(2);
}
const layerArg = process.argv.includes('--layer')
    ? parseInt(process.argv[process.argv.indexOf('--layer') + 1], 10)
    : 0;

const fd = fs.openSync(file, 'r');
const magic = Buffer.alloc(16);
fs.readSync(fd, magic, 0, 16, 0);
const metaLen = Number(readU64(fd, 16));
const tableOff = 24 + metaLen;
const sizes = [];
{
    let p = tableOff, seen = false;
    for (;;) {
        const v = Number(readU64(fd, p));
        if (v === 0) { if (seen) break; } else { seen = true; sizes.push(v); }
        p += 8;
    }
}
if (layerArg >= sizes.length) {
    console.error(`layer ${layerArg} out of range (0..${sizes.length - 1})`);
    process.exit(2);
}
const blockOff = 4096 + sizes.slice(0, layerArg).reduce((a, b) => a + b, 0);
const blockSize = sizes[layerArg];
const blk = Buffer.alloc(blockSize);
fs.readSync(fd, blk, 0, blockSize, blockOff);

function readU64(f, off) {
    const b = Buffer.alloc(8);
    fs.readSync(f, b, 0, 8, off);
    return b.readBigUInt64LE(0);
}
const align = (x, n) => Math.ceil(x / n) * n;

// --- locate records by decrypting names -----------------------------------
const records = [];
for (let p = 0; p + 16 < blk.length; p++) {
    const n = Number(blk.readBigUInt64LE(p));
    if (n < 8 || n > 80 || p + 8 + n > blk.length) continue;
    const name = Buffer.alloc(n);
    let ok = true;
    for (let i = 0; i < n; i++) {
        const c = blk[p + 8 + i] ^ magic[i % 16];
        if (!((c >= 97 && c <= 122) || (c >= 65 && c <= 90) || (c >= 48 && c <= 57) || c === 46 || c === 95)) {
            ok = false; break;
        }
        name[i] = c;
    }
    if (!ok) continue;
    const nm = name.toString('latin1');
    if (!/^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$/.test(nm)) continue;
    records.push({ start: p, nameLen: n, name: nm });
}
records.sort((a, b) => a.start - b.start);

// --- for each record, find the weight block by closure --------------------
const out = [];
for (let i = 0; i < records.length; i++) {
    const rec = records[i];
    const end = i + 1 < records.length ? records[i + 1].start : blockSize;
    const bodyEnd = end;
    let weight = null;
    for (let q = rec.start; q + 16 < bodyEnd; q += 8) {
        const size = Number(blk.readBigUInt64LE(q));
        if (size < 1024 || size > bodyEnd) continue;
        const payload = q + 4096;
        if (payload + size === bodyEnd) {
            weight = { header: q, size, chunks: blk.readUInt32LE(q + 8), payload };
            break;
        }
    }
    out.push({
        name: rec.name,
        start: rec.start,
        end,
        span: end - rec.start,
        weight
    });
}

if (process.argv.includes('--json')) {
    console.log(JSON.stringify({ blockOff, blockSize, records: out }, null, 2));
    process.exit(0);
}
console.log(`file ${file}  layer ${layerArg}  block @${blockOff} size=${blockSize}  records=${out.length}\n`);
for (const r of out) {
    const w = r.weight;
    console.log(`${r.name}`);
    console.log(`  record @${r.start} span=${r.span}`);
    if (w) {
        console.log(`  weight header @${w.header} size=${w.size} chunks=${w.chunks}`);
        console.log(`  payload @${w.payload}  (${w.size} B, ${(w.size * 8 / 2).toFixed(0)} int4 params)  closure=${w.payload + w.size === r.end ? 'exact' : 'MISMATCH'}`);
    } else {
        console.log('  weight block: not found');
    }
}
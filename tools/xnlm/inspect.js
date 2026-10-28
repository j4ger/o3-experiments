#!/usr/bin/env node
/*
 * XNLM container inspector (0.4.0 and 0.5.0).
 *
 *   node inspect.js <file>                     header + decrypted meta
 *   node inspect.js <file> --layer 0            tensor records of one layer block
 *   node inspect.js <file> --layer 0 --json     machine-readable
 *   node inspect.js <file> --meta-only          works on a small prefix
 *
 * Format facts established against the shipped models (see docs/FORMAT-xnlm.md):
 *
 *   [0..16)    magic "XNLM<ver>:<tag>"
 *   [16..24)   u64 metaLen
 *   [24..24+L) meta blob.
 *                0.4.0: plaintext JSON
 *                0.5.0: JSON XOR-encrypted with the 16-byte magic itself
 *     The JSON carries the architecture (hidden size, layers, heads, kv heads,
 *     head_dim, vocab, intermediate, rope/SWA, qk-norm flags, weight_compress).
 *   [..]       u64 block-size table: 0, size0, … terminated by 0
 *   [..4096)   zero padding
 *   [4096..)   blocks; 0.5.0 block = per-tensor records
 *                [u64 nameLen][name bytes]   (name XOR-encrypted in 0.5.0)
 *                [descriptor]                (dtype/size/quant fields)
 *                [tensor payload]
 */
'use strict';
const fs = require('fs');

function xor(buf, key, off = 0) {
    const out = Buffer.alloc(buf.length);
    for (let i = 0; i < buf.length; i++) {
        out[i] = buf[i] ^ key[(off + i) % key.length];
    }
    return out;
}

function readChunk(file, offset, length) {
    const fd = fs.openSync(file, 'r');
    try {
        const b = Buffer.alloc(length);
        fs.readSync(fd, b, 0, length, offset);
        return b;
    } finally {
        fs.closeSync(fd);
    }
}

function main() {
    const file = process.argv[2];
    if (!file) {
        console.error('usage: inspect.js <file> [--meta-only] [--layer N] [--json]');
        process.exit(2);
    }
    const size = fs.statSync(file).size;
    const head = readChunk(file, 0, Math.min(size, 64 * 1024));
    const magic = head.subarray(0, 16).toString('latin1');
    const metaLen = Number(head.readBigUInt64LE(16));
    const metaRaw = head.subarray(24, 24 + metaLen);

    let meta = null, metaState = 'unknown';
    const asText = metaRaw.toString('utf8');
    if (asText.trimStart().startsWith('{')) {
        metaState = 'plaintext-json';
        meta = asText;
    } else {
        const dec = xor(metaRaw, Buffer.from(magic, 'latin1')).toString('utf8');
        if (dec.trimStart().startsWith('{')) {
            metaState = 'xor-json (key = 16-byte magic)';
            meta = dec;
        }
    }

    const out = { file, size, magic, metaLen, metaState };
    try {
        out.config = JSON.parse(meta);
    } catch (e) {
        out.config = null;
    }

    // block table
    const t0 = 24 + metaLen;
    const sizes = [];
    let p = t0, sawNonZero = false;
    while (p + 8 <= head.length) {
        const v = Number(head.readBigUInt64LE(p));
        if (v === 0) {
            if (sawNonZero) break;
        } else {
            sawNonZero = true;
            sizes.push(v);
        }
        p += 8;
    }
    out.blockTableOffset = t0;
    out.blocks = sizes;
    out.blockSum = sizes.reduce((a, b) => a + b, 0);
    out.expectedSize = 4096 + out.blockSum;

    if (!process.argv.includes('--meta-only') && out.config && sizes.length) {
        const layerIdx = process.argv.includes('--layer')
            ? parseInt(process.argv[process.argv.indexOf('--layer') + 1], 10)
            : null;
        if (layerIdx !== null) {
            const layerOff = 4096 + sizes.slice(0, layerIdx).reduce((a, b) => a + b, 0);
            const layerSize = sizes[layerIdx];
            if (layerOff + layerSize <= size) {
                const blk = readChunk(file, layerOff, layerSize);
                out.layer = { index: layerIdx, offset: layerOff, size: layerSize, records: [] };
                const key = Buffer.from(magic, 'latin1');
                const encrypted = out.metaState !== 'plaintext-json';
                for (let q = 0; q + 8 < blk.length; q++) {
                    const n = Number(blk.readBigUInt64LE(q));
                    if (n < 8 || n > 80 || q + 8 + n > blk.length) continue;
                    let name = blk.subarray(q + 8, q + 8 + n);
                    if (encrypted) {
                        name = xor(name, key);
                    }
                    const s = name.toString('latin1');
                    // payload bytes can accidentally decrypt to the magic itself;
                    // keep only plausible tensor names and never the magic string
                    if (s === magic || s.includes('XNLM')) continue;
                    if (!/^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$/.test(s)) continue;
                    const dtype = blk.readUInt32LE(q + 8 + n);
                    const bytes = blk.readUInt32LE(q + 8 + n + 4);
                    out.layer.records.push({ offset: q, name: s, dtype, bytes });
                    q += 8 + n - 1;
                }
            } else {
                out.layer = { index: layerIdx, error: 'block not present in this (partial) file' };
            }
        }
    }

    if (process.argv.includes('--json')) {
        console.log(JSON.stringify(out, null, 2));
        return;
    }
    console.log(`${file}  ${size} B`);
    console.log(`  magic     ${JSON.stringify(magic)}`);
    console.log(`  metaLen   ${metaLen}  (${metaState})`);
    if (out.config) {
        for (const [k, v] of Object.entries(out.config)) {
            console.log(`    ${k} = ${Array.isArray(v) ? '[' + v.join(',') + ']' : v}`);
        }
    }
    console.log(`  blockTable@${t0}: ${sizes.length} blocks, sum=${out.blockSum}`);
    const uniq = new Map();
    for (const s of sizes) uniq.set(s, (uniq.get(s) || 0) + 1);
    if (sizes.length) {
        console.log(`    ${[...uniq.entries()].map(([s, c]) => `${s}x${c}`).join(', ')}`);
    }
    console.log(`  size check: ${size === out.expectedSize ? 'OK' : 'partial/MISMATCH (' + (size - out.expectedSize) + ')'}`);
    if (out.layer && out.layer.records) {
        console.log(`  layer ${out.layer.index} @${out.layer.offset} (${out.layer.size} B): ${out.layer.records.length} records`);
        let prev = null;
        for (const r of out.layer.records) {
            const span = prev === null ? '' : `  span=${r.offset - prev}`;
            console.log(`    +${String(r.offset).padStart(8)}  dtype=${r.dtype} bytes=${r.bytes}  ${r.name}${span}`);
            prev = r.offset;
        }
    }
}

main();
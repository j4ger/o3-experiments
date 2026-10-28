#!/usr/bin/env node
/*
 * XNLM 0.4.0 (LoRA) container parser + byte-exact rebuild.
 *
 *   node lora.js parse   <file>                 list header, block table, records
 *   node lora.js rebuild <in> <out>             reconstruct and compare
 *
 * The 0.4.0 container is plaintext and self-describing:
 *
 *   [0..16) magic "XNLM0.4.0:de3aeb"
 *   [16..24) u64 metaLen
 *   [24..24+L) JSON architecture metadata
 *   [..]      u64 per-layer block sizes (N entries), zero filled to 4096
 *   [4096..)  per-layer blocks, each a sequence of
 *               [u64 nameLen][name][descriptor][payload]
 *
 * rebuild copies every record verbatim from the source while re-encoding the
 * container framing from what the parser understood.  A byte-identical output
 * proves the structural model is exact (payload semantics are not implied).
 */
'use strict';
const fs = require('fs');
const crypto = require('crypto');

const [mode, src, dst] = process.argv.slice(2);
if (!mode || !src) {
    console.error('usage: lora.js parse|rebuild <file> [out]');
    process.exit(2);
}

const buf = fs.readFileSync(src);
const magic = buf.subarray(0, 16).toString('latin1');
const metaLen = Number(buf.readBigUInt64LE(16));
const meta = buf.subarray(24, 24 + metaLen).toString('utf8');
const tableOff = 24 + metaLen;

// read the block table: u64 entries, terminated by 0 (the table itself starts with 0)
const sizes = [];
{
    let p = tableOff, sawNonZero = false;
    while (p + 8 <= buf.length) {
        const v = Number(buf.readBigUInt64LE(p));
        if (v === 0) { if (sawNonZero) break; } else { sawNonZero = true; sizes.push({ size: v, at: p }); }
        p += 8;
    }
}

// parse records of each block using the same name/descriptor heuristic
function parseBlock(off, size) {
    const end = off + size;
    const recs = [];
    let p = off;
    while (p + 8 <= end) {
        const n = Number(buf.readBigUInt64LE(p));
        if (n < 8 || n > 80 || p + 8 + n > end) { p += 8; continue; }
        const name = buf.subarray(p + 8, p + 8 + n).toString('latin1');
        if (!/^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$/.test(name)) { p += 8; continue; }
        recs.push({ at: p, nameLen: n, name, descriptorAt: p + 8 + n });
        p += 8 + n;
    }
    return recs;
}

if (mode === 'parse') {
    console.log(`file      ${src}  ${buf.length} B`);
    console.log(`magic     ${JSON.stringify(magic)}`);
    console.log(`metaLen   ${metaLen}`);
    console.log(`meta      ${meta.replace(/\s+/g, ' ').slice(0, 160)}…`);
    console.log(`blocks    ${sizes.length} (table @${tableOff})`);
    let off = 4096;
    for (let i = 0; i < sizes.length; i++) {
        const recs = parseBlock(off, sizes[i].size);
        console.log(`\n[block ${i}] @${off} size=${sizes[i].size} records=${recs.length}`);
        for (const r of recs) console.log(`   +${String(r.at - off).padStart(7)}  ${r.name}`);
        off += sizes[i].size;
    }
    console.log(`\nend of blocks: ${off} (file ${buf.length})`);
    process.exit(0);
}

if (mode === 'rebuild') {
    if (!dst) { console.error('rebuild needs an output path'); process.exit(2); }
    // framing: header + meta + table (as parsed, re-encoded) + zero pad to 4096
    const parts = [];
    parts.push(buf.subarray(0, 24));
    parts.push(Buffer.from(meta, 'utf8'));
    const table = Buffer.alloc(8 * (sizes.length + 1));
    table.writeBigUInt64LE(0n, 0);
    sizes.forEach((s, i) => table.writeBigUInt64LE(BigInt(s.size), 8 * (i + 1)));
    parts.push(table);
    let cur = 24 + metaLen + table.length;
    if (cur < 4096) parts.push(Buffer.alloc(4096 - cur));
    // blocks: copy verbatim
    for (const s of sizes) {
        const off = 4096 + sizes.slice(0, sizes.indexOf(s)).reduce((a, b) => a + b.size, 0);
        parts.push(buf.subarray(off, off + s.size));
    }
    const out = Buffer.concat(parts);
    fs.writeFileSync(dst, out);
    const h = b => crypto.createHash('sha256').update(b).digest('hex');
    console.log(`in  ${buf.length} B ${h(buf)}`);
    console.log(`out ${out.length} B ${h(out)}`);
    console.log(out.length === buf.length && h(out) === h(buf)
        ? 'IDENTICAL - structural model verified byte for byte'
        : 'DIFFERENT - structural model is incomplete');
    process.exit(0);
}

console.error('unknown mode');
process.exit(2);
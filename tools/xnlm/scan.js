#!/usr/bin/env node
/*
 * Structural profile of an XNLM weight file (or any large binary).
 *
 *   node scan.js <file> [windowMB] [--fp16runs] [--regions]
 *
 * Per window it reports: byte entropy, share of bytes whose two nibbles are
 * both < 8 (int4 packing signature), share of "-8..7 / -128..-121" bytes (int8
 * near-zero signature), and the longest run of plausible FP16 values.  Region
 * boundaries are where the profile changes, which is how the container layout
 * (packed weights / scale arrays / fp16 tensors / metadata) becomes visible.
 */
'use strict';
const fs = require('fs');

function f16(b, o) {
    const v = b.readUInt16LE(o);
    const s = (v & 0x8000) ? -1 : 1;
    const e = (v >> 10) & 0x1f;
    const m = v & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return NaN;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

function plausibleF16(v) {
    return Number.isFinite(v) && Math.abs(v) > 1e-4 && Math.abs(v) < 1e3;
}

function profile(buf) {
    const h = new Array(256).fill(0);
    let nib = 0, narrow = 0, fp16max = 0, fp16cur = 0;
    for (let i = 0; i < buf.length; i++) {
        const v = buf[i];
        h[v]++;
        if ((v >> 4) < 8 && (v & 15) < 8) nib++;
        if (v <= 8 || v >= 248) narrow++;
        if (i % 2 === 0) {
            if (plausibleF16(f16(buf, i))) {
                fp16cur++;
                if (fp16cur > fp16max) fp16max = fp16cur;
            } else fp16cur = 0;
        }
    }
    let ent = 0;
    for (const c of h) {
        if (c) { const p = c / buf.length; ent -= p * Math.log2(p); }
    }
    return {
        entropy: ent,
        nibble: nib / buf.length,
        narrow: narrow / buf.length,
        fp16run: fp16max
    };
}

function fmt(p) {
    return `H=${p.entropy.toFixed(2)} nib=${(p.nibble * 100).toFixed(0)}% ` +
        `narrow=${(p.narrow * 100).toFixed(0)}% fp16run=${p.fp16run}`;
}

function main() {
    const file = process.argv[2];
    const winMB = parseInt(process.argv[3] || '4', 10);
    const wantRuns = process.argv.includes('--fp16runs');
    const win = winMB << 20;
    const st = fs.statSync(file);
    const fd = fs.openSync(file, 'r');
    console.log(`${file}  ${st.size} bytes  window=${winMB}MiB`);
    let prev = null, prevStart = 0;
    const buf = Buffer.alloc(win);
    for (let pos = 0; pos < st.size; pos += win) {
        const n = fs.readSync(fd, buf, 0, Math.min(win, st.size - pos), pos);
        const p = profile(buf.subarray(0, n));
        const sig = `${p.entropy.toFixed(1)}|${(p.nibble * 100).toFixed(0)}|${(p.narrow * 100).toFixed(0)}|${Math.min(p.fp16run, 8)}`;
        if (sig !== prev) {
            if (prev !== null) {
                console.log(`  [${prevStart} .. ${pos})  ${fmt(lastP)}`);
            }
            prev = sig; prevStart = pos;
        }
        var lastP = p;
    }
    if (prev !== null) console.log(`  [${prevStart} .. ${st.size})  ${fmt(lastP)}`);

    if (wantRuns) {
        // longest plausible-FP16 runs across the whole file (scale arrays)
        const chunk = 8 << 20;
        const b = Buffer.alloc(chunk);
        let cur = 0, curStart = 0, best = [];
        let base = 0, carry = Buffer.alloc(0);
        for (let pos = 0; pos < st.size; pos += chunk) {
            const n = fs.readSync(fd, b, 0, Math.min(chunk, st.size - pos), pos);
            const hay = Buffer.concat([carry, b.subarray(0, n)]);
            const hayBase = pos - carry.length;
            for (let i = 0; i + 2 <= hay.length; i += 2) {
                if (plausibleF16(f16(hay, i))) {
                    if (cur === 0) curStart = hayBase + i;
                    cur++;
                    if (cur > 32) best.push({ start: curStart, count: cur });
                } else cur = 0;
            }
            carry = hay.subarray(Math.max(0, hay.length - 4));
            base += n;
        }
        const merged = [];
        for (const r of best) {
            const last = merged[merged.length - 1];
            if (last && r.start <= last.start + last.count * 2 + 4) {
                last.count = Math.max(last.count, (r.start - last.start) / 2 + r.count);
            } else merged.push({ ...r });
        }
        merged.sort((a, b2) => b2.count - a.count);
        console.log('top FP16-plausible runs (offset, values, bytes):');
        for (const r of merged.slice(0, 10)) {
            console.log(`  @${r.start}  ${r.count} values  ${r.count * 2} B`);
        }
    }
}

main();
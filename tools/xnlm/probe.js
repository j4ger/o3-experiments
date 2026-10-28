#!/usr/bin/env node
/*
 * Probe the quantized payload of one XNLM 0.5.0 tensor against a HuggingFace
 * reference tensor (ground truth for the same architecture).
 *
 *   node probe.js <xnlm.bin> <hf_tensor.bf16> <tensorRegion> [--group 64]
 *
 * tensorRegion: "<blockOffset>:<recordOffset>:<recordEnd>" in bytes of the
 * whole file, e.g. layer0 q_proj = "4096:0:1227840" (block 0 starts at 4096,
 * record at +0, next record at +1227840).
 *
 * What it establishes (and what it cannot):
 *   - the FP16 scale array of the region: its value multiset correlates with the
 *     per-group amax of the HF tensor (log-corr ~0.96), i.e. it IS that tensor's
 *     group scale array;
 *   - the scale *order* does not correlate with row-major groups, so the payload
 *     is written in a permuted (NPU-specific) order;
 *   - scale value / amax ≈ 1/100 while int4 qmax 8 gives only amax/12.5, so a
 *     further global factor is still unexplained.
 */
'use strict';
const fs = require('fs');

const [file, hfFile, region] = process.argv.slice(2);
if (!file || !hfFile || !region) {
    console.error('usage: probe.js <xnlm.bin> <hf_bf16> <block:recordStart:recordEnd> [--group 64]');
    process.exit(2);
}
const groupArg = process.argv.includes('--group')
    ? parseInt(process.argv[process.argv.indexOf('--group') + 1], 10)
    : 64;
const [blockOff, recStart, recEnd] = region.split(':').map(Number);

function f16(b, o) {
    const x = b.readUInt16LE(o);
    const s = (x & 0x8000) ? -1 : 1;
    const e = (x >> 10) & 0x1f;
    const m = x & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return NaN;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

function corr(a, b) {
    const n = Math.min(a.length, b.length);
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let sa = 0, sb = 0, sab = 0;
    for (let i = 0; i < n; i++) {
        const x = a[i] - ma, y = b[i] - mb;
        sa += x * x; sb += y * y; sab += x * y;
    }
    const d = Math.sqrt(sa * sb);
    return d ? sab / d : 0;
}

const blk = fs.readFileSync(file).subarray(blockOff + recStart, blockOff + recEnd);
const hf = fs.readFileSync(hfFile);
const hfVal = o => {
    const v = hf.readUInt16LE(o) << 16;
    return new Float32Array(new Uint32Array([v]).buffer)[0];
};

// locate the FP16 scale array: longest run of plausible FP16 values
let best = { start: 0, len: 0 }, cur = 0, start = 0;
for (let o = 0; o + 2 <= blk.length; o += 2) {
    const v = f16(blk, o);
    if (Number.isFinite(v) && Math.abs(v) > 1e-6 && Math.abs(v) < 1e3) {
        if (cur === 0) start = o;
        cur++;
        if (cur > best.len) best = { start, len: cur };
    } else cur = 0;
}
console.log(`region bytes ${blk.length}; scale run @+${best.start} len=${best.len} (${(best.len * 2)} B)`);

// HF per-group amax assuming row-major groups of `group` along the columns
const cols = 1024, rows = 2048;
if (hf.length !== rows * cols * 2) {
    console.log(`note: HF tensor is ${hf.length} B, expected ${rows * cols * 2} for 2048x1024`);
}
const nGroups = Math.floor(rows * cols / groupArg);
const amax = new Float64Array(nGroups);
for (let g = 0; g < nGroups; g++) {
    let m = 0;
    for (let k = 0; k < groupArg; k++) {
        const v = Math.abs(hfVal((g * groupArg + k) * 2));
        if (v > m) m = v;
    }
    amax[g] = m;
}
const nScale = Math.min(best.len, nGroups);
const sc = new Float64Array(nScale);
for (let i = 0; i < nScale; i++) sc[i] = f16(blk, best.start + i * 2);

const ratioMed = (() => {
    const r = [];
    for (let i = 0; i < nScale; i++) if (amax[i] > 0) r.push(amax[i] / sc[i]);
    r.sort((a, b) => a - b);
    return r[r.length >> 1];
})();

console.log(`scale[0..3]  ${[...sc.slice(0, 4)].map(x => x.toExponential(3)).join(' ')}`);
console.log(`amax[0..3]   ${[...amax.slice(0, 4)].map(x => x.toExponential(3)).join(' ')}`);
console.log(`ordered log-corr      ${corr(Array.from(sc).map(Math.log), Array.from(amax.slice(0, nScale)).map(Math.log)).toFixed(4)}  (≈0 → permuted order)`);
const sortedS = Array.from(sc).sort((a, b) => a - b).map(Math.log);
const sortedA = Array.from(amax).slice(0, nScale).sort((a, b) => a - b).map(Math.log);
console.log(`sorted  log-corr      ${corr(sortedS, sortedA).toFixed(4)}  (≈1 → same value distribution)`);
console.log(`median amax/scale     ${ratioMed.toFixed(1)}  (int4 qmax 8 would give ≈ 8/… → global factor still open)`);
console.log(`distinct scale values ${new Set(Array.from(sc).map(x => x.toExponential(6))).size} of ${nScale}`);
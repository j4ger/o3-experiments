#!/usr/bin/env node
/*
 * Parse the XNLM 0.5.0 container header (verified against both shipped models).
 *
 *   node header.js <file> [--block <n>] [--json]
 *
 * Layout (arithmetic-verified):
 *   [0..16)      ASCII magic "XNLM<ver>:<tag>"
 *   [16..24)     u64 metaLen
 *   [24..24+L)   meta blob (0.4.0: JSON; 0.5.0: binary)
 *   [..]         u64 block-size table: 0, size_0, size_1, ... terminated by 0
 *   [..4096)     zero padding so the payload starts at a 4096 boundary
 *   [4096..)     payload blocks, back to back, sizes from the table
 *
 * For a full file the tool asserts 4096 + sum(blocks) == file size; with only a
 * prefix available it still prints the table (that is how the in-device mimo
 * model was identified from a 4 KiB sample).
 */
'use strict';
const fs = require('fs');

function main() {
    const file = process.argv[2];
    if (!file) {
        console.error('usage: header.js <file> [--block <n>] [--json]');
        process.exit(2);
    }
    const st = fs.statSync(file);
    const fd = fs.openSync(file, 'r');
    const head = Buffer.alloc(Math.min(64 * 1024, st.size));
    fs.readSync(fd, head, 0, head.length, 0);

    const magic = head.subarray(0, 16).toString('latin1');
    const metaLen = Number(head.readBigUInt64LE(16));

    // find the block table: starts after the meta blob, ends at the first 0 after
    // at least one nonzero entry (the table itself starts with a 0 entry)
    const t0 = 24 + metaLen;
    const sizes = [];
    let p = t0;
    let sawNonZero = false;
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

    const sum = sizes.reduce((a, b) => a + b, 0);
    const expected = 4096 + sum;
    const out = {
        file,
        fileSize: st.size,
        magic,
        metaLen,
        tableOffset: t0,
        blocks: sizes.length,
        blockSum: sum,
        payloadStart: 4096,
        expectedSize: expected,
        sizeMatches: st.size === expected,
        sizeDelta: st.size - expected,
        metaPreview: head.subarray(24, 24 + Math.min(96, metaLen)).toString('latin1')
    };

    if (process.argv.includes('--block')) {
        const idx = parseInt(process.argv[process.argv.indexOf('--block') + 1], 10);
        out.blockOffsets = sizes.map((s, i) => ({ index: i, size: s, offset: 4096 + sizes.slice(0, i).reduce((a, b) => a + b, 0) }));
        out.selected = out.blockOffsets[idx] || null;
    }

    if (process.argv.includes('--json')) {
        console.log(JSON.stringify(out, null, 2));
    } else {
        console.log(`${file}: ${st.size} B`);
        console.log(`  magic      ${JSON.stringify(magic)}`);
        console.log(`  metaLen    ${metaLen}`);
        console.log(`  meta       ${JSON.stringify(out.metaPreview)}${metaLen > 96 ? '…' : ''}`);
        console.log(`  blockTable @${t0}: ${sizes.length} blocks, sum=${sum}`);
        if (sizes.length <= 8) {
            console.log(`  sizes      ${sizes.join(', ')}`);
        } else {
            const uniq = new Map();
            for (const s of sizes) uniq.set(s, (uniq.get(s) || 0) + 1);
            console.log(`  sizes      ${[...uniq.entries()].map(([s, c]) => `${s}×${c}`).join(', ')}`);
        }
        console.log(`  payload    starts at 4096; expected file size ${expected}`);
        console.log(`  check      ${out.sizeMatches ? 'OK (exact match)' : 'MISMATCH ' + out.sizeDelta}`);
        if (out.blockOffsets) {
            for (const b of out.blockOffsets) {
                console.log(`    block ${b.index}: @${b.offset} size=${b.size}`);
            }
        }
    }
    fs.closeSync(fd);
}

main();
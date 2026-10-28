#!/usr/bin/env node
/*
 * Stream-inflate just the head of one ZIP entry over HTTP Range.
 *
 *   node ziphead.js <url> <entry> [outBytes] [--out <file>]
 *
 * Prints entry metadata, the first 256 decompressed bytes as hex + ascii, and
 * a list of printable strings (>=6 chars) from the captured window.  With
 * --out the captured window is written verbatim for further analysis.
 *
 * Needed because model weights are deflate-compressed inside the .taiyi.model
 * ZIP and can be hundreds of MB — we only want the header/tensor table.
 */
'use strict';
const fs = require('fs');
const http = require('http');
const https = require('https');
const zlib = require('zlib');

const args = process.argv.slice(2);
const url = args[0];
if (!url || !args[1]) {
    console.error('usage: ziphead.js <url> <entry> [outBytes] [--out <file>]');
    process.exit(2);
}
const entryName = args[1];
const want = args[2] && !args[2].startsWith('--') ? parseInt(args[2], 10) : 65536;
const outIdx = args.indexOf('--out');
const outFile = outIdx >= 0 ? args[outIdx + 1] : null;

function request(method, headers) {
    return new Promise((resolve, reject) => {
        const u = new URL(url);
        const mod = u.protocol === 'https:' ? https : http;
        const req = mod.request(u, { method, headers }, res => resolve(res));
        req.on('error', reject);
        req.end();
    });
}

async function range(a, b) {
    const res = await request('GET', { Range: `bytes=${a}-${b}` });
    if (res.statusCode !== 206 && res.statusCode !== 200) {
        throw new Error('range ' + a + '-' + b + ' -> HTTP ' + res.statusCode);
    }
    const chunks = [];
    for await (const c of res) chunks.push(c);
    return Buffer.concat(chunks);
}

async function totalSize() {
    const res = await request('HEAD', {});
    if (res.statusCode >= 400) throw new Error('HEAD -> ' + res.statusCode);
    return parseInt(res.headers['content-length'], 10);
}

async function readCentralDirectory() {
    const len = await totalSize();
    const tailLen = Math.min(65557, len);
    const tailStart = len - tailLen;
    const tail = await range(tailStart, len - 1);
    let p = -1;
    for (let i = tail.length - 22; i >= 0; i--) {
        if (tail.readUInt32LE(i) === 0x06054b50) { p = i; break; }
    }
    if (p < 0) throw new Error('EOCD not found');
    const cdSize = tail.readUInt32LE(p + 12);
    const cdOffset = tail.readUInt32LE(p + 16);
    const cd = await range(cdOffset, cdOffset + cdSize - 1);
    const entries = {};
    let q = 0;
    while (q + 46 <= cd.length && cd.readUInt32LE(q) === 0x02014b50) {
        const method = cd.readUInt16LE(q + 10);
        const csize = cd.readUInt32LE(q + 20);
        const usize = cd.readUInt32LE(q + 24);
        const nlen = cd.readUInt16LE(q + 28);
        const elen = cd.readUInt16LE(q + 30);
        const clen = cd.readUInt16LE(q + 32);
        const lho = cd.readUInt32LE(q + 42);
        const name = cd.slice(q + 46, q + 46 + nlen).toString();
        entries[name] = { method, csize, usize, lho };
        q += 46 + nlen + elen + clen;
    }
    return { len, entries };
}

function scanStrings(buf, min = 6) {
    const out = [];
    let cur = '';
    for (const byte of buf) {
        if (byte >= 0x20 && byte <= 0x7e) {
            cur += String.fromCharCode(byte);
        } else {
            if (cur.length >= min) out.push(cur);
            cur = '';
        }
    }
    if (cur.length >= min) out.push(cur);
    return out;
}

async function main() {
    const { entries } = await readCentralDirectory();
    const e = entries[entryName];
    if (!e) throw new Error('entry not found: ' + entryName + ' (have: ' + Object.keys(entries).join(',') + ')');
    console.error(`entry=${entryName} method=${e.method} usize=${e.usize} csize=${e.csize}`);
    if (e.usize === 0) { console.error('empty entry'); return; }

    const head = await range(e.lho, e.lho + 30 + 1024 - 1);
    const nlen = head.readUInt16LE(26), elen = head.readUInt16LE(28);
    const dataOfs = e.lho + 30 + nlen + elen;

    let captured = Buffer.alloc(0);
    if (e.method === 0) {
        captured = await range(dataOfs, dataOfs + Math.min(want, e.usize) - 1);
    } else {
        // stream a bounded amount of compressed input, stop once we have `want`
        const need = Math.min(e.csize, Math.max(1 << 20, Math.min(8 << 20, Math.ceil(want * 1.6) + 65536)));
        const comp = await range(dataOfs, dataOfs + need - 1);
        const inflate = zlib.createInflateRaw();
        const parts = [];
        let got = 0;
        const done = new Promise(resolve => {
            inflate.on('data', chunk => {
                if (got < want) {
                    const take = chunk.subarray(0, want - got);
                    parts.push(take);
                    got += take.length;
                }
                if (got >= want) { inflate.destroy(); resolve(); }
            });
            inflate.on('end', resolve);
            inflate.on('error', err => {
                // truncated input is expected; keep what we have
                if (got > 0) resolve(); else resolve();
            });
        });
        inflate.write(comp);
        await Promise.race([done, new Promise(r => setTimeout(r, 20000))]);
        captured = Buffer.concat(parts);
    }

    console.error(`captured ${captured.length} decompressed bytes`);
    const hx = captured.subarray(0, 256).toString('hex').match(/../g).join(' ');
    console.error('--- first 256 bytes hex ---\n' + hx);
    console.error('--- first 256 bytes ascii ---\n' + JSON.stringify(
        captured.subarray(0, 256).toString('latin1').replace(/[^\x20-\x7e]/g, '.')));
    const strs = scanStrings(captured, 6);
    console.error(`--- strings (${strs.length}) ---`);
    for (const s of strs.slice(0, 120)) console.error('  ' + s);
    if (outFile) {
        fs.writeFileSync(outFile, captured);
        console.error('wrote ' + outFile);
    }
}

main().catch(e => { console.error('ERROR', e.message); process.exit(1); });
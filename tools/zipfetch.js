#!/usr/bin/env node
/*
 * Stream one ZIP entry (deflate or stored) out of a remote .taiyi.model over
 * HTTP Range and inflate it to a local file, with progress on stderr.
 *
 *   node zipfetch.js <url> <entry> <outFile>
 *
 * Only the entry's compressed byte range is requested (single ranged GET), so
 * this never downloads the other entries or the ZIP container overhead.
 */
'use strict';
const fs = require('fs');
const http = require('http');
const https = require('https');
const zlib = require('zlib');

const [url, entryName, outFile] = process.argv.slice(2);
if (!url || !entryName || !outFile) {
    console.error('usage: zipfetch.js <url> <entry> <outFile>');
    process.exit(2);
}

function request(method, headers) {
    return new Promise((resolve, reject) => {
        const u = new URL(url);
        const mod = u.protocol === 'https:' ? https : http;
        const req = mod.request(u, { method, headers }, res => resolve(res));
        req.on('error', reject);
        req.end();
    });
}

async function totalSize() {
    const res = await request('HEAD', {});
    if (res.statusCode >= 400) throw new Error('HEAD ' + res.statusCode);
    return parseInt(res.headers['content-length'], 10);
}

async function range(a, b) {
    const res = await request('GET', { Range: `bytes=${a}-${b}` });
    if (res.statusCode !== 206 && res.statusCode !== 200) {
        throw new Error(`range ${a}-${b} -> ${res.statusCode}`);
    }
    const parts = [];
    for await (const c of res) parts.push(c);
    return Buffer.concat(parts);
}

async function findEntry(name) {
    const len = await totalSize();
    const tailLen = Math.min(65557, len);
    const tail = await range(len - tailLen, len - 1);
    let p = -1;
    for (let i = tail.length - 22; i >= 0; i--) {
        if (tail.readUInt32LE(i) === 0x06054b50) { p = i; break; }
    }
    if (p < 0) throw new Error('EOCD not found');
    const cdSize = tail.readUInt32LE(p + 12);
    const cdOffset = tail.readUInt32LE(p + 16);
    const cd = await range(cdOffset, cdOffset + cdSize - 1);
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
        if (name === entryName) {
            const h = await range(lho, lho + 30 + 1024 - 1);
            const n2 = h.readUInt16LE(26), e2 = h.readUInt16LE(28);
            return { method, csize, usize, dataOfs: lho + 30 + n2 + e2 };
        }
        q += 46 + nlen + elen + clen;
    }
    throw new Error('entry not found: ' + entryName);
}

async function main() {
    const e = await findEntry(entryName);
    console.error(`entry=${entryName} method=${e.method} usize=${e.usize} csize=${e.csize}`);
    const res = await request('GET', { Range: `bytes=${e.dataOfs}-${e.dataOfs + e.csize - 1}` });
    if (res.statusCode !== 206 && res.statusCode !== 200) {
        throw new Error('data GET ' + res.statusCode);
    }
    const out = fs.createWriteStream(outFile);
    let got = 0;
    let last = Date.now();
    const sink = e.method === 8 ? zlib.createInflateRaw() : res;
    if (e.method === 8) res.pipe(sink);
    sink.on('data', chunk => {
        got += chunk.length;
        out.write(chunk);
        if (Date.now() - last > 5000) {
            last = Date.now();
            console.error(`  ${(got / 1048576).toFixed(1)} MiB / ${(e.usize / 1048576).toFixed(1)} MiB`);
        }
    });
    await new Promise((resolve, reject) => {
        sink.on('end', resolve);
        sink.on('error', reject);
        res.on('error', reject);
    });
    await new Promise(r => out.end(r));
    console.error(`done ${outFile} ${got} bytes (expected ${e.usize})`);
    process.exit(got === e.usize ? 0 : 1);
}

main().catch(e => { console.error('ERROR', e.message); process.exit(2); });
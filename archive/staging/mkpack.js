#!/usr/bin/env node
/*
 * Build a ZIP archive of an EXACT total size, with the central directory and
 * EOCD at the very end and the entry data at the front, followed by zero
 * padding.  This matters because:
 *
 *  - the resource download path is compared against the expected byte size in
 *    several places, so a replacement package should match it exactly;
 *  - the installer opens the package with java.util.zip.ZipFile, which locates
 *    the EOCD by scanning the last 64 KiB and then reads the central directory,
 *    whose entry offsets are absolute from the start of the file.
 *
 * Usage:
 *   node mkpack.js --out <file> --size <bytes> --add <name>=<path> [--add ...]
 *
 * Entries are stored (no compression) so the payload is fully predictable.
 * Prints a JSON summary; exits non-zero if the requested size cannot fit.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

function u16(v) { const b = Buffer.alloc(2); b.writeUInt16LE(v >>> 0, 0); return b; }
function u32(v) { const b = Buffer.alloc(4); b.writeUInt32LE(v >>> 0, 0); return b; }

function dosTime(d) {
    const time = ((d.getHours() & 31) << 11) | ((d.getMinutes() & 63) << 5) | ((d.getSeconds() / 2) & 31);
    const date = (((d.getFullYear() - 1980) & 127) << 9) | (((d.getMonth() + 1) & 15) << 5) | (d.getDate() & 31);
    return { time, date };
}

function buildZip(entries, totalSize, now) {
    const locals = [];
    const centrals = [];
    let offset = 0;

    for (const e of entries) {
        const name = Buffer.from(e.name, 'utf8');
        const data = e.data;
        const crc = zlib.crc32(data) >>> 0;
        const { time, date } = dosTime(now);

        const lfh = Buffer.concat([
            u32(0x04034b50), u16(20), u16(0), u16(0), u16(time), u16(date),
            u32(crc), u32(data.length), u32(data.length), u16(name.length), u16(0), name
        ]);
        locals.push(lfh, data);
        centrals.push({ name, crc, size: data.length, offset, time, date });
        offset += lfh.length + data.length;
    }

    const cdParts = [];
    let cdSize = 0;
    for (const c of centrals) {
        const cdh = Buffer.concat([
            u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(c.time), u16(c.date),
            u32(c.crc), u32(c.size), u32(c.size), u16(c.name.length), u16(0), u16(0),
            u16(0), u16(0), u32(0), u32(c.offset), c.name
        ]);
        cdParts.push(cdh);
        cdSize += cdh.length;
    }
    const cdOffset = offset + 0; // padding goes BEFORE the central directory
    const eocd = Buffer.concat([
        u32(0x06054b50), u16(0), u16(0), u16(centrals.length), u16(centrals.length),
        u32(cdSize), u32(cdOffset), u16(0)
    ]);

    const bodySize = offset + cdSize + eocd.length;
    if (totalSize && totalSize < bodySize) {
        throw new Error(`requested size ${totalSize} < minimum ${bodySize}`);
    }
    const padding = totalSize ? Buffer.alloc(totalSize - bodySize, 0) : Buffer.alloc(0);
    // layout: locals | padding | central directory | EOCD  (cdOffset already
    // points past padding because offset excludes padding)
    const cdOffsetFinal = offset + padding.length;
    const eocdFinal = Buffer.concat([
        u32(0x06054b50), u16(0), u16(0), u16(centrals.length), u16(centrals.length),
        u32(cdSize), u32(cdOffsetFinal), u16(0)
    ]);
    return Buffer.concat([...locals, padding, ...cdParts, eocdFinal]);
}

function main() {
    const argv = process.argv.slice(2);
    let out = null, size = 0;
    const adds = [];
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--out') out = argv[++i];
        else if (argv[i] === '--size') size = parseInt(argv[++i], 10);
        else if (argv[i] === '--add') adds.push(argv[++i]);
        else if (argv[i] === '--text') adds.push('marker.txt=' + argv[++i]);
        else throw new Error('unknown arg ' + argv[i]);
    }
    if (!out) throw new Error('--out is required');
    const entries = adds.map(spec => {
        const eq = spec.indexOf('=');
        const name = spec.slice(0, eq);
        const rest = spec.slice(eq + 1);
        const data = fs.existsSync(rest) && fs.statSync(rest).isFile()
            ? fs.readFileSync(rest)
            : Buffer.from(rest, 'utf8');
        return { name, data };
    });
    if (!entries.length) entries.push({ name: 'marker.txt', data: Buffer.from('oring_exp staging payload\n') });

    const zip = buildZip(entries, size, new Date());
    fs.writeFileSync(out, zip);
    console.log(JSON.stringify({
        out, size: zip.length, entries: entries.map(e => ({ name: e.name, bytes: e.data.length })),
        crc32: entries.map(e => zlib.crc32(e.data) >>> 0)
    }, null, 2));
}

main();
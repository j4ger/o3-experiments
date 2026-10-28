#!/usr/bin/env node
/*
 * Mine libxnnl.so for (a) its exported surface and (b) assertion strings, which
 * act as an undocumented spec for legal tensor layouts / dtypes / shapes.
 *
 *   node mine.js exports <libxnnl.so>
 *   node mine.js spec    <libxnnl.so>
 *
 * The assertion output groups messages by the identifier they talk about, so
 * each group reads like a constraint list for one operator or descriptor.
 */
'use strict';
const { execFileSync } = require('child_process');

const mode = process.argv[2];
const lib = process.argv[3];
if (!mode || !lib) {
    console.error('usage: mine.js exports|spec <libxnnl.so>');
    process.exit(2);
}
const strings = execFileSync('strings', ['-n', '6', lib], { maxBuffer: 1 << 28 })
    .toString('utf8').split('\n');

if (mode === 'exports') {
    const syms = execFileSync('readelf', ['-sW', lib], { maxBuffer: 1 << 28 })
        .toString('utf8').split('\n')
        .filter(l => /\s(FUNC|OBJECT)\s+GLOBAL\s+DEFAULT\s+/.test(l))
        .map(l => l.trim().split(/\s+/).pop())
        .filter(n => /^[A-Za-z_]/.test(n));
    const uniq = [...new Set(syms)].sort();
    const fam = new Map();
    for (const s of uniq) {
        // family = leading word, e.g. xnnlMatmulCreateGraph -> xnnlMatmul
        const m = /^(xnnl[A-Z][A-Za-z]*?)(?:[A-Z]|$)/.exec(s);
        const key = m ? m[1] : s.replace(/[0-9].*$/, '');
        fam.set(key, (fam.get(key) || 0) + 1);
    }
    console.log(`# libxnnl exported symbols: ${uniq.length}\n`);
    for (const [k, n] of [...fam.entries()].sort((a, b) => b[1] - a[1])) {
        console.log(`${String(n).padStart(5)}  ${k}`);
    }
    process.exit(0);
}

if (mode === 'spec') {
    const re = /(!=|==|not support|unsupported|invalid|failed|must be|out of range|larger than|dtype|layout)/i;
    const interesting = strings.filter(s => re.test(s) && /[A-Za-z]{3}/.test(s) && s.length < 200);
    const groups = new Map();
    for (const s of interesting) {
        const m = /([A-Za-z_][A-Za-z0-9_]*)(?:->|\.|::)/.exec(s);
        const key = m ? m[1] : (s.match(/^([A-Za-z_]{4,})\b/) || ['', 'general'])[1];
        const list = groups.get(key) || [];
        list.push(s.trim());
        groups.set(key, list);
    }
    const enums = new Set();
    for (const s of strings) {
        for (const m of s.matchAll(/\b(XNNL_[A-Z0-9_]{3,})\b/g)) enums.add(m[1]);
    }
    console.log(`# libxnnl constraint strings: ${interesting.length}, groups: ${groups.size}`);
    console.log(`# distinct XNNL_* constants seen: ${enums.size}\n`);
    console.log('## constants');
    console.log([...enums].sort().join('\n'));
    console.log('\n## groups');
    const sorted = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
    for (const [k, list] of sorted.slice(0, 60)) {
        console.log(`\n### ${k} (${list.length})`);
        for (const s of [...new Set(list)].slice(0, 12)) console.log('  - ' + s);
    }
    process.exit(0);
}

console.error('unknown mode');
process.exit(2);
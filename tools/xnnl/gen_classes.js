#!/usr/bin/env node
/*
 * Generate a C++ header for the xnnl layer classes from libxnlm's DWARF.
 *
 *   node gen_classes.js xnlm.dwarf --out docs/generated/xnnl_classes.h
 *
 * libxnlm embeds declared (and partly defined) xnnl layer classes such as
 * xnnlDecoderLayer, xnnlLLMPrefill and xnnlLLMDecode.  Their method DIEs carry
 * full parameter lists, and the class DIEs carry member offsets, so a usable
 * declaration header can be reconstructed without the vendor headers.
 * STL instantiations and compiler internals are filtered out.
 */
'use strict';
const fs = require('fs');

const file = process.argv[2];
const outPath = process.argv.includes('--out')
    ? process.argv[process.argv.indexOf('--out') + 1]
    : 'xnnl_classes.h';
if (!file) {
    console.error('usage: gen_classes.js <dwarfdump.txt> [--out file]');
    process.exit(2);
}

const lines = fs.readFileSync(file, 'utf8').split('\n');
const dieRe = /^(0x[0-9a-f]+):(\s+)(DW_TAG_[A-Za-z_]+)/;
const attrRe = /^\s+DW_AT_([a-z_]+)\s+(.*)$/;
const typeRe = /^\(0x[0-9a-f]+ "(.*)"\)$/;
const strRe = /^\("(.*)"\)$/;
const clean = v => {
    if (!v) return null;
    const t = typeRe.exec(v);
    if (t) return t[1];
    const s = strRe.exec(v);
    if (s) return s[1];
    return v;
};

const stack = [{ indent: -1, tag: 'root', attrs: {}, children: [] }];
for (let i = 0; i < lines.length; i++) {
    const m = dieRe.exec(lines[i]);
    if (!m) continue;
    const die = { tag: m[3], indent: m[2].length, attrs: {}, children: [] };
    for (let j = i + 1; j < lines.length; j++) {
        if (dieRe.test(lines[j])) break;
        const a = attrRe.exec(lines[j]);
        if (a) die.attrs[a[1]] = clean(a[2].trim());
    }
    while (stack.length > 1 && stack[stack.length - 1].indent >= die.indent) stack.pop();
    stack[stack.length - 1].children.push(die);
    stack.push(die);
}

const STL = /std::|__ndk1|vector<|allocator|basic_string|__compressed|__wrap|reverse_iterator|pair<|__hash|__tree|__split|__allocation|__libcpp|__function|unique_ptr|shared_ptr|tuple|initializer_list/;
const classes = new Map();

(function walk(node) {
    for (const c of node.children) {
        if (c.tag === 'DW_TAG_class_type' && c.attrs.name && c.attrs.name.startsWith('xnnl') && !STL.test(c.attrs.name)) {
            const name = c.attrs.name;
            const entry = classes.get(name) || { size: c.attrs.byte_size || '', methods: [], members: [] };
            if (c.attrs.byte_size) entry.size = c.attrs.byte_size;
            for (const ch of c.children) {
                if (ch.tag === 'DW_TAG_subprogram' && ch.attrs.name) {
                    const params = ch.children
                        .filter(x => x.tag === 'DW_TAG_formal_parameter')
                        .map(x => `${x.attrs.type || '?'}${x.attrs.name ? ' ' + x.attrs.name : ''}`);
                    entry.methods.push({ name: ch.attrs.name, ret: ch.attrs.type || 'void', params });
                } else if (ch.tag === 'DW_TAG_member' && ch.attrs.name) {
                    entry.members.push({ name: ch.attrs.name, type: ch.attrs.type || '?', off: ch.attrs.data_member_location || '' });
                }
            }
            classes.set(name, entry);
        }
        walk(c);
    }
})(stack[0]);

const names = [...classes.keys()].sort();
const out = [];
out.push('/* Generated from libxnlm.so DWARF by tools/xnnl/gen_classes.js.');
out.push(' * Vendor C++ layer classes referenced by the engine; member offsets are');
out.push(' * observed, method signatures come from declaration/definition DIEs.');
out.push(' * This is a reconstruction for analysis, not the vendor header. */');
out.push('#pragma once');
out.push('#include <cstdint>');
out.push('');
for (const n of names) {
    const c = classes.get(n);
    if (!c.methods.length && !c.members.length) continue;
    out.push(`class ${n} {${c.size ? `  /* size ${c.size} */` : ''}`);
    out.push('public:');
    for (const m of c.methods) {
        out.push(`    ${m.ret} ${m.name}(${m.params.join(', ')});`);
    }
    if (c.members.length) {
        out.push('    /* members */');
        for (const m of c.members) {
            out.push(`    ${m.type} ${m.name};${m.off ? `  /* +${m.off} */` : ''}`);
        }
    }
    out.push('};');
    out.push('');
}
fs.writeFileSync(outPath, out.join('\n'));
console.log(`wrote ${outPath}: ${names.length} classes, ${out.reduce((a, l) => a + (l.includes('(') && l.trim().startsWith('xnnl') === false && l.includes(');') ? 1 : 0), 0)} method lines`);
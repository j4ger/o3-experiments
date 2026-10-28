#!/usr/bin/env node
/*
 * Harvest an external API surface from a DWARF dump.
 *
 *   llvm-dwarfdump --debug-info libxnlm.so > xnlm.dwarf
 *   node harvest.js xnlm.dwarf --prefix xnnl [--json]
 *
 * libxnlm.so declares (and calls) the xnnl NPU API, and its DWARF carries the
 * exact prototypes plus the original header path/line for every declaration.
 * That makes a large, authoritative slice of the otherwise undocumented NPU
 * surface recoverable statically, without disassembly.
 *
 * Output: one line per declared function with its reconstructed prototype,
 * followed by the structs/enums of the same family and their members.
 */
'use strict';
const fs = require('fs');

const file = process.argv[2];
const prefix = process.argv.includes('--prefix')
    ? process.argv[process.argv.indexOf('--prefix') + 1]
    : 'xnnl';
if (!file) {
    console.error('usage: harvest.js <dwarfdump.txt> [--prefix xnnl] [--json]');
    process.exit(2);
}

const lines = fs.readFileSync(file, 'utf8').split('\n');

// index of first DIE line of each kind, with indentation depth
const dieRe = /^(0x[0-9a-f]+):(\s+)(DW_TAG_[A-Za-z_]+)/;
const attrRe = /^\s+DW_AT_([a-z_]+)\s+(.*)$/;
const typeRe = /^\(0x[0-9a-f]+ "([^"]+)"\)/;

const dies = [];
for (let i = 0; i < lines.length; i++) {
    const m = dieRe.exec(lines[i]);
    if (!m) continue;
    const indent = (dieRe.exec(lines[i]) || [0, "", ""])[2].length;
    const die = { offset: m[1], tag: m[3], indent, attrs: {}, children: [] };
    for (let j = i + 1; j < lines.length; j++) {
        if (dieRe.test(lines[j])) break;
        const a = attrRe.exec(lines[j]);
        if (!a) continue;
        const val = a[2].trim();
        const t = typeRe.exec(val);
        const str = /^\("(.*)"\)$/.exec(val);
        die.attrs[a[1]] = t ? t[1] : (str ? str[1] : val);
    }
    dies.push(die);
}

// rebuild the DIE tree by indentation
const root = { indent: -1, children: [] };
const stack = [root];
for (const d of dies) {
    while (stack.length > 1 && stack[stack.length - 1].indent >= d.indent) stack.pop();
    stack[stack.length - 1].children.push(d);
    stack.push(d);
}

function walk(node, fn) {
    for (const c of node.children) {
        fn(c);
        walk(c, fn);
    }
}

const funcs = [];
const types = new Map();
const seen = new Set();
walk(root, d => {
    const name = d.attrs.name;
    if (!name || !String(name).startsWith(prefix)) return;
    if (d.tag === 'DW_TAG_subprogram') {
        const key = name + '|' + (d.attrs.decl_line || '');
        if (seen.has(key)) return;
        seen.add(key);
        const params = d.children
            .filter(c => c.tag === 'DW_TAG_formal_parameter')
            .map(c => ({ name: c.attrs.name, type: c.attrs.type }));
        funcs.push({
            name,
            returns: d.attrs.type || 'void',
            declaration: d.attrs.declaration || '',
            file: (d.attrs.decl_file || '').replace(/^"/, '').replace(/"$/, ''),
            line: d.attrs.decl_line || '',
            params
        });
    } else if (d.tag === 'DW_TAG_structure_type' || d.tag === 'DW_TAG_enumeration_type' ||
               d.tag === 'DW_TAG_union_type') {
        if (types.has(name)) return;
        const members = d.children
            .filter(c => c.tag === 'DW_TAG_member' || c.tag === 'DW_TAG_enumerator')
            .map(c => ({
                name: c.attrs.name,
                type: c.attrs.type || '',
                value: c.attrs.const_value || '',
                loc: c.attrs.data_member_location || ''
            }));
        types.set(name, {
            tag: d.tag, size: d.attrs.byte_size || '', members,
            file: (d.attrs.decl_file || '').replace(/^"/, '').replace(/"$/, ''),
            line: d.attrs.decl_line || ''
        });
    }
});

const out = { functions: funcs, types: Object.fromEntries(types) };
if (process.argv.includes('--json')) {
    console.log(JSON.stringify(out, null, 2));
    process.exit(0);
}

console.log(`# ${prefix}* surface harvested from ${file}\n`);
console.log(`functions: ${funcs.length}; types: ${types.size}\n`);
const byFile = new Map();
for (const f of funcs) {
    const base = (f.file.split('/').pop() || '?');
    if (!byFile.has(base)) byFile.set(base, []);
    byFile.get(base).push(f);
}
for (const [base, list] of [...byFile.entries()].sort()) {
    console.log(`## ${base}  (${list.length})`);
    for (const f of list.sort((a, b) => a.name.localeCompare(b.name))) {
        const ps = f.params.map(p => `${p.type || '?'} ${p.name || ''}`.trim()).join(', ');
        console.log(`  ${f.returns || 'void'} ${f.name}(${ps})${f.line ? `   // :${f.line}` : ''}`);
    }
    console.log('');
}
console.log(`## types (${types.size})`);
for (const [name, t] of [...types.entries()].sort()) {
    const kind = t.tag.replace('DW_TAG_', '');
    console.log(`  ${kind} ${name}${t.size ? ' size=' + t.size : ''}${t.members.length ? ` members=${t.members.length}` : ''}`);
}
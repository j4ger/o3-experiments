#!/usr/bin/env node
/*
 * Recover the typed NPU function table from libxnlm's WrapperFuncTable.
 *
 *   llvm-dwarfdump --debug-info libxnlm.so > xnlm.dwarf
 *   node functable.js xnlm.dwarf [--json] [--filter substr]
 *
 * libxnlm does not link libxnnl/libmida statically: it dlopens them and fills
 * struct WrapperFuncTable with typed function pointers (FuncType_*).  DWARF
 * keeps those typedefs, so following typedef -> pointer -> subroutine_type by
 * DIE offset yields exact prototypes (return type, parameter types, names,
 * order) for the whole NPU surface the engine uses.
 */
'use strict';
const fs = require('fs');

const file = process.argv[2];
if (!file) {
    console.error('usage: functable.js <dwarfdump.txt> [--json] [--filter substr]');
    process.exit(2);
}
const lines = fs.readFileSync(file, 'utf8').split('\n');
const dieRe = /^(0x[0-9a-f]+):(\s+)(DW_TAG_[A-Za-z_]+)/;
const attrRe = /^\s+DW_AT_([a-z_]+)\s+(.*)$/;

const typeRe = /^\((0x[0-9a-f]+) "([^"]*)"\)$/;
const strRe = /^\("(.*)"\)$/;
const numRe = /^\(([^()]*)\)$/;

function parseVal(v) {
    let m = typeRe.exec(v);
    if (m) return { off: m[1], name: m[2] };
    m = strRe.exec(v);
    if (m) return { name: m[1] };
    m = numRe.exec(v);
    if (m) return { value: m[1] };
    return { name: v };
}

// ---- parse DIEs -----------------------------------------------------------
const byOffset = new Map();
const byName = new Map();
const stack = [{ offset: null, indent: -1, tag: 'root', attrs: {}, children: [] }];
for (let i = 0; i < lines.length; i++) {
    const m = dieRe.exec(lines[i]);
    if (!m) continue;
    const die = { offset: m[1], tag: m[3], indent: m[2].length, attrs: {}, children: [] };
    for (let j = i + 1; j < lines.length; j++) {
        if (dieRe.test(lines[j])) break;
        const a = attrRe.exec(lines[j]);
        if (!a) continue;
        die.attrs[a[1]] = parseVal(a[2].trim());
    }
    byOffset.set(die.offset, die);
    if (die.attrs.name && die.attrs.name.name) {
        const list = byName.get(die.attrs.name.name) || [];
        list.push(die);
        byName.set(die.attrs.name.name, list);
    }
    while (stack.length > 1 && stack[stack.length - 1].indent >= die.indent) stack.pop();
    stack[stack.length - 1].children.push(die);
    stack.push(die);
}

function deref(t) {
    return t && t.off ? byOffset.get(t.off) : null;
}

// follow typedef chains down to a subroutine_type
function signature(typeRef) {
    let node = deref(typeRef);
    let guard = 0;
    while (node && guard++ < 8) {
        if (node.tag === 'DW_TAG_subroutine_type') break;
        node = deref(node.attrs.type);
    }
    if (!node || node.tag !== 'DW_TAG_subroutine_type') return null;
    const params = node.children
        .filter(c => c.tag === 'DW_TAG_formal_parameter')
        .map(c => ({
            type: c.attrs.type ? c.attrs.type.name : '?',
            name: c.attrs.name ? c.attrs.name.name : ''
        }));
    return { returns: node.attrs.type ? node.attrs.type.name : 'void', params };
}

const table = (byName.get('WrapperFuncTable') || [])
    .sort((a, b) => b.children.length - a.children.length)[0];
if (!table) {
    console.error('WrapperFuncTable not found');
    process.exit(1);
}

const filter = process.argv.includes('--filter')
    ? process.argv[process.argv.indexOf('--filter') + 1]
    : null;

const out = [];
for (const m of table.children) {
    if (m.tag !== 'DW_TAG_member') continue;
    const field = m.attrs.name ? m.attrs.name.name : '?';
    if (filter && !field.includes(filter)) continue;
    out.push({ field, funcType: m.attrs.type ? m.attrs.type.name : '', sig: signature(m.attrs.type) });
}

if (process.argv.includes('--json')) {
    console.log(JSON.stringify(out, null, 2));
    process.exit(0);
}
console.log(`# WrapperFuncTable: ${out.length} entries\n`);
let resolved = 0;
for (const e of out) {
    if (!e.sig) {
        console.log(`? ${e.field}(...)            // unresolved ${e.funcType}`);
        continue;
    }
    resolved++;
    const ps = e.sig.params.map(p => `${p.type}${p.name ? ' ' + p.name : ''}`).join(', ');
    console.log(`${e.sig.returns} ${e.field}(${ps})`);
}
console.log(`\nresolved ${resolved}/${out.length}`);
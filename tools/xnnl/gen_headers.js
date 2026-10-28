#!/usr/bin/env node
/*
 * Generate C headers for the XRING NPU stack from libxnlm's DWARF.
 *
 *   llvm-dwarfdump --debug-info libxnlm.so > xnlm.dwarf
 *   node gen_headers.js xnlm.dwarf --out-dir docs/generated
 *
 * Emits:
 *   xnlm_api.h   exact prototypes of the 19 exported XNLM_* functions
 *   mida_api.h   exact prototypes of the 57 mida* function pointers
 *   xnnl_api.h   xnnl* declarations (return type known, arity from call sites)
 *   xnnl_types.h structs and enums used by the above, with member offsets
 *
 * Why this works: libxnlm ships full DWARF.  Exported XNLM_* have definitions
 * (with formal parameters), the mida API lives in struct WrapperFuncTable as
 * typed function pointers, and xnnl* only appears as declarations - for those
 * we recover arity from DW_TAG_GNU_call_site_parameter records.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const file = process.argv[2];
const outDir = process.argv.includes('--out-dir')
    ? process.argv[process.argv.indexOf('--out-dir') + 1]
    : '.';
if (!file) {
    console.error('usage: gen_headers.js <dwarfdump.txt> --out-dir <dir>');
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

const byOffset = new Map();
const byName = new Map();
const root = { indent: -1, children: [], tag: 'root', attrs: {} };
{
    const stack = [root];
    for (let i = 0; i < lines.length; i++) {
        const m = dieRe.exec(lines[i]);
        if (!m) continue;
        const die = { offset: m[1], tag: m[3], indent: m[2].length, attrs: {}, children: [] };
        for (let j = i + 1; j < lines.length; j++) {
            if (dieRe.test(lines[j])) break;
            const a = attrRe.exec(lines[j]);
            if (a) die.attrs[a[1]] = parseVal(a[2].trim());
        }
        byOffset.set(die.offset, die);
        const nm = die.attrs.name && die.attrs.name.name;
        if (nm) {
            const list = byName.get(nm) || [];
            list.push(die);
            byName.set(nm, list);
        }
        while (stack.length > 1 && stack[stack.length - 1].indent >= die.indent) stack.pop();
        stack[stack.length - 1].children.push(die);
        stack.push(die);
    }
}

const typeName = t => (t && t.name) || 'void';
// emit a valid C declarator: "int32_t[4] shape" -> "int32_t shape[4]"
const decl = (t, n) => {
    const m = /^(.*?)\s*(\[[^\]]*\])$/.exec(typeName(t));
    return m ? `${m[1]} ${n}${m[2]}` : `${typeName(t)} ${n}`;
};
const deref = t => (t && t.off ? byOffset.get(t.off) : null);
const pick = (name, tag) => {
    const list = (byName.get(name) || []).filter(d => !tag || d.tag === tag);
    return list.sort((a, b) => b.children.length - a.children.length)[0] || null;
};

// ---- call-site arity index -------------------------------------------------
const arity = new Map();
(function walk(node) {
    for (const c of node.children) {
        if (c.tag === 'DW_TAG_GNU_call_site' && c.attrs.abstract_origin) {
            const name = c.attrs.abstract_origin.name;
            const n = c.children.filter(x => x.tag === 'DW_TAG_GNU_call_site_parameter').length;
            if (name) arity.set(name, Math.max(arity.get(name) || 0, n));
        }
        walk(c);
    }
})(root);

// ---- prototypes -----------------------------------------------------------
function protoOf(die) {
    const params = die.children
        .filter(c => c.tag === 'DW_TAG_formal_parameter')
        .map(c => ({ type: typeName(c.attrs.type), name: c.attrs.name ? c.attrs.name.name : '' }));
    return { ret: typeName(die.attrs.type), params };
}

function collect(subprogramName) {
    const list = (byName.get(subprogramName) || [])
        .filter(d => d.tag === 'DW_TAG_subprogram')
        .sort((a, b) => b.children.length - a.children.length);
    const def = list.find(d => d.children.some(c => c.tag === 'DW_TAG_formal_parameter')) || list[0];
    return def ? protoOf(def) : null;
}

// ---- type emission --------------------------------------------------------
const emittedTypes = new Set();
const typeBlocks = [];

function emitStruct(name) {
    if (emittedTypes.has(name)) return;
    const st = pick(name, 'DW_TAG_structure_type') || pick(name, 'DW_TAG_union_type') ||
        pick(name, 'DW_TAG_class_type');
    if (!st) return;
    emittedTypes.add(name);
    const size = st.attrs.byte_size ? st.attrs.byte_size.value : '?';
    const members = st.children.filter(c => c.tag === 'DW_TAG_member');
    const out = [];
    out.push(`/* size ${size} */`);
    out.push(`struct ${name.replace(/[^A-Za-z0-9_]/g, '_')} {`);
    for (const m of members) {
        const mn = m.attrs.name ? m.attrs.name.name : '?';
        const loc = m.attrs.data_member_location ? m.attrs.data_member_location.value : '';
        out.push(`    ${decl(m.attrs.type, mn)};${loc ? `  /* +${loc} */` : ''}`);
    }
    out.push('};');
    typeBlocks.push(out.join('\n'));
}

function emitTypedef(name) {
    if (emittedTypes.has(name)) return;
    const td = pick(name, 'DW_TAG_typedef');
    if (!td) return;
    // enum through typedef?
    let node = deref(td.attrs.type);
    let guard = 0;
    while (node && node.tag !== 'DW_TAG_enumeration_type' && guard++ < 6) {
        if (node.tag !== 'DW_TAG_typedef' && node.tag !== 'DW_TAG_const_type') break;
        node = deref(node.attrs.type);
    }
    if (node && node.tag === 'DW_TAG_enumeration_type') {
        emittedTypes.add(name);
        const out = [`typedef enum ${name.replace(/[^A-Za-z0-9_]/g, '_')} {`];
        for (const e of node.children.filter(c => c.tag === 'DW_TAG_enumerator')) {
            const v = e.attrs.const_value ? e.attrs.const_value.value : '?';
            out.push(`    ${e.attrs.name.name} = ${v},`);
        }
        out.push(`} ${name};`);
        typeBlocks.push(out.join('\n'));
        return;
    }
    // anonymous struct behind a typedef -> emit the body under the typedef name
    if (node && (node.tag === 'DW_TAG_structure_type' || node.tag === 'DW_TAG_union_type') &&
        node.children.some(c => c.tag === 'DW_TAG_member')) {
        emittedTypes.add(name);
        const out = [`typedef ${node.tag === 'DW_TAG_union_type' ? 'union' : 'struct'} {`];
        for (const m of node.children.filter(c => c.tag === 'DW_TAG_member')) {
            const loc = m.attrs.data_member_location ? m.attrs.data_member_location.value : '';
            out.push(`    ${decl(m.attrs.type, m.attrs.name ? m.attrs.name.name : '?')};${loc ? `  /* +${loc} */` : ''}`);
        }
        out.push(`} ${name};`);
        typeBlocks.push(out.join('\n'));
        return;
    }
    // pointer typedef to a struct, or plain typedef
    const target = node && node.tag === 'DW_TAG_pointer_type'
        ? 'struct ' + (node.attrs.type ? node.attrs.type.name.replace(/[^A-Za-z0-9_]/g, '_') : 'opaque') + ' *'
        : typeName(td.attrs.type);
    if (target && target !== 'void') {
        emittedTypes.add(name);
        typeBlocks.push(`typedef ${target} ${name};`);
    }
}

function emitEnum(name) {
    if (emittedTypes.has(name)) return;
    const en = pick(name, 'DW_TAG_enumeration_type');
    if (!en) return;
    emittedTypes.add(name);
    const out = [`enum ${name.replace(/[^A-Za-z0-9_]/g, '_')} {`];
    for (const e of en.children.filter(c => c.tag === 'DW_TAG_enumerator')) {
        const v = e.attrs.const_value ? e.attrs.const_value.value : '?';
        out.push(`    ${e.attrs.name.name} = ${v},`);
    }
    out.push('};');
    typeBlocks.push(out.join('\n'));
}

// ---- emit XNLM_* ----------------------------------------------------------
const xnlm = [];
for (const name of [...byName.keys()].sort()) {
    if (!name.startsWith('XNLM_')) continue;
    const p = collect(name);
    if (!p) continue;
    const ps = p.params.map(a => `${a.type}${a.name ? ' ' + a.name : ''}`).join(', ');
    xnlm.push(`${p.ret} ${name}(${ps || 'void'});`);
}

// ---- emit mida* -----------------------------------------------------------
const table = pick('WrapperFuncTable', 'DW_TAG_structure_type');
const mida = [];
const midaTypes = new Set();
if (table) {
    for (const m of table.children.filter(c => c.tag === 'DW_TAG_member')) {
        const field = m.attrs.name ? m.attrs.name.name : '?';
        if (field === 'dlHandle') continue;
        let node = deref(m.attrs.type);
        let guard = 0;
        while (node && node.tag !== 'DW_TAG_subroutine_type' && guard++ < 8) {
            node = deref(node.attrs.type);
        }
        if (!node) continue;
        const params = node.children.filter(c => c.tag === 'DW_TAG_formal_parameter')
            .map(c => ({ type: typeName(c.attrs.type), name: c.attrs.name ? c.attrs.name.name : '' }));
        for (const p of params) {
            const base = p.type.replace(/\s*\*+$/, '').trim();
            if (/^(mida|XNNL)/.test(base)) midaTypes.add(base);
        }
        const ret = typeName(node.attrs.type);
        mida.push(`${ret} ${field}(${params.map(p => `${p.type}${p.name ? ' ' + p.name : ''}`).join(', ') || 'void'});`);
    }
}

// ---- emit xnnl* -----------------------------------------------------------
const xnnl = [];
const xnnlUnknown = [];
{
    const names = [...byName.entries()]
        .filter(([n]) => n.startsWith('xnnl'))
        .map(([n, list]) => ({ n, list }));
    for (const { n, list } of names.sort((a, b) => a.n.localeCompare(b.n))) {
        const decl = list.filter(d => d.tag === 'DW_TAG_subprogram')
            .sort((a, b) => b.children.length - a.children.length)[0];
        if (!decl) continue;
        const p = protoOf(decl);
        const looksCtor = p.ret === 'void' && p.params.length &&
            p.params[0].type.replace(/[^A-Za-z0-9_]/g, '') === n.replace(/[^A-Za-z0-9_]/g, '') + '';
        if (looksCtor) continue;
        const a = arity.get(n) || 0;
        const ps = p.params.length
            ? p.params.map(x => `${x.type}${x.name ? ' ' + x.name : ''}`).join(', ')
            : (a ? `/* arity ${a} - params unknown */` : '/* params unknown */');
        if (p.params.length) {
            xnnl.push(`${p.ret} ${n}(${ps});`);
        } else {
            xnnlUnknown.push({ name: n, ret: p.ret, arity: a, params: ps });
        }
    }
}

// ---- write ----------------------------------------------------------------
fs.mkdirSync(outDir, { recursive: true });
const banner = f => `/* Generated from libxnlm.so DWARF by tools/xnnl/gen_headers.js.\n * Source of truth: the shipped library; do not edit by hand.\n */\n#ifndef ${f}\n#define ${f}\n\n#include <stdint.h>\n\n`;

// xnlm_api.h
{
    for (const t of ['XnlmStatus', 'XnlmDataType', 'XnlmQuantParam', 'XnlmTensorDesc', 'XnlmTensorDescImpl', 'XnlmMemoryImpl',
        'XnlmRuntime_t', 'XnlmModel_t', 'XnlmMemory_t', 'XnlmKVCache_t', 'XnlmLoraWeight_t',
        'XnlmTensorDesc_t', 'XnlmRunConfig_t']) {
        emitTypedef(t); emitEnum(t); emitStruct(t);
    }
    const body = [
        banner('XNLM_API_H'),
        '/* ---- types ---- */',
        ...typeBlocks,
        '',
        '/* ---- exported API ---- */',
        ...xnlm,
        '',
        '#endif'
    ].join('\n');
    fs.writeFileSync(path.join(outDir, 'xnlm_api.h'), body);
}

// mida_api.h (types are opaque handles; emit what we know)
{
    typeBlocks.length = 0;
    emittedTypes.clear();
    for (const t of [...midaTypes].sort()) {
        emitTypedef(t); emitEnum(t); emitStruct(t);
    }
    const body = [
        banner('MIDA_API_H'),
        '/* opaque handles as declared in the vendor headers */',
        ...typeBlocks,
        '',
        '/* ---- resolved through struct WrapperFuncTable (dlsym) ---- */',
        ...mida,
        '',
        '#endif'
    ].join('\n');
    fs.writeFileSync(path.join(outDir, 'mida_api.h'), body);
}

// xnnl_api.h
{
    typeBlocks.length = 0;
    emittedTypes.clear();
    for (const t of ['xnnlStatus_t', 'xnnlTensorDescriptor', 'xnnlQuantParam', 'xnnlTensorData',
        'xnnlDecoderWeight', 'xnnlEagletLmheadWeight', 'xnnlWeightPatch',
        'xnnlEpilogueDescriptor', 'xnnlTensorLayout_t', 'xnnlDataType_t',
        'xnnlQuantizeMode_t', 'xnnlMemLocation_t', 'xnnlHandle_t', 'xnnlTensorDescriptor_t',
        'xnnlEpilogueDescriptor_t', 'xnnlEpilogueDataMode_t', 'xnnlActivation1Type_t',
        'xnnlActivation2Type_t', 'xnnlEwOp_t']) {
        emitTypedef(t); emitEnum(t); emitStruct(t);
    }
    const body = [
        banner('XNNL_API_H'),
        '/* ---- types ---- */',
        ...typeBlocks,
        '',
        '/* ---- declarations with full signatures (from DWARF) ---- */',
        ...xnnl,
        '',
        '/* ---- declarations whose parameters are not in DWARF;',
        ' *      arity recovered from DW_TAG_GNU_call_site_parameter ---- */',
        ...xnnlUnknown.map(u => `/* arity ${u.arity} */ ${u.ret} ${u.name}(/* TODO */);`),
        '',
        '#endif'
    ].join('\n');
    fs.writeFileSync(path.join(outDir, 'xnnl_api.h'), body);
}

console.log(`wrote ${outDir}/xnlm_api.h     (${xnlm.length} prototypes)`);
console.log(`wrote ${outDir}/mida_api.h     (${mida.length} prototypes)`);
console.log(`wrote ${outDir}/xnnl_api.h     (${xnnl.length} full, ${xnnlUnknown.length} arity-only)`);
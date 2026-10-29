#!/usr/bin/env node
/*
 * list-kernels.mjs — enumerate the NPU kernels that ship precompiled inside
 * libxnnl.so.
 *
 * libxnnl.so is not a thin dispatcher: it embeds one ELF object per kernel
 * group.  Each embedded object carries
 *
 *     .k_text.<Kernel>   the compiled VIP9000 code
 *     .k_abi.<Kernel>    the argument ABI descriptor
 *     .k_arg.<Kernel>    argument layout (when present)
 *     .k_info.<Kernel>   launch geometry / metadata
 *
 * xnnlCreate() loads them into the device through midaModuleLoadData(), which
 * is why no compiler is needed on the device: compilation happened at Xiaomi
 * build time (see the `compile_discrete_mida_genrule` path in the strings).
 *
 * Usage:
 *     node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so
 *     node tools/npu/list-kernels.mjs artifacts/vendor/libxnnl.so --json
 */

import { readFileSync } from "node:fs";

const ELF_MAGIC = [0x7f, 0x45, 0x4c, 0x46];

function findElfOffsets(buf) {
    const out = [];
    for (let i = 0; i + 4 <= buf.length; i++) {
        if (buf[i] === ELF_MAGIC[0] && buf[i + 1] === ELF_MAGIC[1] &&
            buf[i + 2] === ELF_MAGIC[2] && buf[i + 3] === ELF_MAGIC[3]) {
            out.push(i);
        }
    }
    return out;
}

function parseSections(buf, off) {
    // Only 64-bit little-endian ELF images are expected here.
    if (buf[off + 4] !== 2 || buf[off + 5] !== 1) return null;

    const shoff = Number(buf.readBigUInt64LE(off + 0x28));
    const shentsize = buf.readUInt16LE(off + 0x3a);
    const shnum = buf.readUInt16LE(off + 0x3c);
    const shstrndx = buf.readUInt16LE(off + 0x3e);

    if (shoff <= 0 || shoff >= buf.length) return null;
    if (shentsize !== 64 || shnum <= 0 || shnum >= 512) return null;
    if (shstrndx >= shnum) return null;

    const shstr = Number(buf.readBigUInt64LE(off + shoff + shstrndx * 64 + 0x18));
    if (shstr <= 0 || shstr >= buf.length) return null;

    const names = [];
    for (let i = 0; i < shnum; i++) {
        const nameOff = buf.readUInt32LE(off + shoff + i * 64);
        const p = off + shstr + nameOff;
        if (p >= buf.length) continue;
        const end = buf.indexOf(0, p);
        if (end < 0 || end - p > 200) continue;
        names.push(buf.toString("latin1", p, end));
    }
    return names;
}

const KERNEL_PREFIXES = [".k_text.", ".k_abi.", ".k_arg.", ".k_info."];

function kernelName(section) {
    for (const p of KERNEL_PREFIXES) {
        if (section.startsWith(p)) return section.slice(p.length);
    }
    return null;
}

const CATEGORIES = [
    ["gemm/matmul", /gemm|matmul/i],
    ["bmm/attention", /bmm|attn/i],
    ["rmsnorm", /rmsnorm/i],
    ["layernorm", /layernorm/i],
    ["softmax", /softmax/i],
    ["rope", /rope|rotary/i],
    ["quant", /quant|dequant/i],
    ["activation", /silu|gelu|swiglu|relu|sigmoid/i],
    ["conv", /conv/i],
    ["pool", /pool/i],
    ["pad/tile/copy", /pad|tile|copy|slice|concat/i],
];

function main() {
    const args = process.argv.slice(2);
    const path = args.find((a) => !a.startsWith("--"));
    const asJson = args.includes("--json");
    if (!path) {
        console.error("usage: node list-kernels.mjs <libxnnl.so> [--json]");
        process.exit(2);
    }

    const buf = readFileSync(path);
    const modules = [];

    for (const off of findElfOffsets(buf)) {
        let names;
        try {
            names = parseSections(buf, off);
        } catch {
            continue;
        }
        if (!names) continue;

        const abi = [];
        const text = [];
        const seenAbi = new Set();
        const seenText = new Set();
        for (const n of names) {
            const k = kernelName(n);
            if (!k) continue;
            if (n.startsWith(".k_abi.") && !seenAbi.has(k)) {
                seenAbi.add(k);
                abi.push(k);
            } else if (n.startsWith(".k_text.") && !seenText.has(k)) {
                seenText.add(k);
                text.push(k);
            }
        }
        if (abi.length === 0 && text.length === 0) continue;
        modules.push({ offset: off, abi, text });
    }

    const all = [...new Set(modules.flatMap((m) => m.abi.concat(m.text)))].sort();
    const summary = Object.fromEntries(
        CATEGORIES.map(([label, re]) => [label, all.filter((k) => re.test(k)).length]),
    );

    if (asJson) {
        console.log(JSON.stringify({
            path,
            size: buf.length,
            kernel_modules: modules.length,
            kernels: all.length,
            categories: summary,
            modules: modules.map((m) => ({ offset: `0x${m.offset.toString(16)}`, kernels: m.abi })),
        }, null, 2));
        return;
    }

    console.log(`file            ${path} (${buf.length} bytes)`);
    console.log(`kernel modules  ${modules.length}`);
    console.log(`kernel names    ${all.length}`);
    console.log("");
    console.log("by operator family");
    for (const [label, n] of Object.entries(summary)) {
        console.log(`  ${label.padEnd(14)} ${n}`);
    }
    console.log("");
    console.log("modules");
    for (const m of modules) {
        console.log(`  0x${m.offset.toString(16).padStart(6, "0")}  ${String(m.abi.length).padStart(3)} kernels  ${m.abi.slice(0, 3).join(", ")}`);
    }
}

main();

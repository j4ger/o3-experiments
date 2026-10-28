#!/usr/bin/env node
/*
 * Host simulation of the staging-dir swap mechanism (no device required).
 *
 * It runs the very same script the app pushes to the device
 * (chat/assets/staging_swap.sh) against a fake staging path while a fake
 * downloader writes "genuine" bytes, then plays the installer's copy step and
 * asserts that the installer read OUR payload.
 *
 * Two downloader styles are covered:
 *   fd      - one long-lived file descriptor (writes continue into the orphaned
 *             inode after the swap, so the path keeps our content)
 *   reopen  - open/append per chunk (the writer grows our file, so the swap loop
 *             must keep re-installing the payload until the window closes)
 *
 * usage: node swap-sim.js [--size N] [--chunks N] [--delay ms] [--style fd|reopen|both]
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const SWAP = path.join(ROOT, 'chat', 'assets', 'staging_swap.sh');
const MKPACK = path.join(__dirname, 'mkpack.js');

function bytesEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

function sha256(buf) {
    return require('crypto').createHash('sha256').update(buf).digest('hex');
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runStyle(style, opts) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'stagesim-'));
    const target = path.join(dir, 'model', 'plug_f3', '25', 'plugin-mock-F3-1.taiyi');
    fs.mkdirSync(path.dirname(target), { recursive: true });
    const repl = path.join(dir, 'payload.zip');

    // our replacement: valid ZIP of exactly `size` bytes, central dir at the end
    const marker = `oring_exp-swap-${style}-${Date.now()}`;
    execFileSync(process.execPath, [
        MKPACK, '--out', repl, '--size', String(opts.size),
        '--add', `mock_plugin.apk=PK\x03\x04 ${marker}`,
        '--add', `taiyi_hash.json=[{"name":"mock_plugin.apk"}]`
    ], { stdio: 'pipe' });
    const payload = fs.readFileSync(repl);
    const payloadSha = sha256(payload);

    // arm the swap loop for the whole window (download + completion + copy)
    const durationMs = opts.durationMs;
    const swapper = spawn('sh', [SWAP, target, repl, String(opts.size), String(durationMs), '0.02'], {
        stdio: ['ignore', 'pipe', 'pipe']
    });
    let swapLog = '';
    swapper.stdout.on('data', d => { swapLog += d.toString(); });
    swapper.stderr.on('data', d => { swapLog += d.toString(); });

    await sleep(50); // let it arm

    // fake downloader: genuine bytes, chunked
    const chunkBytes = Math.max(1, Math.floor(opts.size / opts.chunks));
    let fd = null;
    if (style === 'fd') fd = fs.openSync(target, 'w');
    for (let i = 0; i < opts.chunks; i++) {
        const buf = Buffer.alloc(chunkBytes, 0x41 + (i % 26)); // "genuine" filler
        if (style === 'fd') fs.writeSync(fd, buf);
        else fs.appendFileSync(target, buf);
        await sleep(opts.delayMs);
    }
    if (style === 'fd') { fs.closeSync(fd); }
    // pad genuine file to the expected size so the download looks complete
    const now = fs.existsSync(target) ? fs.statSync(target).size : 0;
    if (now < opts.size) {
        const pad = Buffer.alloc(opts.size - now, 0x5A);
        if (style === 'fd') fs.appendFileSync(target, pad);
        else fs.appendFileSync(target, pad);
    }
    const afterWriter = fs.existsSync(target) ? fs.readFileSync(target) : Buffer.alloc(0);

    // give the swap loop a couple of polls to fix up the path, then play installer
    await sleep(120);
    const installDir = path.join(dir, 'install');
    fs.mkdirSync(installDir, { recursive: true });
    const installed = path.join(installDir, 'copied.taiyi');
    fs.copyFileSync(target, installed);           // ResourceInstaller e(str2, srcPath)
    const installedBytes = fs.readFileSync(installed);

    await new Promise(r => swapper.on('exit', r));

    const okPath = sha256(afterWriter) === payloadSha && afterWriter.length === opts.size;
    const okInstall = sha256(installedBytes) === payloadSha;
    const result = {
        style,
        size: opts.size,
        expected_sha256: payloadSha,
        writer_output_sha256: sha256(afterWriter),
        writer_output_len: afterWriter.length,
        installer_sha256: sha256(installedBytes),
        path_is_our_payload: okPath,
        installer_read_our_payload: okInstall,
        swap_lines: swapLog.trim().split('\n').slice(0, 6)
    };
    fs.rmSync(dir, { recursive: true, force: true });
    return result;
}

async function main() {
    const argv = process.argv.slice(2);
    const opts = { size: 300000, chunks: 12, delayMs: 25, durationMs: 3000, style: 'both' };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--size') opts.size = parseInt(argv[++i], 10);
        else if (a === '--chunks') opts.chunks = parseInt(argv[++i], 10);
        else if (a === '--delay') opts.delayMs = parseInt(argv[++i], 10);
        else if (a === '--duration') opts.durationMs = parseInt(argv[++i], 10);
        else if (a === '--style') opts.style = argv[++i];
        else throw new Error('unknown arg ' + a);
    }
    if (!fs.existsSync(SWAP)) throw new Error('missing ' + SWAP);
    const styles = opts.style === 'both' ? ['fd', 'reopen'] : [opts.style];
    const results = [];
    for (const s of styles) results.push(await runStyle(s, opts));
    console.log(JSON.stringify(results, null, 2));
    const failed = results.filter(r => !r.installer_read_our_payload);
    if (failed.length) {
        console.error('FAIL: installer did not read our payload for style(s) ' +
            failed.map(f => f.style).join(', '));
        process.exit(1);
    }
    console.log('PASS: installer read our payload in all styles (' +
        results.map(r => r.style).join(', ') + ')');
}

main().catch(e => { console.error('ERROR', e); process.exit(2); });
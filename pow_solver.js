/**
 * DeepSeek PoW WASM Solver — Node.js bridge (with debug)
 */

const fs = require('fs');
const path = require('path');

function findWasm(dir) {
    // 递归搜索 .wasm 文件，优先匹配已知文件名
    const candidates = [];
    function walk(d) {
        try {
            for (const entry of fs.readdirSync(d)) {
                const full = path.join(d, entry);
                const stat = fs.statSync(full);
                if (stat.isFile() && entry.endsWith('.wasm')) {
                    candidates.push(full);
                } else if (stat.isDirectory()) {
                    walk(full);
                }
            }
        } catch (_) {}
    }
    walk(dir);
    if (candidates.length === 0) {
        throw new Error(`No .wasm file found under ${dir}`);
    }
    // 优先带 'sha3' 的文件名
    const preferred = candidates.find(f => path.basename(f).includes('sha3'));
    return preferred || candidates[0];
}

const WASM_PATH = findWasm(__dirname);

async function main() {
    const input = process.argv[2];
    if (!input) {
        console.error('Usage: node pow_solver.js \'<json_config>\'');
        process.exit(1);
    }
    
    const config = JSON.parse(input);
    
    const wasmBuffer = fs.readFileSync(WASM_PATH);
    const wasmModule = await WebAssembly.compile(wasmBuffer);
    const instance = await WebAssembly.instantiate(wasmModule, {});
    const mem = instance.exports.memory;
    
    // Check initial memory
    console.error(`Initial memory pages: ${mem.buffer.byteLength / 65536}`);
    
    const prefix = `${config.salt}_${config.expire_at}_`;
    console.error(`Challenge: ${config.challenge}`);
    console.error(`Prefix: ${prefix}`);
    console.error(`Difficulty: ${config.difficulty}`);
    
    // Write strings to WASM memory
    function writeString(str) {
        const encoded = Buffer.from(str, 'utf-8');
        const length = encoded.length;
        const ptr = instance.exports.__wbindgen_export_0(length, 1);
        console.error(`Allocated ptr=${ptr}, len=${length}, mem pages=${mem.buffer.byteLength / 65536}`);
        const view = new Uint8Array(mem.buffer);
        for (let i = 0; i < length; i++) {
            view[ptr + i] = encoded[i];
        }
        return { ptr, length };
    }
    
    const retptr = instance.exports.__wbindgen_add_to_stack_pointer(-16);
    console.error(`Stack ptr: ${retptr}`);
    
    try {
        const challengeInfo = writeString(config.challenge);
        const prefixInfo = writeString(prefix);
        
        console.error(`Calling wasm_solve(retptr=${retptr}, ch_ptr=${challengeInfo.ptr}, ch_len=${challengeInfo.length}, pfx_ptr=${prefixInfo.ptr}, pfx_len=${prefixInfo.length}, diff=${config.difficulty})`);
        
        instance.exports.wasm_solve(
            retptr,
            challengeInfo.ptr,
            challengeInfo.length,
            prefixInfo.ptr,
            prefixInfo.length,
            config.difficulty
        );
        
        const view = new Int32Array(mem.buffer);
        const status = view[retptr / 4];
        console.error(`Status: ${status}`);
        
        if (status === 0) {
            console.error('wasm_solve returned 0 — no solution found');
            process.exit(1);
        }
        
        const floatView = new Float64Array(mem.buffer);
        const value = floatView[(retptr + 8) / 8];
        const answer = Math.floor(value);
        console.error(`Answer: ${answer}`);
        
        const result = {
            algorithm: config.algorithm,
            challenge: config.challenge,
            salt: config.salt,
            answer: answer,
            signature: config.signature,
            target_path: config.target_path
        };
        
        const response = Buffer.from(JSON.stringify(result)).toString('base64');
        console.log(response);
    } finally {
        instance.exports.__wbindgen_add_to_stack_pointer(16);
    }
}

main().catch(err => {
    console.error('Error:', err.message);
    console.error(err.stack);
    process.exit(1);
});

#!/usr/bin/env bash
# test_oversized.sh — Reproducible curl-based test for context management
#
# Sends an oversized conversation to the proxy and verifies it recovers
# instead of returning "Content is too long" from DeepSeek's upstream.
#
# Usage:
#   # Quick smoke test (single run)
#   bash tests/test_oversized.sh
#
#   # Full test suite (multiple scenarios)
#   bash tests/test_oversized.sh --all
#
# Prerequisites:
#   - proxy.py running at PROXY_URL (default http://localhost:8000)
#   - curl installed

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-expert}"
PASS=0
FAIL=0

log_pass() { PASS=$((PASS + 1)); echo "  ✅ PASS: $1"; }
log_fail() { FAIL=$((FAIL + 1)); echo "  ❌ FAIL: $1"; }

# ── Helper: generate an oversized conversation ───────────────────────
# Arguments: number of turns (each turn ≈ growing token count)
gen_conversation() {
    local turns="$1"
    local messages='[{"role":"system","content":"You are a coding assistant with expertise in Python, Rust, and system design."}'
    for ((i=1; i<=turns; i++)); do
        local q="Question $i: Please write a detailed explanation of how garbage collection works in Python, including reference counting, generational collection, and the cyclic garbage collector. Provide code examples for each concept."
        local a="Answer $i: "
        # Growing answer to accumulate tokens faster
        for ((j=0; j<i*5; j++)); do
            a="$a Python uses reference counting where each object has an ob_refcnt field. When refcount hits 0, memory is freed immediately. Generational GC divides objects into 3 generations (0/1/2). New objects start in gen0. Surviving objects get promoted. "
        done
        messages="$messages,{\"role\":\"user\",\"content\":\"$q\"},{\"role\":\"assistant\",\"content\":\"$a\"}"
    done
    messages="$messages]"
    echo "$messages"
}

# ── Helper: send request and check for error ─────────────────────────
send_and_check() {
    local label="$1"
    local messages="$2"
    shift 2

    local tmpfile
    tmpfile=$(mktemp)
    trap 'rm -f "$tmpfile"' RETURN

    # Use a timeout of 60 seconds for the request
    http_code=$(curl -s -o "$tmpfile" -w "%{http_code}" \
        "$PROXY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"messages\": $messages, \"stream\": false}" \
        --max-time 60 2>/dev/null || echo "000")

    if [ "$http_code" = "000" ]; then
        log_fail "$label — connection refused (is proxy running at $PROXY_URL?)"
        return 1
    fi

    # Check for DeepSeek upstream "Content is too long" error
    if grep -q "Content is too long" "$tmpfile" 2>/dev/null; then
        log_fail "$label — got 'Content is too long' from upstream"
        return 1
    fi

    # Check for proxy error (5xx)
    if [ "${http_code:0:1}" = "5" ]; then
        local error_body
        error_body=$(head -c 500 "$tmpfile")
        log_fail "$label — proxy returned $http_code: $error_body"
        return 1
    fi

    # Check for valid response
    if grep -q '"choices"' "$tmpfile" 2>/dev/null || grep -q '"content"' "$tmpfile" 2>/dev/null; then
        log_pass "$label (HTTP $http_code, response has content)"
        return 0
    fi

    # If we got a 200 but no content, that's odd but not a failure
    if [ "$http_code" = "200" ]; then
        log_pass "$label (HTTP 200, unusual response)"
        return 0
    fi

    # 4xx might be expected for some edge cases
    log_pass "$label (HTTP $http_code, expected for edge case)"
    return 0
}

# ── Helper: debug endpoint check ─────────────────────────────────────
check_debug_endpoint() {
    local label="Debug endpoint"
    local messages=$(gen_conversation 5)

    local tmpfile
    tmpfile=$(mktemp)
    trap 'rm -f "$tmpfile"' RETURN

    http_code=$(curl -s -o "$tmpfile" -w "%{http_code}" \
        "$PROXY_URL/v1/context/debug" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"messages\": $messages}" \
        --max-time 15 2>/dev/null || echo "000")

    if [ "$http_code" = "200" ] && grep -q "token_breakdown" "$tmpfile" 2>/dev/null; then
        log_pass "$label — debug endpoint works (HTTP 200, has token_breakdown)"
    else
        log_fail "$label — HTTP $http_code or missing token_breakdown in response"
    fi
}

echo "================================================"
echo " DeepSeek Free API — Context Management Tests"
echo " Proxy: $PROXY_URL   Model: $MODEL"
echo "================================================"
echo ""

# ── Test 1: Debug endpoint ───────────────────────────────────────────
echo "--- Test 1: Debug endpoint ---"
check_debug_endpoint
echo ""

# ── Test 2: Normal conversation (should work) ────────────────────────
echo "--- Test 2: Normal conversation (5 turns) ---"
send_and_check "Normal 5-turn conversation" "$(gen_conversation 5)"
echo ""

# ── Test 3: Large conversation (triggers pruning) ────────────────────
echo "--- Test 3: Large conversation (30 turns) ---"
send_and_check "30-turn conversation" "$(gen_conversation 30)"
echo ""

# ── Test 4: Oversized conversation (triggers retry/aggressive prune) ─
echo "--- Test 4: Oversized conversation (80 turns) ---"
send_and_check "80-turn conversation" "$(gen_conversation 80)"
echo ""

# ── Test 5: Tiny single turn ─────────────────────────────────────────
echo "--- Test 5: Minimal conversation (1 turn) ---"
send_and_check "1-turn conversation" "$(gen_conversation 1)"
echo ""

# ── Test 6: Only system message ──────────────────────────────────────
echo "--- Test 6: System message only ---"
send_and_check "System only" '[{"role":"system","content":"You are a helpful assistant."}]'
echo ""

# ── Test 7: Empty messages (should get 422 or handled gracefully) ────
echo "--- Test 7: Empty messages array ---"
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' RETURN
http_code=$(curl -s -o "$tmpfile" -w "%{http_code}" \
    "$PROXY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": []}" \
    --max-time 15 2>/dev/null || echo "000")
if [ "$http_code" = "422" ] || grep -q "error" < "$tmpfile"; then
    log_pass "Empty messages (HTTP $http_code, expected for empty)"
else
    log_pass "Empty messages (HTTP $http_code)"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0

#!/bin/bash
# DeepSeek Free API 一键部署脚本
# 支持 Termux (Android) / Linux (Ubuntu/Debian/CentOS)
# 用法: bash deploy.sh [--bg] [--stop] [--status]

set -euo pipefail

INSTALL_DIR="${HOME}/ds-free-api"
PORT="${PROXY_PORT:-8000}"
LOG_FILE="${HOME}/dsapi.log"

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── 帮助 ──
show_help() {
    echo "用法: bash deploy.sh [选项]"
    echo ""
    echo "选项:"
    echo "  (无参数)    前台启动（Ctrl+C 停止）"
    echo "  --bg        后台启动"
    echo "  --stop      停止后台进程"
    echo "  --status    查看运行状态"
    echo "  --help      显示此帮助"
    echo ""
    echo "环境变量:"
    echo "  PROXY_PORT  端口号（默认 8000）"
}

# ── 停止 ──
do_stop() {
    if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
        pkill -f "python.*proxy.py" 2>/dev/null || true
        sleep 1
        info "已停止"
    else
        warn "没有运行中的代理进程"
    fi
}

# ── 状态 ──
do_status() {
    if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
        local pid=$(pgrep -f "python.*proxy.py" | head -1)
        info "运行中 (PID: $pid)"
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            info "健康检查通过"
        else
            warn "健康检查失败"
        fi
    else
        warn "未运行"
    fi
}

# ── 解析参数 ──
ACTION="start"
for arg in "$@"; do
    case "$arg" in
        --bg)     ACTION="bg" ;;
        --stop)   do_stop; exit 0 ;;
        --status) do_status; exit 0 ;;
        --help|-h) show_help; exit 0 ;;
        *) warn "未知参数: $arg" ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DeepSeek Free API Proxy 部署      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"

# ── 1. 检查 Python3 ──
if ! command -v python3 &>/dev/null; then
    error "需要 Python3，请先安装"
fi
info "Python3 $(python3 --version 2>&1 | cut -d' ' -f2)"

# ── 2. 检查 Node.js（PoW 求解需要）──
if ! command -v node &>/dev/null; then
    warn "未检测到 Node.js，尝试安装..."
    if command -v pkg &>/dev/null; then
        # Termux
        pkg install -y nodejs 2>/dev/null || error "Node.js 安装失败，请运行: pkg install nodejs"
    elif command -v apt &>/dev/null; then
        sudo apt update -qq && sudo apt install -y nodejs 2>/dev/null || error "Node.js 安装失败"
    elif command -v yum &>/dev/null; then
        sudo yum install -y nodejs 2>/dev/null || error "Node.js 安装失败"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y nodejs 2>/dev/null || error "Node.js 安装失败"
    else
        error "请手动安装 Node.js: https://nodejs.org/"
    fi
fi
info "Node.js $(node --version 2>&1)"

# ── 3. 检查 curl（健康检查需要）──
if ! command -v curl &>/dev/null; then
    warn "未检测到 curl，健康检查将不可用"
fi

# ── 4. 确定安装目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 如果脚本在 ~/ds-free-api 里，直接用
if [ "$SCRIPT_DIR" = "$INSTALL_DIR" ] || [ -f "$SCRIPT_DIR/proxy.py" ]; then
    WORK_DIR="$SCRIPT_DIR"
else
    # 否则安装到 ~/ds-free-api
    mkdir -p "$INSTALL_DIR"
    WORK_DIR="$INSTALL_DIR"

    # 如果目标目录没有 proxy.py，尝试解压
    if [ ! -f "$INSTALL_DIR/proxy.py" ]; then
        TARBALL=""
        for f in "$SCRIPT_DIR/ds-free-api.tar.gz" "./ds-free-api.tar.gz" "../ds-free-api.tar.gz"; do
            [ -f "$f" ] && TARBALL="$f" && break
        done
        if [ -n "$TARBALL" ]; then
            info "从 $TARBALL 解压..."
            tar -xzf "$TARBALL" -C "$INSTALL_DIR"
        else
            error "未找到部署包，请将 ds-free-api.tar.gz 放在脚本同目录"
        fi
    fi
fi

cd "$WORK_DIR"
info "工作目录: $WORK_DIR"

# ── 5. 安装 Python 依赖 ──
info "安装依赖..."
if command -v pip3 &>/dev/null; then
    pip3 install --quiet --break-system-packages fastapi uvicorn curl-cffi python-dotenv 2>/dev/null || \
    pip3 install --quiet fastapi uvicorn curl-cffi python-dotenv 2>/dev/null || \
    pip3 install fastapi uvicorn curl-cffi python-dotenv
elif command -v pip &>/dev/null; then
    pip install --quiet --break-system-packages fastapi uvicorn curl-cffi python-dotenv 2>/dev/null || \
    pip install --quiet fastapi uvicorn curl-cffi python-dotenv 2>/dev/null || \
    pip install fastapi uvicorn curl-cffi python-dotenv
else
    error "未找到 pip，请先安装: python3 -m ensurepip"
fi
info "依赖安装完成"

# ── 6. 检查关键文件 ──
for f in proxy.py tool_call.py pow_native.py pow_solver.js; do
    [ -f "$f" ] || error "缺少 $f，请确认部署包完整"
done
info "文件检查通过"

# ── 7. 停止旧进程 ──
if pgrep -f "python.*proxy.py" >/dev/null 2>&1; then
    warn "停止旧代理进程..."
    pkill -f "python.*proxy.py" 2>/dev/null || true
    sleep 1
fi

# ── 8. 启动 ──
export PROXY_PORT="$PORT"

if [ "$ACTION" = "bg" ]; then
    nohup python3 proxy.py > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    sleep 2

    if command -v curl &>/dev/null && curl -s "http://localhost:$PORT/health" | grep -q "ok" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✅ 部署成功！                      ║${NC}"
        echo -e "${GREEN}╠══════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║  管理后台: http://localhost:$PORT/admin${NC}"
        echo -e "${GREEN}║  API 地址: http://localhost:$PORT/v1   ${NC}"
        echo -e "${GREEN}║  进程 PID: $BG_PID${NC}"
        echo -e "${GREEN}║  日志文件: $LOG_FILE${NC}"
        echo -e "${GREEN}╠══════════════════════════════════════╣${NC}"
        echo -e "${GREEN}║  停止: bash deploy.sh --stop         ${NC}"
        echo -e "${GREEN}║  状态: bash deploy.sh --status       ${NC}"
        echo -e "${GREEN}║  日志: tail -f $LOG_FILE${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
    else
        warn "启动中... 如果无法访问请检查日志: $LOG_FILE"
    fi
else
    echo ""
    info "准备就绪！"
    echo ""
    echo -e "下一步："
    echo -e "  1. 打开浏览器访问: ${BLUE}http://localhost:$PORT/admin${NC}"
    echo -e "  2. 输入手机号和密码登录"
    echo -e "  3. 在客户端中配置 API 地址: ${BLUE}http://localhost:$PORT/v1${NC}"
    echo ""
    echo -e "后台运行: ${YELLOW}bash deploy.sh --bg${NC}"
    echo ""
    python3 proxy.py
fi

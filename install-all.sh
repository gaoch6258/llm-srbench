#!/usr/bin/env bash
set -euo pipefail
# Claude Code 一键安装脚本
# 自动执行所有必要的安装步骤
# Change to the directory where this script is located
cd "$(dirname "$0")"
PLATFORM="claude"
PLATFORM_NAME="Claude Code"
# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "\${GREEN}==========================================\${NC}"
echo -e "\${GREEN}   Claude Code CLI 一键安装\${NC}"
echo -e "\${GREEN}==========================================\${NC}"
echo ""
echo "此脚本将自动完成以下操作："
echo "1. 安装依赖环境 (Node.js, Git)"
echo "2. 安装 Claude Code CLI"
echo "3. 配置环境变量"
echo ""
# Display current directory for debugging
echo "[DEBUG] Current directory: $(pwd)"
echo "[DEBUG] Checking scripts folder..."
if [ -d "scripts" ]; then
    echo "[DEBUG] scripts folder exists"
    ls -la scripts/
else
    echo "[ERROR] scripts folder does not exist!"
fi
echo ""
# 步骤 1: 安装依赖环境
echo -e "\${CYAN}==========================================\${NC}"
echo -e "\${CYAN}步骤 1/3: 安装依赖环境\${NC}"
echo -e "\${CYAN}==========================================\${NC}"
echo ""
DEPENDENCY_SCRIPT="scripts/claude-dependencies-install.sh"
echo "[DEBUG] Looking for: $DEPENDENCY_SCRIPT"
if [ -f "$DEPENDENCY_SCRIPT" ]; then
    echo "[DEBUG] File found! Executing..."
    echo -e "\${YELLOW}[INFO] 执行依赖安装脚本...\${NC}"
    chmod +x "$DEPENDENCY_SCRIPT"
    "$DEPENDENCY_SCRIPT"
    if [ $? -ne 0 ]; then
        echo -e "\${RED}[ERROR] 依赖环境安装失败\${NC}"
        exit 1
    fi
else
    echo -e "\${YELLOW}[WARNING] 依赖安装脚本不存在，跳过此步骤\${NC}"
    echo "[DEBUG] File does not exist: $DEPENDENCY_SCRIPT"
fi
echo ""
# 步骤 2: 安装 CLI 工具
echo -e "\${CYAN}==========================================\${NC}"
echo -e "\${CYAN}步骤 2/3: 安装 Claude Code CLI\${NC}"
echo -e "\${CYAN}==========================================\${NC}"
echo ""
echo -e "\${YELLOW}[INFO] 使用 npm 安装...\${NC}"
if npm install -g @anthropic-ai/claude-code; then
    echo -e "\${GREEN}[SUCCESS] CLI 安装完成\${NC}"
else
    echo -e "\${RED}[ERROR] Claude Code CLI 安装失败\${NC}"
    exit 1
fi
echo ""
# 步骤 3: 配置环境变量
echo -e "\${CYAN}==========================================\${NC}"
echo -e "\${CYAN}步骤 3/3: 配置环境变量\${NC}"
echo -e "\${CYAN}==========================================\${NC}"
echo ""
ENV_SCRIPT="scripts/claude-env-setup.sh"
echo "[DEBUG] Looking for: $ENV_SCRIPT"
if [ -f "$ENV_SCRIPT" ]; then
    echo "[DEBUG] File found! Executing..."
    echo -e "\${YELLOW}[INFO] 执行环境变量配置脚本...\${NC}"
    chmod +x "$ENV_SCRIPT"
    "$ENV_SCRIPT"
    if [ $? -ne 0 ]; then
        echo -e "\${RED}[ERROR] 环境变量配置失败\${NC}"
        exit 1
    fi
else
    echo -e "\${YELLOW}[WARNING] 环境变量配置脚本不存在，跳过此步骤\${NC}"
    echo "[DEBUG] File does not exist: $ENV_SCRIPT"
fi
echo ""
echo -e "\${GREEN}==========================================\${NC}"
echo -e "\${GREEN}   安装完成！\${NC}"
echo -e "\${GREEN}==========================================\${NC}"
echo ""
echo "下一步操作："
echo "1. 重新加载环境变量: source ~/.bashrc（或 ~/.zshrc）"
echo "2. 运行: claude --version"
echo "3. 运行: claude 'Hello' 测试连接"
echo "4. 访问文档: https://coder.visioncoder.cn/docx"
echo ""
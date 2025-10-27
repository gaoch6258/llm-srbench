#!/usr/bin/env bash

# Claude Code 依赖环境安装脚本
# 安装 Node.js、Git、Claude Code
# 版本: 1.1.0
# 作者:  Team

# 注意：此脚本仅安装依赖，不包含配置功能
# 安装完成后请使用配置脚本设置 API Key

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 常量定义
NODE_REQUIRED_VERSION="18"
NODE_INSTALL_VERSION="22.9.0"
CLAUDE_CODE_PKG_NAME="@anthropic-ai/claude-code"

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 网络连通性检查
check_network_connectivity() {
    log_info "检查网络连通性..."
    
    # 检查npm仓库连通性
    if ! curl -s --connect-timeout 10 https://registry.npmjs.org/ > /dev/null; then
        log_error "无法访问 npm 仓库，请检查网络连接"
        log_info "建议："
        log_info "1. 检查网络连接"
        log_info "2. 使用科学上网工具"
        log_info "3. 稍后重试安装"
        return 1
    fi
    
    log_success "网络连通性检查通过"
    return 0
}

# 显示欢迎信息
show_welcome() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  Claude Code 依赖环境安装脚本"
    echo "=========================================="
    echo -e "${NC}"
    echo "此脚本将帮助您安装："
    echo "1. Node.js (≥v18)"
    echo "2. Git"
    echo "3. Claude Code"
    echo ""
    echo "⚠️  注意：此脚本仅安装依赖，不包含 API 配置"
    echo "安装完成后请使用环境配置脚本设置 API Key"
    echo ""
    echo -e "${YELLOW}🌐 网络提醒：${NC}"
    echo "本安装脚本需要从国外服务器下载安装包"
    echo "如遇下载超时或连接失败，建议："
    echo "• 使用科学上网工具"
    echo "• 或稍后重试安装脚本"
    echo ""
    echo -e "${CYAN}📚 需要帮助？访问我们的文档：${NC}"
    echo -e "${YELLOW}• 详细文档: https://coder.visioncoder.cn/docx${NC}"
    echo -e "${YELLOW}• 常见问题: https://coder.visioncoder.cn/start${NC}"
    echo -e "${YELLOW}• 使用手册: https://coder.visioncoder.cn/manual${NC}"
    echo ""
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# 版本比较函数
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C 2>/dev/null
}

# 设置 npm 用户级全局安装
setup_npm_user_global() {
    log_info "配置 npm 用户级全局目录..."
    
    local npm_global_dir="$HOME/.npm-global"
    mkdir -p "$npm_global_dir"
    npm config set prefix "$npm_global_dir"
    
    # 更新 PATH
    export PATH="$npm_global_dir/bin:$PATH"
    
    # 添加到 shell 配置文件
    local shell_config=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        shell_config="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        shell_config="$HOME/.bashrc"
    fi
    
    if [[ -n "$shell_config" ]] && [[ -w "$shell_config" ]]; then
        if ! grep -q "npm-global/bin" "$shell_config"; then
            echo "export PATH=$HOME/.npm-global/bin:$PATH" >> "$shell_config"
            log_success "PATH 已添加到 $shell_config"
        fi
    fi
    
    log_success "npm 用户级全局目录配置完成"
}

# 检查并安装 Homebrew (macOS)
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        log_info "未检测到 Homebrew，开始自动安装..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            log_success "Homebrew 安装完成"
        else
                    log_success "Homebrew 已安装"
    fi
}

# 检查并安装 Node.js
check_install_nodejs() {
    log_info "检查 Node.js 安装状态..."
    
    if command -v node &> /dev/null; then
        local current_version=$(node -v | sed 's/v//')
        local major_version=$(echo $current_version | cut -d. -f1)
        
        log_info "当前 Node.js 版本: v$current_version"
        
        if [ "$major_version" -ge "$NODE_REQUIRED_VERSION" ]; then
            log_success "Node.js 版本满足要求 (≥v$NODE_REQUIRED_VERSION)"
            # 检查 npm 是否可用
            if command -v npm &> /dev/null; then
                log_success "npm 已安装，版本: $(npm -v)"
                return 0
            else
                log_warning "Node.js 已安装但缺少 npm，需要重新安装"
            fi
        else
            log_warning "Node.js 版本过低 (v$current_version < v$NODE_REQUIRED_VERSION)，需要升级到 v$NODE_INSTALL_VERSION"
        fi
    else
        log_info "未检测到 Node.js，开始自动安装 v$NODE_INSTALL_VERSION"
    fi
    
            install_nodejs
    }

# 安装 Node.js
install_nodejs() {
    local os=$(detect_os)
    
    case $os in
        "macos")
            check_homebrew
            log_info "通过 Homebrew 安装 Node.js..."
            brew install node@22 || brew upgrade node@22
            ;;
        "linux")
            log_info "安装 Node.js v$NODE_INSTALL_VERSION..."
            
            # 检测 Linux 发行版
            if command -v apt-get &> /dev/null; then
                # Ubuntu/Debian 系统
                log_info "检测到 Ubuntu/Debian 系统"
                
                # 彻底移除旧版本
                log_info "清理旧版本 Node.js..."
                sudo apt-get remove -y nodejs npm nodejs-dev node-gyp libssl1.0-dev 2>/dev/null || true
                sudo apt-get purge -y nodejs npm nodejs-dev 2>/dev/null || true
                sudo apt-get autoremove -y 2>/dev/null || true
                
                # 清理残留文件和软链接
                sudo rm -rf /usr/local/bin/npm /usr/local/share/man/man1/node* /usr/local/lib/dtrace/node.d ~/.npm 2>/dev/null || true
                sudo rm -rf /usr/local/lib/node* /usr/local/bin/node /usr/local/include/node* 2>/dev/null || true
                sudo rm -f /usr/bin/node /usr/bin/npm 2>/dev/null || true
                
                # 安装必要的依赖
                sudo apt-get update
                sudo apt-get install -y curl ca-certificates gnupg
                
                # 添加 NodeSource 仓库
                log_info "添加 NodeSource 仓库..."
                curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
                
                # 安装 Node.js
                log_info "安装 Node.js 22.x..."
                sudo apt-get install -y nodejs
                
            elif command -v yum &> /dev/null; then
                # CentOS/RHEL 系统
                log_info "检测到 CentOS/RHEL 系统"
                sudo yum remove -y nodejs npm 2>/dev/null || true
                curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
                sudo yum install -y nodejs
                
            elif command -v dnf &> /dev/null; then
                # Fedora 系统
                log_info "检测到 Fedora 系统"
                sudo dnf remove -y nodejs npm 2>/dev/null || true
                curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
                sudo dnf install -y nodejs
                
            else
                log_error "不支持的 Linux 发行版，请手动安装 Node.js"
                exit 1
            fi
            
            # 强制刷新环境变量和命令缓存
            log_info "刷新环境变量..."
            hash -r 2>/dev/null || true
            export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
            
            # 等待系统更新
            sleep 3
            
            # 验证安装
            log_info "验证 Node.js 安装..."
            local node_found=false
            local npm_found=false
            
            # 检查多个可能的路径
            for node_path in "/usr/bin/node" "/usr/local/bin/node" "$(which node 2>/dev/null)"; do
                if [[ -x "$node_path" ]]; then
                    log_info "找到 Node.js: $node_path"
                    local new_version=$($node_path -v)
                    log_info "Node.js 版本: $new_version"
                    local new_major=$(echo $new_version | sed 's/v//' | cut -d. -f1)
                    if [ "$new_major" -ge "$NODE_REQUIRED_VERSION" ]; then
                        node_found=true
                        export PATH="$(dirname "$node_path"):$PATH"
                        break
                    fi
                fi
            done
            
            # 检查 npm
            for npm_path in "/usr/bin/npm" "/usr/local/bin/npm" "$(which npm 2>/dev/null)"; do
                if [[ -x "$npm_path" ]]; then
                    log_info "找到 npm: $npm_path"
                    npm_found=true
                    export PATH="$(dirname "$npm_path"):$PATH"
                    break
                fi
            done
            
            if [[ "$node_found" == false ]]; then
                log_error "Node.js 安装失败或版本不满足要求"
                exit 1
            fi
            
            if [[ "$npm_found" == false ]]; then
                log_error "npm 安装失败"
                exit 1
            fi
            
            ;;
        *)
            log_error "不支持的操作系统，请手动安装 Node.js"
            exit 1
            ;;
    esac
    
    # 最终验证和环境变量设置
    log_info "最终验证安装结果..."
    
    # 强制刷新命令缓存
    hash -r 2>/dev/null || true
    
    # 验证 Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node -v)
        log_success "Node.js 安装完成，版本: $node_version"
        
        # 验证版本要求
        local major_version=$(echo $node_version | sed 's/v//' | cut -d. -f1)
        if [ "$major_version" -lt "$NODE_REQUIRED_VERSION" ]; then
            log_error "Node.js 版本过低: $node_version (要求 ≥v$NODE_REQUIRED_VERSION)"
            exit 1
        fi
    else
        log_error "Node.js 安装失败 - 命令不可用"
        exit 1
    fi
    
    # 验证 npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm -v)
        log_success "npm 安装完成，版本: $npm_version"
    else
        log_error "npm 安装失败 - 命令不可用"
        log_info "这可能是 PATH 环境变量问题，请尝试重新登录或运行: source ~/.bashrc"
        exit 1
    fi
}

# 检查并安装 Git
check_install_git() {
    log_info "检查 Git 安装状态..."
    
    if command -v git &> /dev/null; then
        log_success "Git 已安装，版本: $(git --version)"
        return 0
    fi
    
    log_info "未检测到 Git，开始自动安装..."
            install_git
    }

# 安装 Git
install_git() {
    local os=$(detect_os)
    
    case $os in
        "macos")
            check_homebrew
            log_info "通过 Homebrew 安装 Git..."
            brew install git
            ;;
        "linux")
            log_info "安装 Git..."
            sudo apt-get update
            sudo apt-get install -y git
            ;;
        *)
            log_error "不支持的操作系统，请手动安装 Git"
            exit 1
            ;;
    esac
    
    if command -v git &> /dev/null; then
        log_success "Git 安装完成，版本: $(git --version)"
    else
        log_error "Git 安装失败"
        exit 1
    fi
}

# 检查并安装 Claude Code
check_install_claude_code() {
    log_info "检查 Claude Code 安装状态..."
    
    if command -v claude &> /dev/null; then
        local version=$(claude --version 2>/dev/null | head -1 || echo "unknown")
        log_success "Claude Code 已安装，版本: $version"
        return 0
    fi
    
    log_info "未检测到 Claude Code，开始自动安装..."
        install_claude_code
    }

# 安装 Claude Code
install_claude_code() {
    log_info "安装 Claude Code..."
    
    # 检查 npm 是否可用
    if ! command -v npm &> /dev/null; then
        log_error "npm 命令不可用，无法安装 Claude Code"
        log_info "请确保Node.js和npm已正确安装并在PATH中"
        log_info "尝试运行: source ~/.bashrc 或重新登录"
        exit 1
    fi
    
    local os=$(detect_os)
    
    # 配置npm（特别是Linux系统）
    if [[ "$os" == "linux" ]]; then
        log_info "配置npm全局安装路径..."
        # 检查并创建用户级全局目录（避免权限问题）
        local npm_global_dir="$HOME/.npm-global"
        if [[ ! -d "$npm_global_dir" ]]; then
            mkdir -p "$npm_global_dir"
            npm config set prefix "$npm_global_dir"
            log_info "设置npm全局目录为: $npm_global_dir"
        fi
        
        # 更新PATH
        export PATH="$npm_global_dir/bin:$PATH"
    fi
    
    log_info "使用npm安装 Claude Code..."
    
    # 根据系统选择安装方式
    if [[ "$os" == "linux" ]]; then
        # Linux: 尝试用户级安装，失败则用sudo
        if npm install -g "$CLAUDE_CODE_PKG_NAME" 2>/dev/null; then
            log_success "用户级全局安装成功"
            else
                log_warning "用户级安装失败，尝试系统级安装..."
                if sudo npm install -g "$CLAUDE_CODE_PKG_NAME" --unsafe-perm=true --allow-root; then
                    log_success "系统级安装成功"
                else
                    log_error "Claude Code 安装失败"
                    log_info "请手动运行: npm install -g @anthropic-ai/claude-code"
                exit 1
            fi
        fi
    else
        # macOS
        if npm install -g "$CLAUDE_CODE_PKG_NAME"; then
            log_success "Claude Code安装成功"
        else
            log_error "Claude Code 安装失败"
            log_info "请手动运行: npm install -g @anthropic-ai/claude-code"
            exit 1
        fi
    fi
    
    # 等待安装完成
    sleep 3
    
    # 刷新命令缓存
    hash -r 2>/dev/null || true
    
    # 为Linux系统添加更多可能的路径
    if [[ "$os" == "linux" ]]; then
        local additional_paths=(
            "$HOME/.npm-global/bin"
            "/usr/local/bin"
            "/usr/bin"
            "$(npm root -g 2>/dev/null)/../.bin"
        )
        
        for path in "${additional_paths[@]}"; do
            if [[ -d "$path" ]]; then
                export PATH="$path:$PATH"
            fi
        done
    fi
    
    if command -v claude &> /dev/null; then
        local version=$(claude --version 2>/dev/null | head -1 || echo "unknown")
        log_success "Claude Code 安装完成，版本: $version"
        
        # 验证 Node.js 版本是否满足 Claude Code 要求
        if command -v node &> /dev/null; then
            local node_version=$(node -v | sed 's/v//')
            local node_major=$(echo $node_version | cut -d. -f1)
            if [ "$node_major" -ge "18" ]; then
                log_success "Node.js 版本 v$node_version 满足 Claude Code 要求"
            else
                log_warning "Node.js 版本 v$node_version 可能过低，Claude Code 需要 ≥v18"
            fi
        fi
    else
        log_error "Claude Code 安装失败"
        log_info "请检查："
        log_info "1. npm 全局安装目录权限"
        log_info "2. Node.js 版本是否 ≥v18"
        log_info "3. 网络连接是否正常"
        exit 1
    fi
}

# 获取 Claude Code 版本号
get_claude_code_version() {
    if command -v claude &> /dev/null; then
        claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
    else
        echo "0.0.0"
    fi
}

# 显示完成信息
show_completion() {
    echo -e "${GREEN}"
    echo "=========================================="
    echo "  依赖环境安装完成！"
    echo "=========================================="
    echo -e "${NC}"
    echo "已成功安装："
    echo "✅ Node.js: $(node --version 2>/dev/null || echo '未安装')"
    echo "✅ Git: $(git --version 2>/dev/null | head -1 || echo '未安装')"
    echo "✅ Claude Code: $(claude --version 2>/dev/null | head -1 || echo '未安装')"
    echo ""
    echo "🚀 下一步：使用环境配置脚本设置 API Key"
    echo "   ./scripts/claude-env-setup.sh"
    echo ""
    echo "💡 完整配置指南请访问："
    echo -e "${CYAN}   https://coder.visioncoder.cn/start${NC}"
    echo ""
    log_success "依赖环境安装完成！"
}

# 主函数
main() {
    show_welcome
    
    # 检测操作系统
    local os=$(detect_os)
    if [ "$os" == "unknown" ]; then
        log_error "不支持的操作系统"
        exit 1
    fi
    
    log_info "检测到操作系统: $os"
    
    # 检查网络连通性
    check_network_connectivity || log_warning "网络连通性检查失败，将尝试继续..."
    
    # 预检查需要安装的组件
    local need_nodejs=false
    local need_git=false
    local need_claude=false
    
    # 检查 Node.js
    if ! command -v node &> /dev/null; then
        need_nodejs=true
    else
        local current_version=$(node -v | sed 's/v//')
        local major_version=$(echo $current_version | cut -d. -f1)
        if [ "$major_version" -lt "$NODE_REQUIRED_VERSION" ]; then
            need_nodejs=true
        fi
    fi
    
    # 检查 Git
    if ! command -v git &> /dev/null; then
        need_git=true
    fi
    
    # 检查 Claude Code
    if ! command -v claude &> /dev/null; then
        need_claude=true
    fi
    
    # 显示需要安装的组件
    if [ "$need_nodejs" = true ] || [ "$need_git" = true ] || [ "$need_claude" = true ]; then
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}需要安装以下组件：${NC}"
        [ "$need_nodejs" = true ] && echo -e "${CYAN}  • Node.js v$NODE_INSTALL_VERSION${NC}"
        [ "$need_git" = true ] && echo -e "${CYAN}  • Git${NC}"
        [ "$need_claude" = true ] && echo -e "${CYAN}  • Claude Code${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        # 统一确认
        echo -e "${YELLOW}是否开始自动安装以上组件？ (y/N)${NC}"
        read -r response
        case "$response" in
            [yY]|[yY][eE][sS])
                log_info "开始安装..."
                ;;
            *)
                log_info "已取消自动安装"
                echo ""
                echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo -e "${CYAN}请手动安装以下环境：${NC}"
                echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo ""
                
                if [ "$need_nodejs" = true ]; then
                    echo -e "${YELLOW}📦 Node.js v$NODE_INSTALL_VERSION 或更高版本${NC}"
                    if [[ "$os" == "macos" ]]; then
                        echo -e "${GREEN}  brew install node@22${NC}"
                    elif [[ "$os" == "linux" ]]; then
                        echo -e "${GREEN}  # Ubuntu/Debian:${NC}"
                        echo -e "${GREEN}  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -${NC}"
                        echo -e "${GREEN}  sudo apt-get install -y nodejs${NC}"
                        echo ""
                        echo -e "${GREEN}  # CentOS/RHEL:${NC}"
                        echo -e "${GREEN}  curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -${NC}"
                        echo -e "${GREEN}  sudo yum install -y nodejs${NC}"
                    fi
                    echo ""
                fi
                
                if [ "$need_git" = true ]; then
                    echo -e "${YELLOW}📦 Git${NC}"
                    if [[ "$os" == "macos" ]]; then
                        echo -e "${GREEN}  brew install git${NC}"
                    elif [[ "$os" == "linux" ]]; then
                        echo -e "${GREEN}  # Ubuntu/Debian:${NC}"
                        echo -e "${GREEN}  sudo apt-get update && sudo apt-get install -y git${NC}"
                        echo ""
                        echo -e "${GREEN}  # CentOS/RHEL:${NC}"
                        echo -e "${GREEN}  sudo yum install -y git${NC}"
                    fi
                    echo ""
                fi
                
                if [ "$need_claude" = true ]; then
                    echo -e "${YELLOW}📦 Claude Code${NC}"
                    echo -e "${GREEN}  npm install -g @anthropic-ai/claude-code${NC}"
                    echo ""
                fi
                
                echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                exit 0
                ;;
        esac
        echo ""
    else
        log_success "所有依赖已安装，无需重复安装"
        show_completion
        exit 0
    fi
    
    # 主安装流程
    check_install_nodejs
    check_install_git
    check_install_claude_code
    
    # 测试安装
    if claude --help &> /dev/null; then
        log_success "Claude Code 测试通过"
    else
        log_warning "Claude Code 测试失败，请检查安装"
    fi
    
    show_completion
}

# 错误处理
trap 'log_error "脚本执行过程中发生错误，退出码: $?"' ERR

# 执行主函数
main "$@"
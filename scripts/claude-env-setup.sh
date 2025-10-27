#!/usr/bin/env bash
set -euo pipefail

# Claude Code 环境配置脚本
# 配置 Claude Code 环境变量

TOKEN="sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863"
BASE_URL="https://coder.api.visioncoder.cn"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 检测操作系统
OS_TYPE="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
  OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
  OS_TYPE="linux"
fi

# 根据操作系统和shell类型选择配置文件
RC_FILE=""

if [[ "${SHELL:-}" == *"zsh" ]]; then
  RC_FILE="$HOME/.zshrc"
elif [[ "${SHELL:-}" == *"bash" ]]; then
  # 智能检测bash配置文件
  if [[ "$OS_TYPE" == "macos" ]]; then
    # macOS: 检查用户实际使用的配置文件
    if [[ -f "$HOME/.bash_profile" && -s "$HOME/.bash_profile" ]]; then
      # .bash_profile 存在且非空，优先使用
      RC_FILE="$HOME/.bash_profile"
    elif [[ -f "$HOME/.bashrc" && -s "$HOME/.bashrc" ]]; then
      # .bashrc 存在且非空
      RC_FILE="$HOME/.bashrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
      # .bash_profile 存在但为空
      RC_FILE="$HOME/.bash_profile"
    else
      # 都不存在，创建 .bash_profile（macOS 传统）
      RC_FILE="$HOME/.bash_profile"
    fi
  else
    # Linux: 优先使用 .bashrc
    if [[ -f "$HOME/.bashrc" ]]; then
      RC_FILE="$HOME/.bashrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
      RC_FILE="$HOME/.bash_profile"
    else
      # 默认创建 .bashrc
      RC_FILE="$HOME/.bashrc"
    fi
  fi
elif [[ "${SHELL:-}" == *"fish" ]]; then
  RC_FILE="$HOME/.config/fish/config.fish"
  # 确保 fish 配置目录存在
  mkdir -p "$HOME/.config/fish"
else
  # 未知shell，根据系统和现有文件智能选择
  if [[ "$OS_TYPE" == "macos" ]]; then
    # macOS: 检查现有文件
    if [[ -f "$HOME/.zshrc" ]]; then
      RC_FILE="$HOME/.zshrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
      RC_FILE="$HOME/.bash_profile"
    elif [[ -f "$HOME/.bashrc" ]]; then
      RC_FILE="$HOME/.bashrc"
    else
      # 默认使用 zsh（macOS 默认）
      RC_FILE="$HOME/.zshrc"
    fi
  else
    # Linux: 检查现有文件
    if [[ -f "$HOME/.bashrc" ]]; then
      RC_FILE="$HOME/.bashrc"
    elif [[ -f "$HOME/.bash_profile" ]]; then
      RC_FILE="$HOME/.bash_profile"
    elif [[ -f "$HOME/.zshrc" ]]; then
      RC_FILE="$HOME/.zshrc"
    else
      # 默认使用 bashrc（Linux 常见）
      RC_FILE="$HOME/.bashrc"
    fi
  fi
fi

echo "操作系统: $OS_TYPE"
echo "Shell: ${SHELL:-unknown}"
echo "写入到: $RC_FILE"
echo ""
echo -e "${CYAN}📝 将要配置的环境变量：${NC}"
echo -e "${YELLOW}• ANTHROPIC_AUTH_TOKEN: sk-MKGtNl8Vm...${NC}"
echo -e "${YELLOW}• ANTHROPIC_API_KEY: sk-MKGtNl8Vm...${NC}"
echo -e "${YELLOW}• ANTHROPIC_BASE_URL: https://coder.api.visioncoder.cn${NC}"
echo ""
echo -e "${CYAN}📚 需要帮助？访问我们的文档：${NC}"
echo -e "${YELLOW}• 详细文档: https://coder.visioncoder.cn/docx${NC}"
echo -e "${YELLOW}• 常见问题: https://coder.visioncoder.cn/start${NC}"
echo -e "${YELLOW}• 使用手册: https://coder.visioncoder.cn/manual${NC}"
echo ""
touch "$RC_FILE"

BLOCK_START="# >>> CLAUDE ENV BEGIN >>>"
BLOCK_END="# <<< CLAUDE ENV END <<<"

# 根据Shell类型设置不同的环境变量语法
if [[ "${SHELL:-}" == *"fish" ]]; then
  # Fish shell 语法
  BLOCK_CONTENT=$(cat <<EOF
$BLOCK_START
# Claude Code 环境配置
set -x ANTHROPIC_AUTH_TOKEN "$TOKEN"
set -x ANTHROPIC_API_KEY "$TOKEN"
set -x ANTHROPIC_BASE_URL "$BASE_URL"
$BLOCK_END
EOF
  )
else
  # Bash/Zsh 语法
  BLOCK_CONTENT=$(cat <<EOF
$BLOCK_START
# Claude Code 环境配置
export ANTHROPIC_AUTH_TOKEN="$TOKEN"
export ANTHROPIC_API_KEY="$TOKEN"
export ANTHROPIC_BASE_URL="$BASE_URL"
$BLOCK_END
EOF
  )
fi

# 检查并更新配置文件
if grep -q "$BLOCK_START" "$RC_FILE" 2>/dev/null; then
  echo "更新现有配置块..."
  # 创建临时文件存储新内容
  TEMP_FILE="$RC_FILE.tmp"
  > "$TEMP_FILE"
  
  # 读取原文件并替换配置块
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "$BLOCK_START" ]]; then
      # 找到开始标记，写入新的配置块
      echo "$BLOCK_CONTENT" >> "$TEMP_FILE"
      # 跳过直到结束标记
      while IFS= read -r line; do
        if [[ "$line" == "$BLOCK_END" ]]; then
          break
        fi
      done
    else
      # 写入其他行
      echo "$line" >> "$TEMP_FILE"
    fi
  done < "$RC_FILE"
  
  # 替换原文件
  mv "$TEMP_FILE" "$RC_FILE"
else
  echo "添加新的配置块..."
  echo "" >> "$RC_FILE"
  echo "$BLOCK_CONTENT" >> "$RC_FILE"
fi

echo "✅ 已写入 $RC_FILE"
echo "👉 请手动执行以下命令让配置生效:"
if [[ "${SHELL:-}" == *"fish" ]]; then
  echo "   source ~/.config/fish/config.fish"
elif [[ "${SHELL:-}" == *"zsh" ]]; then
  echo "   source ~/.zshrc"
else
  echo "   source $RC_FILE"
fi
# 🎯 Claude Code 配置说明

> **🚀 一键配置 Claude Code 开发环境，支持所有版本和平台**

## 📋 包含文件

| 文件名 | 用途 | 适用版本 |
|--------|------|----------|
| `scripts/claude-dependencies-install.sh` | 🛠️ 依赖环境安装 | 所有版本 |
| `scripts/claude-env-setup.sh` | ⚙️ 环境变量配置 | ≥ 1.0.63 |
| `settings.json` | 📝 JSON 配置文件 | < 1.0.63 |
| `README.md` | 📚 使用说明 | - |

---

## ⚡ 快速开始

### 🛠️ 步骤 1: 安装运行环境

> 📌 首次使用必须运行，安装 Node.js、Git 和 Claude Code

```bash
chmod +x ./scripts/claude-dependencies-install.sh
./scripts/claude-dependencies-install.sh
```

✅ **自动安装内容:**
- Node.js (≥ v18.0.0, 推荐 v22.19.0 LTS)
- Git (最新版)
- Claude Code (@anthropic-ai/claude-code)

### ⚙️ 步骤 2: 配置 API 密钥

> 📋 根据你的 Claude Code 版本选择对应方法

**🔍 先检查版本:**
```bash
claude --version
```

**然后选择配置方法:**

| 版本范围 | 推荐方法 | 备用方法 |
|----------|----------|----------|
| **≥ 1.0.63** | 🎯 环境变量脚本 | ⚙️ 手动配置环境变量 |
| **< 1.0.63** | 📝 settings.json | - |

---

## 📋 版本配置指南

### 🎯 Claude Code ≥ 1.0.63 (推荐使用环境变量)

#### 方法一: 自动配置脚本 (推荐)

```bash
# 直接运行，API Key 已内置在脚本中
chmod +x ./scripts/claude-env-setup.sh
./scripts/claude-env-setup.sh

# 或临时指定不同的 Key (用于更新)
TOKEN="your-new-key" BASE_URL="您的API服务器地址" ./scripts/claude-env-setup.sh
```

#### 方法二: 手动配置环境变量

```bash
# 添加到你的 shell 配置文件 (.bashrc, .zshrc, .bash_profile)
export ANTHROPIC_API_KEY="sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863"
export ANTHROPIC_AUTH_TOKEN="sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863"
export ANTHROPIC_BASE_URL="https://coder.api.visioncoder.cn"

# 重新加载配置
source ~/.bashrc  # 或 ~/.zshrc, ~/.bash_profile
```

#### 方法三: settings.json (≥1.0.63 也支持)

> 💡 新版本同样支持 settings.json，可作为备用配置方式

```bash
mkdir -p ~/.config/claude-code
cp settings.json ~/.config/claude-code/
```

### 📝 Claude Code < 1.0.63 (仅支持 settings.json)

```bash
# 将配置文件复制到指定位置
mkdir -p ~/.config/claude-code
cp settings.json ~/.config/claude-code/
```

**配置文件内容预览:**
```json
{
  "env": {
    "ANTHROPIC_API_KEY": "sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863",
    "ANTHROPIC_BASE_URL": "https://coder.api.visioncoder.cn",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "CLAUDE_MODEL": "claude-opus-4-20250514"
  },
  "permissions": {
    "allow": [],
    "deny": []
  },
  "apiKeyHelper": "echo 'sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863'"
}
```

---

## 🛠️ 脚本详细说明

### 🛠️ scripts/claude-dependencies-install.sh

**功能:** 一键安装 Claude Code 运行环境

| 组件 | 版本要求 | 安装方式 |
|------|----------|----------|
| Node.js | ≥ 18.0.0 (推荐 22.19.0 LTS) | macOS: Homebrew / Linux: NodeSource |
| Git | 最新版本 | macOS: Homebrew / Linux: apt-get |
| Claude Code | 最新版本 | npm install -g @anthropic-ai/claude-code |

**平台支持:**
- ✅ macOS (Intel + Apple Silicon)
- ✅ Linux (Ubuntu, Debian, CentOS, RHEL, Fedora, Arch)
- ❌ Windows (请使用 WSL)

### ⚙️ /scripts/claude-env-setup.sh

**功能:** 智能配置环境变量

**智能特性:**
- 🎯 自动检测操作系统 (macOS/Linux)
- 🐚 自动识别 Shell 类型 (bash/zsh/fish)
- 📂 智能选择配置文件：
  - macOS bash: `.bash_profile` > `.bashrc`
  - Linux bash: `.bashrc` > `.bash_profile`
  - Zsh: `.zshrc`
  - Fish: `.config/fish/config.fish`
- 🔄 支持更新已有配置

**使用示例:**
```bash
# 使用内置 Key
./scripts/claude-env-setup.sh

# 临时指定新 Key (用于更新)
TOKEN="sk-ant-new-key" ./claude-env-setup.sh

# 指定完整配置
TOKEN="sk-ant-new-key" BASE_URL="https://coder.api.vayvi.cn" ./scripts/claude-env-setup.sh
```

---

## ⚙️ 手动配置参考

### 🌍 环境变量配置 (适用于 ≥ 1.0.63)

**必需变量:**
```bash
export ANTHROPIC_API_KEY="sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863"        # 你的 API 密钥
export ANTHROPIC_AUTH_TOKEN="sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863"     # 认证令牌 (同 API Key)
export ANTHROPIC_BASE_URL="https://coder.api.visioncoder.cn"       # API 基础地址
```

**配置文件位置:**
- **Zsh (macOS 默认):** `~/.zshrc`
- **Bash (macOS):** `~/.bash_profile` 或 `~/.bashrc`
- **Bash (Linux):** `~/.bashrc` 或 `~/.bash_profile`
- **Fish Shell:** `~/.config/fish/config.fish`

**重新加载配置:**
```bash
source ~/.zshrc        # Zsh
source ~/.bashrc       # Bash (Linux)
source ~/.bash_profile # Bash (macOS)
source ~/.config/fish/config.fish  # Fish
```

### 📝 settings.json 配置 (所有版本通用)

**配置路径:** `~/.config/claude-code/settings.json`

**完整配置示例:**
```json
{
  "env": {
    "ANTHROPIC_API_KEY": "sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863",
    "ANTHROPIC_BASE_URL": "https://coder.api.visioncoder.cn",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
    "CLAUDE_MODEL": "claude-opus-4-20250514"
  },
  "permissions": {
    "allow": [],
    "deny": []
  },
  "apiKeyHelper": "echo 'sk-MKGtNl8VmKb8FiatG3f6bCTuXdpT6Jkz72Cc79C7686a4a4fA861Ba38C43f2863'"
}
```

---

## 🚨 故障排除

### ❓ 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 🔴 配置不生效 | 环境变量未加载 | `source ~/.bashrc` 或重启终端 |
| 🔴 claude 命令未找到 | PATH 配置问题 | 重新安装: `npm install -g @anthropic-ai/claude-code` |
| 🔴 API 密钥无效 | 密钥错误或过期 | 检查密钥格式: `echo $ANTHROPIC_API_KEY` |
| 🔴 IDE 被占用 | 进程冲突 | 删除 `~/.claude/ide/*.lock` 文件 |
| 🔴 权限不足 | sudo 权限问题 | 脚本运行时输入管理员密码 |
| ⚠️ 环境更新失败 | 配置文件冲突 | 手动编辑配置文件或删除冲突行 |

> 💡 **关于环境更新失败：** 如果使用 `scripts/claude-env-setup.sh` 更新环境变量时失败，可能是因为配置文件中已存在冲突的设置。请手动打开对应的配置文件（如 `~/.zshrc` 或 `~/.bashrc`），删除旧的 `ANTHROPIC_*` 相关行，然后重新运行脚本。

### 🔍 诊断命令

```bash
# 检查 Claude Code 安装
claude --version
which claude

# 检查环境变量
echo "API Key: $ANTHROPIC_API_KEY"
echo "Base URL: $ANTHROPIC_BASE_URL"
echo "Auth Token: $ANTHROPIC_AUTH_TOKEN"

# 检查配置文件
cat ~/.config/claude-code/settings.json  # 如果使用 settings.json

# 检查 PATH
echo $PATH | tr ":" "\n" | grep -i node
```

### 🔄 重置配置

**完全重置 (谨慎使用):**
```bash
# 删除所有 Claude Code 配置
rm -rf ~/.config/claude-code/
rm -f ~/.claude.json
rm -rf ~/.claude/

# 重新安装
npm uninstall -g @anthropic-ai/claude-code
npm install -g @anthropic-ai/claude-code
```

### ✅ 验证安装

```bash
# 1. 验证基础安装
node --version    # 应该 ≥ 18.0.0
git --version     # 应该显示版本信息
claude --version  # 应该显示版本信息

# 2. 验证环境配置
claude --help     # 应该显示帮助信息

# 3. 测试基本功能 (在项目目录中)
claude           # 应该启动 Claude Code
```

---

---

## 📚 更多资源

- 📄 [详细文档](https://coder.visioncoder.cn/docx)
- ❓ [常见问题](https://coder.visioncoder.cn/start)
- 📖 [使用手册](https://coder.visioncoder.cn/manual)

---

*enjoy coding with Claude Code! 🚀*

---

🎉 **享受 Claude Code 带来的编程体验！**
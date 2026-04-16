# SciDataBot

> **面向通用科学数据准备的智能体系统** —— 一句话完成跨学科、多模态科学数据集建设。

## 研发背景

Openclaw、Nanobot 等项目以 **ReAct 循环**（Reasoning + Acting）为执行引擎，深度整合记忆（Memory）、工具（Tools）、技能（Skills）三大核心能力，彻底解决传统智能体场景单一、扩展性差、无法自主运行的痛点，是目前工业界和学术界主流的通用 Agent 实现方案之一。

## 研发目标

开发面向通用科学数据准备的智能体系统，实现**一句话数据准备能力**，支撑跨学科、多模态科学数据集建设。

SciDataBot 引入 ReAct 循环机制并设计高通量并发智能体架构，具备通用数据处理能力，核心特点：

| 特点 | 说明 |
|------|------|
| **高通量数据并行处理** | TaskPlanner 自动将批量任务分解为 N 条独立流水线并发执行，突破单线程处理瓶颈 |
| **多智能体协作架构** | MainAgent → TaskPlanner → N × Processor → Integrator 四层分工，各 Agent 异步解耦 |
| **SciDataCopilot Workflow** | 内置数据接入→质检→处理→整合→导出完整链路，一句话触发全流程 |
| **本地轻量化常驻部署** | 基于消息总线的事件驱动架构，支持全时在线、Cron 定时任务、多渠道接入 |

## 系统架构

### ReAct 执行引擎

```
用户一句话请求
       │
       ▼
  MainAgent ── ReAct 循环（Reasoning + Acting，最多 40 次迭代）
       │
       ├── 简单任务 ──────────────→ 直接调用工具返回结果
       │                            （read_file / exec / web_search …）
       │
       └── 复杂数据任务 ──→ spawn
                               │
                               ▼
                         TaskPlanner
                         （推理任务边界，输出并行 Pipeline JSON）
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
             Processor-1  Processor-2  Processor-N   ← 高通量并行
                  │            │            │
                  └────────────┴────────────┘
                               │
                               ▼
                          Integrator
                      （汇总结果，生成报告）
                               │
                               ▼
                             用户
```

### 消息总线与事件驱动

```
┌─────────────┐    InboundMessage     ┌──────────────┐
│  TUI / CLI  │ ──────────────────→   │  MessageBus  │
│  Feishu Bot │                       │  (双队列)     │
│  Webhook    │ ←──────────────────   │              │
└─────────────┘    OutboundMessage    └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │  MainAgent   │
                                      │  (ReAct核心) │
                                      └──────┬───────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │       SubagentManager       │
                              │  (asyncio Task 并发调度)     │
                              └─────────────────────────────┘
```

**三大核心能力集成：**
- **Memory**：双层记忆（`MEMORY.md` 长期事实 + `HISTORY.md` 可搜索日志），跨会话保持上下文
- **Tools**：21 个内置工具，涵盖数据接入、处理、整合、文件系统、Shell、Web 等
- **Skills**：分为两类Skill
  - 内置Skill：src/skills/builtin，main_agent必调用
  - 用户Skill：根据用户命令显式调用，或是在ReAct中根据理解调用
  - main_agent调用的skill完整传入subagnt

## 项目结构

```
scidatabot/
├── config.yaml.example         # 配置模板（复制为 config.yaml）
├── pyproject.toml              # 包管理（uv / pip）
├── scidatabot.sh               # 启动脚本
├── skills/                     # 用户个人skills
├── src/
│   ├── main.py                 # 应用入口 create_app()
│   ├── cli/__init__.py         # CLI（typer）：tui / run / channel 命令
│   ├── bus/                    # 消息总线
│   │   ├── events.py           # InboundMessage / OutboundMessage
│   │   └── queue.py            # MessageBus（asyncio.Queue 双队列）
│   ├── core/
│   │   ├── main_agent.py       # MainAgent —— ReAct 核心循环
│   │   ├── subagent.py         # SubagentManager —— 并发子 Agent 调度
│   │   ├── prompt_builder.py   # 系统提示构建（读取 templates/）
│   │   └── session.py          # Session 数据模型
│   ├── session/                # 会话管理
│   │   └── manager.py          # SessionManager —— JSONL 会话持久化
│   ├── config/                 # 配置加载与 Pydantic 模型
│   ├── providers/              # LLM Provider 抽象层
│   │   ├── base.py             # LLMProvider / LLMResponse / ToolCall
│   │   ├── anthropic.py        # Anthropic Claude（兼容第三方代理）
|   |   ├── google.py           # Google Gemini（兼容第三方代理）
|   |   ├── gork.py             # X Gork（兼容第三方代理）
│   │   ├── openai.py           # OpenAI gpt（兼容第三方代理）
│   │   ├── minimax.py          # MiniMax
│   │   ├── glm.py              # 智谱 GLM
│   │   ├── qwen.py             # 阿里 Qwen
│   │   ├── deepseek.py         # Deepseek
│   │   ├── kimi.py             # Kimi
│   │   └── intern_s1.py        # Shanghai Ailab Intern-S1
│   ├── tools/
│   │   ├── registry.py         # ToolRegistry —— 工具注册与动态调度
│   │   ├── base.py             # Tool 抽象基类
│   │   ├── general/            # 通用工具（fs / shell / web / spawn / cron）
│   │   ├── data_access/        # 数据接入（格式检测、元数据、质量评估）
│   │   ├── data_processing/    # 数据处理（抽取、转换、清洗、统计、MAT）
│   │   └── data_integration/   # 数据整合（时间/空间对齐、导出）
│   ├── channels/               # 渠道接入
│   │   ├── feishu_ws.py        # 飞书 WebSocket（长连接）
│   │   ├── feishu.py           # 飞书 HTTP API
│   │   ├── wechat.py           # 企业微信
│   │   └── webhook.py          # Webhook
│   ├── cron/                   # 定时任务服务（CronService + CronTool）
│   ├── skills/                 # 可插拔技能（内置skills必加载）
│   └── tui/simple_tui.py       # 命令行 TUI 界面
└── templates/                  # Agent 提示模板
    ├── AGENTS.md               # 主 Agent 角色与任务路由规则
    ├── SOUL.md / USER.md / TOOLS.md
    └── subagents/
        ├── TASK_PLANNER.md     # 并行任务分解规则
        ├── PROCESSOR.md        # 数据处理 Agent 提示
        └── INTEGRATOR.md       # 结果聚合 Agent 提示
```

## 安装

**推荐使用 [uv](https://github.com/astral-sh/uv)（更快）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo-url> scidatabot && cd scidatabot
uv sync
```

**或使用 pip：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

> 如环境中设置了 SOCKS 代理，安装前请执行 `unset ALL_PROXY all_proxy`。

## 配置

1. 复制配置模板：

```bash
cp config.yaml.example config.yaml
```

2. 编辑 `config.yaml`，填入 LLM API Key 和渠道凭据：

```yaml
llm:
  provider: minimax          # 可选: anthropic / openai / minimax / glm

  anthropic:
    api_key: "sk-ant-..."
    model: "claude-sonnet-4-20250514"

  minimax:
    api_key: "your-minimax-api-key"
    model: "MiniMax-M2.7-highspeed"
    base_url: "https://api.minimaxi.com/anthropic"

  openai:
    api_key: "sk-..."
    model: "gpt-4o"

channel:
  type: feishu_ws            # 可选: console / feishu_ws / telegram / webhook

  feishu_ws:
    app_id: "your-feishu-app-id"
    app_secret: "your-feishu-app-secret"

agent:
  max_iterations: 40          # ReAct 最大迭代轮数

workspace: ~/.scidatabot      # 记忆、会话、Cron 任务持久化目录
```

也可以在 TUI 内使用 `/connect` 命令交互式配置，无需重启即时生效。

## 运行

### TUI 交互模式（配置用）

```bash
scidatabot tui
# 或
./scidatabot.sh tui
```

TUI 内支持命令：
- `/connect` — 配置 API Key，目前只测试的Minimax M2.5/2.7/highspeed（即时热重载，无需重启）
- `/channel` — 配置 channel，目前只测试了 飞书 WebSocket（即时热重载，无需重启）
- `/help` — 显示帮助
- `exit` / Ctrl+C — 退出

配置完成后，可使用飞书访问。

```bash
scidatabot run
```

### 使用示例

```
[SciDataBot] 解析 ./data 文件夹中所有 .mat 文件，提取信号数据，
             对齐时间轴后合并为一个 HDF5 文件
```

SciDataBot 会自动：
1. 识别为复杂批量任务 → 调用 `spawn` 启动流水线
2. TaskPlanner 分析文件列表 → 生成 N 条并行 Pipeline
3. N 个 Processor 并发处理每个文件
4. Integrator 汇总结果 → 返回完整报告

### 定时任务（常驻服务）

```
[SciDataBot] 每天凌晨 2 点自动处理 /data/incoming 目录下的新文件
```

SciDataBot 会调用 `cron` 工具注册定时任务，全时在线自动执行。

## 工具列表

| 类别 | 工具 | 说明 |
|------|------|------|
| general | `read_file` `write_file` `edit_file` `list_dir` | 文件系统操作 |
| general | `exec` | Shell 命令执行 |
| general | `web_search` `web_fetch` | 网络搜索与页面抓取 |
| general | `spawn` | 派生子 Agent 并发流水线 |
| general | `cron` | 定时任务管理 |
| general | `memory` | 对话记忆查询 |
| general | `weather` | 天气查询 |
| data_access | `detect_format` | 数据文件格式检测 |
| data_access | `extract_metadata` | 元数据提取 |
| data_access | `assess_quality` | 数据质量评估 |
| data_processing | `extract_data` `transform_data` `clean_data` | 数据抽取/转换/清洗 |
| data_processing | `analyze_statistics` | 统计分析 |
| data_processing | `extract_mat_files` | MATLAB .mat 文件解析 |
| data_integration | `align_temporal` `align_spatial` | 时间/空间对齐 |
| data_integration | `export_data` | 数据导出 |

**支持的数据格式：**
CSV / TSV / JSON / JSONL · NetCDF (.nc) · HDF5 (.h5/.hdf5) · MATLAB (.mat) · Parquet / Feather · NumPy (.npy/.npz) · FITS（天文）· PNG / JPEG · Gzip / ZIP

## 扩展开发

### 添加新工具

```python
from src.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "工具说明"
    category = "data_processing"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        }

    async def execute(self, file_path: str, **kwargs) -> str:
        return "result"

# 在 src/main.py create_app() 中注册
tool_registry.register(MyTool(), "data_processing")
```

### 添加新 LLM Provider

继承 `src/providers/base.py` 的 `LLMProvider`，实现 `chat()` 方法返回 `LLMResponse`，在 `src/cli/__init__.py` 的 `create_llm_provider()` 中添加分支即可。

### 安装自定义 Skill

```bash
scidatabot skill:install ./my_skill_dir/
```

Skill 目录需包含 `SKILL.md`，描述触发条件和使用方式。已安装的 Skill 会自动注入到 Agent 上下文。

## License

MIT

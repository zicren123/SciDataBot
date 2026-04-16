"""Command-line interface."""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Annotated

import typer
import yaml
from loguru import logger

from ..config import Config, ConfigManager
from ..channels import ChannelManager, ChannelType
from ..providers import (
    OpenAIProvider, AnthropicProvider, MockProvider, MiniMaxProvider, GLMProvider,
    KimiProvider, DeepSeekProvider, QwenProvider, GrokProvider, 
    GoogleGeminiProvider, InternS1Provider, ProxyProvider
)
from ..tools import ToolRegistry
from ..core.main_agent import MainAgent

app = typer.Typer(help="scidatabot - 科学数据智能助手")

# 获取包目录下的默认配置
# __file__ is src/cli/__init__.py, so we need to go up 3 levels to get to scidatabot/
_PACKAGE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# config.yaml is in scidatabot/ directory (same level as src/)
_DEFAULT_CONFIG = _PACKAGE_DIR / "config.yaml"
# Also check parent directory (for when installed as package)
if not _DEFAULT_CONFIG.exists():
    _DEFAULT_CONFIG = _PACKAGE_DIR.parent / "scidatabot" / "config.yaml"
DEFAULT_CONFIG = str(_DEFAULT_CONFIG)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=log_format, level=log_level)

    if log_file:
        logger.add(log_file, format=log_format, level=log_level)


def load_config(config_path: str = DEFAULT_CONFIG) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def create_llm_provider(config: dict):
    """Create LLM provider from config."""
    provider_type = config.get("llm", {}).get("provider", "minimax")

    if provider_type == "minimax":
        mm_config = config.get("llm", {}).get("minimax", {})
        api_key = mm_config.get("api_key") or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            typer.echo("Warning: MiniMax API key not set, using MockProvider")
            return MockProvider(model="mock")

        # 检查是否使用 Anthropic 兼容模式
        base_url = mm_config.get("base_url", "")
        if base_url and "anthropic" in base_url:
            # 使用 Anthropic 兼容格式
            return MiniMaxProvider(
                api_key=api_key,
                model=mm_config.get("model", "MiniMax-M2.5"),
                base_url=base_url,
                temperature=mm_config.get("temperature", 0.7),
                max_tokens=mm_config.get("max_tokens", 4096),
                timeout=mm_config.get("timeout", 60),
                verify_ssl=False,  # 禁用 SSL 验证（解决证书问题）
            )
        else:
            # 使用旧版 OpenAI 兼容格式
            from src.providers.minimax import MiniMaxProvider as OldMiniMaxProvider
            return OldMiniMaxProvider(
                api_key=api_key,
                model=mm_config.get("model", "abab6.5s-chat"),
                base_url=mm_config.get("base_url", "https://api.minimax.chat/v1"),
                temperature=mm_config.get("temperature", 0.7),
                max_tokens=mm_config.get("max_tokens", 4096),
                timeout=mm_config.get("timeout", 60),
                verify_ssl=False,  # 禁用 SSL 验证
            )

    elif provider_type == "boyuerichdata":
        boyuerichdata_config = config.get("llm", {}).get("boyuerichdata", {})
        api_key = boyuerichdata_config.get("api_key") or os.environ.get("BOYUERICHDATA_API_KEY")
        if not api_key:
            typer.echo("Warning: boyuerichdata API key not set, using MockProvider")
            return MockProvider(model="mock")

        return AnthropicProvider(
            api_key=api_key,
            model=boyuerichdata_config.get("model", "claude-sonnet-4-6-thinking"),
            base_url=boyuerichdata_config.get("base_url"),
            temperature=boyuerichdata_config.get("temperature", 0.7),
            max_tokens=boyuerichdata_config.get("max_tokens", 4096),
            timeout=boyuerichdata_config.get("timeout", 300),
            max_retries=boyuerichdata_config.get("max_retries", 3),
        )

    elif provider_type == "anthropic":
        ant_config = config.get("llm", {}).get("anthropic", {})
        api_key = ant_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            typer.echo("Warning: Anthropic API key not set, using MockProvider")
            return MockProvider(model="mock")

        return AnthropicProvider(
            api_key=api_key,
            model=ant_config.get("model", "claude-sonnet-4-20250514"),
            base_url=ant_config.get("base_url"),
            temperature=ant_config.get("temperature", 0.7),
            max_tokens=ant_config.get("max_tokens", 4096),
            timeout=ant_config.get("timeout", 60),
            max_retries=ant_config.get("max_retries", 3),
        )

    elif provider_type == "glm":
        glm_config = config.get("llm", {}).get("glm", {})
        api_key = glm_config.get("api_key") or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
        if not api_key:
            typer.echo("Warning: GLM API key not set, using MockProvider")
            return MockProvider(model="mock")

        return GLMProvider(
            api_key=api_key,
            model=glm_config.get("model", "glm-4-flash"),
            base_url=glm_config.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
            temperature=glm_config.get("temperature", 0.7),
            max_tokens=glm_config.get("max_tokens", 4096),
            timeout=glm_config.get("timeout", 60),
        )

    elif provider_type == "proxy":
        proxy_config = config.get("llm", {}).get("proxy", {})
        api_key = proxy_config.get("api_key") or os.environ.get("PROXY_API_KEY")
        base_url = proxy_config.get("base_url")
        if not api_key or not base_url:
            typer.echo("Warning: Proxy API key or base_url not set, using MockProvider")
            return MockProvider(model="mock")

        return ProxyProvider(
            api_key=api_key,
            model=proxy_config.get("model", "gpt-4o"),
            base_url=base_url,
            temperature=proxy_config.get("temperature", 0.7),
            max_tokens=proxy_config.get("max_tokens", 4096),
            timeout=proxy_config.get("timeout", 60),
        )

    elif provider_type == "kimi":
        kimi_config = config.get("llm", {}).get("kimi", {})
        api_key = kimi_config.get("api_key") or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            typer.echo("Warning: Kimi API key not set (KIMI_API_KEY or MOONSHOT_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return KimiProvider(
            api_key=api_key,
            model=kimi_config.get("model", "moonshot-v1-128k"),
            base_url=kimi_config.get("base_url", "https://api.moonshot.cn/v1"),
            temperature=kimi_config.get("temperature", 0.7),
            max_tokens=kimi_config.get("max_tokens", 4096),
            timeout=kimi_config.get("timeout", 60),
        )

    elif provider_type == "deepseek":
        deepseek_config = config.get("llm", {}).get("deepseek", {})
        api_key = deepseek_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            typer.echo("Warning: DeepSeek API key not set (DEEPSEEK_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return DeepSeekProvider(
            api_key=api_key,
            model=deepseek_config.get("model", "deepseek-chat"),
            base_url=deepseek_config.get("base_url", "https://api.deepseek.com/v1"),
            temperature=deepseek_config.get("temperature", 0.7),
            max_tokens=deepseek_config.get("max_tokens", 4096),
            timeout=deepseek_config.get("timeout", 60),
        )

    elif provider_type == "qwen":
        qwen_config = config.get("llm", {}).get("qwen", {})
        api_key = qwen_config.get("api_key") or os.environ.get("QWEN_API_KEY")
        if not api_key:
            typer.echo("Warning: Qwen API key not set (QWEN_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return QwenProvider(
            api_key=api_key,
            model=qwen_config.get("model", "qwen-plus"),
            base_url=qwen_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=qwen_config.get("temperature", 0.7),
            max_tokens=qwen_config.get("max_tokens", 4096),
            timeout=qwen_config.get("timeout", 60),
        )

    elif provider_type == "grok":
        grok_config = config.get("llm", {}).get("grok", {})
        api_key = grok_config.get("api_key") or os.environ.get("XAI_API_KEY")
        if not api_key:
            typer.echo("Warning: Grok API key not set (XAI_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return GrokProvider(
            api_key=api_key,
            model=grok_config.get("model", "grok-3"),
            base_url=grok_config.get("base_url", "https://api.x.ai/v1"),
            temperature=grok_config.get("temperature", 0.7),
            max_tokens=grok_config.get("max_tokens", 4096),
            timeout=grok_config.get("timeout", 60),
        )

    elif provider_type == "google":
        google_config = config.get("llm", {}).get("google", {})
        api_key = google_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            typer.echo("Warning: Google API key not set (GOOGLE_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return GoogleGeminiProvider(
            api_key=api_key,
            model=google_config.get("model", "gemini-2.0-flash"),
            base_url=google_config.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
            temperature=google_config.get("temperature", 0.7),
            max_tokens=google_config.get("max_tokens", 4096),
            timeout=google_config.get("timeout", 60),
        )

    elif provider_type == "intern_s1":
        intern_config = config.get("llm", {}).get("intern_s1", {})
        api_key = intern_config.get("api_key") or os.environ.get("INTERN_S1_API_KEY")
        if not api_key:
            typer.echo("Warning: Intern-S1 API key not set (INTERN_S1_API_KEY), using MockProvider")
            return MockProvider(model="mock")

        return InternS1Provider(
            api_key=api_key,
            model=intern_config.get("model", "intern-s1"),
            base_url=intern_config.get("base_url", "https://chat.intern-ai.org.cn/api/v1"),
            temperature=intern_config.get("temperature", 0.7),
            max_tokens=intern_config.get("max_tokens", 4096),
            timeout=intern_config.get("timeout", 60),
        )

    elif provider_type == "proxy":
        proxy_config = config.get("llm", {}).get("proxy", {})
        api_key = proxy_config.get("api_key") or os.environ.get("PROXY_API_KEY")
        base_url = proxy_config.get("base_url")
        if not api_key or not base_url:
            typer.echo("Warning: Proxy API key or base_url not set, using MockProvider")
            return MockProvider(model="mock")

        return ProxyProvider(
            api_key=api_key,
            model=proxy_config.get("model", "gpt-4o"),
            base_url=base_url,
            temperature=proxy_config.get("temperature", 0.7),
            max_tokens=proxy_config.get("max_tokens", 4096),
            timeout=proxy_config.get("timeout", 60),
        )

    else:
        typer.echo(f"Unknown provider: {provider_type}, using MockProvider")
        return MockProvider(model="mock")


@app.command()
def run(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Config file path")] = None,
    log_level: Annotated[str, typer.Option("--log-level", "-l", help="Log level")] = "INFO",
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="LLM provider (override config)")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model name (override config)")] = None,
    channel: Annotated[Optional[str], typer.Option("--channel", help="Channel to use")] = None,
    confirm_dangerous: Annotated[bool, typer.Option("--confirm-dangerous", help="Automatically confirm dangerous tools")] = False,
):
    """Start the scidatabot."""

    # Load config file
    config_path = config or DEFAULT_CONFIG
    config_data = load_config(config_path)

    # Setup logging
    log_config = config_data.get("logging", {})
    setup_logging(log_level or log_config.get("level", "INFO"), log_config.get("file"))

    # Create LLM provider
    if provider:
        # Override provider from CLI - need to create minimal config
        config_data = {"llm": {"provider": provider}}
        if provider == "minimax":
            config_data["llm"]["minimax"] = {
                "api_key": os.environ.get("MINIMAX_API_KEY") or config_data.get("llm", {}).get("minimax", {}).get("api_key")
            }

    llm = create_llm_provider(config_data)

    # Get provider info
    provider_name = provider or config_data.get("llm", {}).get("provider", "minimax")
    model_name = model or config_data.get("llm", {}).get(provider_name, {}).get("model", "abab6.5s-chat")

    # Create channel manager
    channel_type = channel or config_data.get("channel", {}).get("type", "console")
    channel_manager = ChannelManager()

    if channel_type == "console":
        from ..channels import ConsoleChannel
        channel_manager.add_channel("main", ChannelType.CONSOLE, {})

    elif channel_type == "feishu":
        from ..channels import FeishuChannel
        feishu_config = config_data.get("channel", {}).get("feishu", {})
        channel_manager.add_channel("main", ChannelType.FEISHU, feishu_config)

    elif channel_type == "feishu_ws":
        from ..channels import FeishuWSChannel
        feishu_ws_config = config_data.get("channel", {}).get("feishu_ws", {})
        channel_manager.add_channel("main", ChannelType.FEISHU_WS, feishu_ws_config)

    elif channel_type == "telegram":
        from ..channels import TelegramChannel
        telegram_config = config_data.get("channel", {}).get("telegram", {})
        channel_manager.add_channel("main", ChannelType.TELEGRAM, telegram_config)

    elif channel_type == "webhook":
        from ..channels import WebhookChannel
        webhook_config = config_data.get("channel", {}).get("webhook", {})
        channel_manager.add_channel("main", ChannelType.WEBHOOK, webhook_config)

    else:
        typer.echo(f"Unknown channel: {channel_type}", err=True)
        raise typer.Exit(1)

    # Create workspace and tool registry
    workspace_path = config_data.get("workspace", "./workspace")
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    # Create MainAgent (new architecture)
    from ..tools.data_access import FormatDetector, MetadataExtractor, QualityAssessor
    from ..tools.general import WeatherTool
    from ..tools.data_processing import DataExtractor, DataTransformer, DataCleaner, StatisticsAnalyzer
    from ..tools.data_integration import TemporalAligner, SpatialAligner, DataExporter

    # Create tool registry with all tools
    tool_registry = ToolRegistry()

    # Register data access tools
    tool_registry.register(FormatDetector(), "data_access")
    tool_registry.register(MetadataExtractor(), "data_access")
    tool_registry.register(QualityAssessor(), "data_access")

    # Register general tools (web search, etc.)
    from src.tools.general import WebSearchTool, WebFetchTool
    tool_registry.register(WeatherTool(), "general")
    web_config = config_data.get("tools", {}).get("web", {})
    tool_registry.register(
        WebSearchTool(
            api_key=web_config.get("brave_api_key"),
            provider=web_config.get("provider"),
            proxy=web_config.get("proxy"),
            kimi_api_key=web_config.get("kimi_api_key"),
            kimi_base_url=web_config.get("kimi_base_url"),
            kimi_search_path=web_config.get("kimi_search_path"),
            timeout=web_config.get("timeout", 10),
        ),
        "general"
    )
    tool_registry.register(WebFetchTool(), "general")
    
    # Register filesystem tools
    from src.tools.general.filesystem import ReadFileTool, WriteFileTool, ListDirTool, EditFileTool
    from src.tools.general.shell import ExecTool
    from src.tools.general.spawn import SpawnTool
    from src.tools.general.cron import CronTool
    from src.tools.data_processing.mat_extractor import MatFileExtractor
    tool_registry.register(ReadFileTool(workspace=workspace, allowed_dir=workspace), "general")
    tool_registry.register(WriteFileTool(workspace=workspace, allowed_dir=workspace), "general")
    tool_registry.register(EditFileTool(workspace=workspace, allowed_dir=workspace), "general")
    tool_registry.register(ListDirTool(workspace=workspace, allowed_dir=workspace), "general")
    tool_registry.register(ExecTool(), "general")
    tool_registry.register(SpawnTool(), "general")
    tool_registry.register(CronTool(), "general")
    tool_registry.register(MatFileExtractor(), "data_processing")

    # Note: Intent recognition moved to LLM (in MainAgent system prompt)

    # Register data processing tools
    tool_registry.register(DataExtractor(), "data_processing")
    tool_registry.register(DataTransformer(), "data_processing")
    tool_registry.register(DataCleaner(), "data_processing")
    tool_registry.register(StatisticsAnalyzer(), "data_processing")

    # Register data integration tools
    tool_registry.register(TemporalAligner(), "data_integration")
    tool_registry.register(SpatialAligner(), "data_integration")
    tool_registry.register(DataExporter(), "data_integration")

    # Connect MCP servers
    mcp_config = config_data.get("tools", {}).get("mcp_servers", {})
    if mcp_config:
        from src.tools.general import MCPConfig, connect_mcp_servers
        from contextlib import AsyncExitStack
        
        async def connect_mcp():
            stack = AsyncExitStack()
            mcp_servers = {
                name: MCPConfig(
                    command=cfg.get("command"),
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                    url=cfg.get("url"),
                    headers=cfg.get("headers"),
                    tool_timeout=cfg.get("tool_timeout", 30),
                )
                for name, cfg in mcp_config.items()
            }
            await connect_mcp_servers(mcp_servers, tool_registry, stack)
        
        try:
            asyncio.run(connect_mcp())
        except Exception as e:
            logger.warning(f"Failed to connect MCP servers: {e}")

    # Create confirmation callback for dangerous tools
    async def confirm_dangerous_tool(tool_name: str, arguments: dict) -> bool:
        if confirm_dangerous:
            return True
        typer.echo(f"\n⚠️  危险工具请求: {tool_name}")
        typer.echo(f"   参数: {arguments}")
        response = typer.prompt("   确认执行? (y/n)", default="n")
        return response.lower() in ("y", "yes")

    # Create MainAgent (new architecture)
    model_name = model or config_data.get("llm", {}).get(provider_name, {}).get("model", "abab6.5s-chat")
    agent = MainAgent(
        provider=llm,
        workspace=workspace,
        model=model_name,
        max_iterations=40,
        tool_registry=tool_registry,
    )

    # Message handler - publish to agent bus
    async def handle_message(message):
        await agent.bus.publish_inbound(message)

    channel_manager.set_global_handler(handle_message)

    # Start
    typer.echo(f"Starting scidatabot with {provider_name}/{model_name}...")

    async def run_with_agent():
        # Start channel
        typer.echo(f"Starting channel '{channel_type}'...")
        await channel_manager.start_channel("main")
        typer.echo("Channel started")

        # Task: consume outbound messages and send back through channel
        async def output_sender():
            while True:
                try:
                    msg = await agent.bus.consume_outbound()
                    if msg:
                        typer.echo(f"[Sending response to {msg.chat_id}]")
                        await channel_manager.send_message("main", msg)
                except Exception as e:
                    typer.echo(f"Error sending message: {e}")
                    await asyncio.sleep(1)

        # Run agent and output sender concurrently
        agent_task = asyncio.create_task(agent.run())
        sender_task = asyncio.create_task(output_sender())
        try:
            await asyncio.Event().wait()  # Wait forever
        except KeyboardInterrupt:
            pass
        finally:
            agent.stop()
            sender_task.cancel()
            await channel_manager.stop_all()

    try:
        asyncio.run(run_with_agent())
    except KeyboardInterrupt:
        typer.echo("\nShutting down...")


@app.command()
def tui(
    config: Annotated[Optional[str], typer.Option("--config", "-c", help="Config file path")] = None,
):
    """Start the TUI (Text User Interface)."""
    import asyncio
    from pathlib import Path

    # Load config
    config_path = config or DEFAULT_CONFIG
    config_data = load_config(config_path)

    # Create scheduler using create_app
    from ..main import create_app
    scheduler = create_app(config_data)

    # Import and run simple TUI
    from ..tui.simple_tui import run_simple_tui
    asyncio.run(run_simple_tui(scheduler))


@app.command("skill:list")
def skill_list():
    """List all installed skills."""
    from ..skills.manager import get_skill_loader

    loader = get_skill_loader()
    skills = loader.list()

    if not skills:
        typer.echo("No skills installed.")
        return

    typer.echo(f"Installed skills ({len(skills)}):\n")
    for skill in skills:
        emoji = skill.metadata.emoji or "📦"
        typer.echo(f"  {emoji} {skill.name}")
        if skill.description:
            typer.echo(f"      {skill.description}")
        typer.echo()


@app.command("skill:install")
def skill_install(
    path: str = typer.Argument(..., help="Path to skill directory containing SKILL.md"),
):
    """Install a skill from a local directory."""
    from pathlib import Path
    from ..skills.manager import get_skill_loader

    skill_path = Path(path).expanduser().resolve()

    if not skill_path.exists():
        typer.echo(f"Error: Path does not exist: {skill_path}", err=True)
        raise typer.Exit(1)

    if not skill_path.is_dir():
        typer.echo(f"Error: Path is not a directory: {skill_path}", err=True)
        raise typer.Exit(1)

    skill_file = skill_path / "SKILL.md"
    if not skill_file.exists():
        typer.echo(f"Error: SKILL.md not found in {skill_path}", err=True)
        raise typer.Exit(1)

    loader = get_skill_loader()
    try:
        skill = loader.install(skill_path)
        typer.echo(f"✓ Installed skill: {skill.name}")
    except Exception as e:
        typer.echo(f"Error installing skill: {e}", err=True)
        raise typer.Exit(1)


@app.command("skill:uninstall")
def skill_uninstall(
    name: str = typer.Argument(..., help="Skill name to uninstall"),
):
    """Uninstall a skill."""
    from ..skills.manager import get_skill_loader

    loader = get_skill_loader()

    if loader.uninstall(name):
        typer.echo(f"✓ Uninstalled skill: {name}")
    else:
        typer.echo(f"Error: Could not uninstall skill: {name}", err=True)
        raise typer.Exit(1)


@app.command("skill:info")
def skill_info(
    name: str = typer.Argument(..., help="Skill name"),
):
    """Show detailed info about a skill."""
    from ..skills.manager import get_skill_loader

    loader = get_skill_loader()
    skill = loader.get(name)

    if not skill:
        typer.echo(f"Error: Skill not found: {name}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Skill: {skill.name}")
    typer.echo(f"Description: {skill.description}")
    typer.echo(f"Path: {skill.path}")
    typer.echo(f"Enabled: {skill.enabled}")
    if skill.metadata.emoji:
        typer.echo(f"Emoji: {skill.metadata.emoji}")
    if skill.metadata.homepage:
        typer.echo(f"Homepage: {skill.metadata.homepage}")
    if skill.metadata.requires:
        typer.echo(f"Requires: {skill.metadata.requires}")

    typer.echo("\n--- Content Preview ---")
    # Show first 50 lines
    lines = skill.content.split("\n")[:50]
    typer.echo("\n".join(lines))
    if len(skill.content.split("\n")) > 50:
        typer.echo("\n... (truncated)")


PROVIDER_MODELS = {
    "anthropic": {
        "default": "claude-sonnet-4-20250514",
        "options": ["claude-sonnet-4-20250514", "claude-opus-4-5-20250514", "claude-3-5-sonnet-20241022"],
    },
    "minimax": {
        "default": "MiniMax-M2.5",
        "options": ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"],
    },
    "glm": {
        "default": "glm-4-flash",
        "options": ["glm-4-flash", "glm-4-plus", "glm-4v", "glm-3-turbo"],
    },
    "kimi": {
        "default": "moonshot-v1-128k",
        "options": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
    "deepseek": {
        "default": "deepseek-chat",
        "options": ["deepseek-chat"],
    },
    "qwen": {
        "default": "qwen-plus",
        "options": ["qwen-plus", "qwen-turbo"],
    },
    "grok": {
        "default": "grok-3",
        "options": ["grok-3"],
    },
    "google": {
        "default": "gemini-2.0-flash",
        "options": ["gemini-2.0-flash", "gemini-1.5-flash"],
    },
    "openai": {
        "default": "gpt-4o",
        "options": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "boyuerichdata": {
        "default": "claude-sonnet-4-6-thinking",
        "options": ["claude-sonnet-4-6-thinking"],
    },
    "custom": {
        "default": "custom",
        "options": [],
    },
}


@app.command("connect")
def connect(
    config_path: Annotated[Optional[str], typer.Option("--config", "-c", help="Config file path")] = None,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="Provider: anthropic, minimax, glm")] = None,
    api_key: Annotated[Optional[str], typer.Option("--api-key", "-k", help="API key")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model name")] = None,
    temperature: Annotated[Optional[float], typer.Option("--temperature", "-t", help="Temperature (0.0-1.0)")] = None,
    max_tokens: Annotated[Optional[int], typer.Option("--max-tokens", help="Max tokens")] = None,
):
    """Configure API settings interactively or via arguments."""
    from pathlib import Path
    import yaml

    typer.echo("\n" + "=" * 50)
    typer.echo("  SciDataBot API Configuration")
    typer.echo("=" * 50 + "\n")

    # Load existing config
    cfg_path = config_path or DEFAULT_CONFIG
    config_data = {}
    if Path(cfg_path).exists():
        with open(cfg_path) as f:
            config_data = yaml.safe_load(f) or {}
    
    if "llm" not in config_data:
        config_data["llm"] = {}
    
    # Show current config
    current_provider = config_data.get("llm", {}).get("provider", "minimax")
    typer.echo(f"Current provider: {current_provider}\n")
    
    # 如果提供了所有必要参数，使用命令行模式
    if provider and api_key and model:
        # 命令行模式
        selected_provider = provider
        selected_api_key = api_key
        selected_model = model
        selected_temp = temperature if temperature is not None else 0.7
        selected_tokens = max_tokens if max_tokens else 4096
    else:
        # 交互式模式
        # Show provider options
        typer.echo("Available providers:")
        typer.echo("  1. anthropic       - Anthropic Claude API")
        typer.echo("  2. minimax         - MiniMax API")
        typer.echo("  3. glm             - Zhipu AI (GLM) API")
        typer.echo("  4. kimi            - Moonshot Kimi API")
        typer.echo("  5. deepseek        - DeepSeek API")
        typer.echo("  6. qwen            - Alibaba Qwen API")
        typer.echo("  7. grok            - xAI Grok API")
        typer.echo("  8. google          - Google Gemini API")
        typer.echo("  9. openai          - OpenAI API")
        typer.echo("  10. boyuerichdata  - BoyueRichData API")
        typer.echo("")
        
        # Get provider choice
        provider_choice = typer.prompt(
            "Select provider (1-10)",
            default="1",
            show_default=False,
        )
        
        provider_map = {
            "1": "anthropic", "2": "minimax", "3": "glm", "4": "kimi", 
            "5": "deepseek", "6": "qwen", "7": "grok", "8": "google",
            "9": "openai", "10": "boyuerichdata"
        }
        selected_provider = provider_map.get(provider_choice, "anthropic")
        
        typer.echo(f"\nSelected: {selected_provider}")
        
        # Get API key
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "glm": "ZHIPU_API_KEY",
            "kimi": "KIMI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "QWEN_API_KEY",
            "grok": "XAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "boyuerichdata": "BOYUERICHDATA_API_KEY",
        }
        env_var = env_var_map.get(selected_provider, "API_KEY")
        config_key = config_data.get("llm", {}).get(selected_provider, {}).get("api_key", "")
        current_key = os.environ.get(env_var, "") or config_key
        
        # Mask the API key for display
        display_key = current_key[:10] + "..." if current_key else ""
        
        selected_api_key = typer.prompt(
            f"API Key (env: {env_var})",
            default=current_key,
            show_default=bool(current_key),
            hide_input=True,
        )
        
        if not selected_api_key:
            selected_api_key = current_key
        
        # Get model
        model_info = PROVIDER_MODELS.get(selected_provider, {"default": "gpt-4o"})
        default_model = model_info["default"]
        options = model_info.get("options", [])
        
        if options:
            typer.echo(f"\nAvailable models: {', '.join(options)}")
            selected_model = typer.prompt("Model", default=default_model)
        else:
            selected_model = default_model
        
        # Get temperature
        temp_input = typer.prompt("Temperature (0.0-1.0)", default="0.7", show_default=False)
        try:
            selected_temp = float(temp_input)
        except ValueError:
            selected_temp = 0.7
        
        # Get max tokens
        tokens_input = typer.prompt("Max tokens", default="4096", show_default=False)
        try:
            selected_tokens = int(tokens_input)
        except ValueError:
            selected_tokens = 4096
    
    # Update config
    config_data["llm"]["provider"] = selected_provider
    
    config_data["llm"][selected_provider] = {
        "api_key": selected_api_key,
        "model": selected_model,
        "temperature": selected_temp,
        "max_tokens": selected_tokens,
    }
    
    # Set base_url for each provider
    base_url_map = {
        "minimax": "https://api.minimaxi.com/anthropic",
        "anthropic": "https://api.anthropic.com",
        "glm": "https://open.bigmodel.cn/api/paas/v4",
        "kimi": "https://api.moonshot.cn/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "grok": "https://api.x.ai/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "openai": "https://api.openai.com/v1",
        "boyuerichdata": "http://35.220.164.252:3888",
    }
    
    if selected_provider in base_url_map:
        config_data["llm"][selected_provider]["base_url"] = base_url_map[selected_provider]
    
    # Save config
    with open(cfg_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    typer.echo(f"\n✓ Configuration saved to {cfg_path}")
    typer.echo(f"\nProvider: {selected_provider}")
    typer.echo(f"Model: {selected_model}")
    typer.echo(f"Temperature: {selected_temp}")
    typer.echo(f"Max tokens: {selected_tokens}")
    typer.echo("\nRestart scidatabot to use the new configuration.")
    typer.echo("Run: scidatabot run --config " + cfg_path)


@app.command("channel")
def channel_cmd(
    config_path: Annotated[Optional[str], typer.Option("--config", "-c", help="Config file path")] = None,
    channel_type: Annotated[Optional[str], typer.Option("--type", "-t", help="Channel type: console, feishu, feishu_ws, telegram, webhook")] = None,
    app_id: Annotated[Optional[str], typer.Option("--app-id", help="Feishu App ID")] = None,
    app_secret: Annotated[Optional[str], typer.Option("--app-secret", help="Feishu App Secret")] = None,
    token: Annotated[Optional[str], typer.Option("--token", help="Telegram Bot Token")] = None,
):
    """Configure channel settings interactively or via arguments."""
    from pathlib import Path

    typer.echo("\n" + "=" * 50)
    typer.echo("  SciDataBot Channel Configuration")
    typer.echo(" " * 50 + "\n")

    # Load existing config
    cfg_path = config_path or DEFAULT_CONFIG
    config_data = {}
    if Path(cfg_path).exists():
        with open(cfg_path) as f:
            config_data = yaml.safe_load(f) or {}

    if "channel" not in config_data:
        config_data["channel"] = {}

    # Show current config
    current_type = config_data.get("channel", {}).get("type", "console")
    typer.echo(f"Current channel type: {current_type}\n")

    # If all arguments provided, use command mode
    if channel_type and app_id and app_secret:
        selected_type = channel_type
        config_data["channel"]["type"] = selected_type
        config_data["channel"][selected_type] = {
            "app_id": app_id,
            "app_secret": app_secret,
        }
    elif channel_type and token:
        selected_type = channel_type
        config_data["channel"]["type"] = selected_type
        config_data["channel"]["telegram"] = {"token": token}
    else:
        # Interactive mode
        typer.echo("Available channel types:")
        typer.echo("  1. console     - Console (local terminal)")
        typer.echo("  2. feishu      - Feishu HTTP API (send only)")
        typer.echo("  3. feishu_ws   - Feishu WebSocket (send & receive)")
        typer.echo("  4. telegram    - Telegram Bot")
        typer.echo("  5. webhook     - Webhook (HTTP callback)")
        typer.echo("")

        choice = typer.prompt("Select channel type (1-5)", default="1", show_default=False)

        channel_map = {
            "1": "console",
            "2": "feishu",
            "3": "feishu_ws",
            "4": "telegram",
            "5": "webhook",
        }
        selected_type = channel_map.get(choice, "console")

        if selected_type == "console":
            config_data["channel"]["type"] = "console"
            for key in ["feishu", "feishu_ws", "telegram"]:
                if key in config_data["channel"]:
                    del config_data["channel"][key]

        elif selected_type in ("feishu", "feishu_ws"):
            config_key = selected_type
            current_config = config_data["channel"].get(config_key, {})
            current_app_id = current_config.get("app_id", "")
            current_secret = current_config.get("app_secret", "")

            new_app_id = typer.prompt("Feishu App ID", default=current_app_id, show_default=bool(current_app_id))
            new_secret = typer.prompt("Feishu App Secret", default=current_secret, hide_input=bool(current_secret), show_default=bool(current_secret))

            if new_app_id and new_secret:
                config_data["channel"]["type"] = config_key
                config_data["channel"][config_key] = {
                    "app_id": new_app_id,
                    "app_secret": new_secret,
                }
            else:
                typer.echo("\nError: app_id and app_secret are required")
                return

        elif selected_type == "telegram":
            current_config = config_data["channel"].get("telegram", {})
            current_token = current_config.get("token", "")

            new_token = typer.prompt("Telegram Bot Token", default=current_token, show_default=bool(current_token))

            if new_token:
                config_data["channel"]["type"] = "telegram"
                config_data["channel"]["telegram"] = {"token": new_token}
            else:
                typer.echo("\nError: token is required")
                return

        elif selected_type == "webhook":
            current_config = config_data["channel"].get("webhook", {})
            current_host = current_config.get("host", "0.0.0.0")
            current_port = current_config.get("port", 8080)
            current_path = current_config.get("path", "/webhook")

            new_host = typer.prompt("Host", default=current_host, show_default=True)
            new_port_str = typer.prompt("Port", default=str(current_port), show_default=True)
            try:
                new_port = int(new_port_str)
            except ValueError:
                new_port = current_port
            new_path = typer.prompt("Path", default=current_path, show_default=True)

            config_data["channel"]["type"] = "webhook"
            config_data["channel"]["webhook"] = {
                "host": new_host,
                "port": new_port,
                "path": new_path,
            }

    # Save config
    with open(cfg_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    typer.echo(f"\n✓ Configuration saved to {cfg_path}")
    typer.echo(f"Channel type: {config_data['channel']['type']}")
    typer.echo("\nRestart scidatabot to use the new configuration.")


@app.command()
def version():
    """Show version information."""
    typer.echo("scidatabot v0.1.0")


if __name__ == "__main__":
    app()


def main():
    """Entry point for CLI."""
    app()

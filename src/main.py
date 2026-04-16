"""SciDataBot Main Entry Point."""

import asyncio
import sys
from pathlib import Path

from loguru import logger

# 配置日志同时输出到 stdout 和文件
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("scidatabot.log", rotation="10 MB", retention="7 days", level="DEBUG")

from src.config import load_config, get_cron_dir
from src.core.main_agent import MainAgent
from src.tools.registry import ToolRegistry
from src.cron import CronService


def create_app(config: dict = None):
    """Create the SciDataBot application."""
    config = config or {}
    
    workspace = Path(config.get("workspace", "~/.scidatabot")).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create provider
    from src.cli import create_llm_provider
    provider = create_llm_provider(config)
    logger.info(f"Using LLM Provider: {provider.name}")
    
    # Create CronService
    cron_dir = get_cron_dir()
    cron_store_path = cron_dir / "jobs.json"
    
    # Create a placeholder for message bus (will be set after agent is created)
    message_bus_holder = {"bus": None}
    
    async def on_cron_job(job):
        """Callback when a cron job executes."""
        bus = message_bus_holder.get("bus")
        if not bus:
            logger.warning("Cron: no message bus available")
            return None
        
        from src.bus.events import InboundMessage
        
        msg = InboundMessage(
            channel=job.payload.channel or "cli",
            sender_id="cron",
            chat_id=job.payload.to or "direct",
            content=job.payload.message,
        )
        await bus.publish_inbound(msg)
        logger.info(f"Cron: published job '{job.name}' to bus")
        return None
    
    cron_service = CronService(store_path=cron_store_path, on_job=on_cron_job)
    logger.info("CronService initialized")
    
    # Create tool registry
    tool_registry = ToolRegistry()
    
    # Register tools
    from src.tools.data_access import FormatDetector, MetadataExtractor, QualityAssessor
    from src.tools.data_processing import DataExtractor, DataTransformer, DataCleaner, StatisticsAnalyzer, MatFileExtractor
    from src.tools.data_integration import TemporalAligner, SpatialAligner, DataExporter
    from src.tools.general import WeatherTool, WebSearchTool, WebFetchTool, ExecTool, ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, SpawnTool, CronTool, KimiWebSearchTool
    
    tool_registry.register(FormatDetector(), "data_access")
    tool_registry.register(MetadataExtractor(), "data_access")
    tool_registry.register(QualityAssessor(), "data_access")
    
    web_config = config.get("tools", {}).get("web", {})
    web_provider = (web_config.get("provider")).lower()
    if web_provider == "kimi":
        tool_registry.register(
            KimiWebSearchTool(
                max_results=web_config.get("max_results", 5),
                proxy=web_config.get("proxy"),
                kimi_api_key=web_config.get("kimi_api_key"),
                kimi_base_url=web_config.get("kimi_base_url"),
                kimi_search_path=web_config.get("kimi_search_path"),
                timeout=web_config.get("timeout", 10),
            ),
            "general"
        )
    else:
        tool_registry.register(
            WebSearchTool(
                api_key=web_config.get("brave_api_key"),
                max_results=web_config.get("max_results", 5),
                proxy=web_config.get("proxy"),
            ),
            "general"
        )
    tool_registry.register(WebFetchTool(), "general")
    tool_registry.register(ExecTool(), "general")
    tool_registry.register(ReadFileTool(), "general")
    tool_registry.register(WriteFileTool(), "general")
    tool_registry.register(EditFileTool(), "general")
    tool_registry.register(ListDirTool(), "general")
    tool_registry.register(WeatherTool(), "general")
    spawn_tool = SpawnTool()
    tool_registry.register(spawn_tool, "general")
    tool_registry.register(CronTool(cron_service=cron_service), "general")
    cron_tool = tool_registry._tools.get("cron")
    if cron_tool:
        cron_tool.set_context(channel="cli", chat_id="direct")
    
    tool_registry.register(DataExtractor(), "data_processing")
    tool_registry.register(DataTransformer(), "data_processing")
    tool_registry.register(DataCleaner(), "data_processing")
    tool_registry.register(StatisticsAnalyzer(), "data_processing")
    tool_registry.register(MatFileExtractor(), "data_processing")
    
    tool_registry.register(TemporalAligner(), "data_integration")
    tool_registry.register(SpatialAligner(), "data_integration")
    tool_registry.register(DataExporter(), "data_integration")
    
    logger.info(f"Registered {len(tool_registry)} tools")
    
    # Create MainAgent
    llm_config = config.get("llm", {})
    provider_type = llm_config.get("provider", "minimax")
    provider_model_config = llm_config.get(provider_type, {})
    model = provider_model_config.get("model", "anthropic/claude-opus-4-5")
    max_iterations = config.get("max_iterations", 40)
    
    agent = MainAgent(
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=max_iterations,
        tool_registry=tool_registry,
    )
    
    # Set message bus for cron service
    message_bus_holder["bus"] = agent.bus

    # Attach cron_service to agent for easy access
    agent.cron_service = cron_service
    
    # Inject message bus into web search tools for TUI notifications
    web_search_tool = tool_registry._tools.get("web_search")
    if web_search_tool:
        web_search_tool.bus = agent.bus
    kimi_search_tool = tool_registry._tools.get("kimi_web_search")
    if kimi_search_tool:
        kimi_search_tool.bus = agent.bus

    # Wire SpawnTool callback → SubagentManager.spawn (task_planner entry point)
    registered_spawn = tool_registry._tools.get("spawn")
    if registered_spawn:
        async def _spawn_callback(
            task: str,
            label: str = None,
            origin_channel: str = "cli",
            origin_chat_id: str = "direct",
            session_key: str = None,
        ) -> str:
            return await agent.subagent_manager.spawn(
                task=task,
                label=label,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                session_key=session_key,
                subagent_type="task_planner",
            )
        registered_spawn._subagent_callback = _spawn_callback
        logger.info("SpawnTool callback wired to SubagentManager")

    return agent


async def main():
    """Main entry point."""
    logger.info("=" * 50)
    logger.info("SciDataBot Starting")
    logger.info("=" * 50)
    
    # Load config
    from src.cli import load_config
    config_dir = Path(__file__).parent.parent
    config_path = config_dir / "config.yaml"
    config = load_config(str(config_path))
    
    # Create app
    agent = create_app(config)
    cron_service = agent.cron_service
    
    # Print available tools
    print("\nAvailable tool categories:")
    for cat in agent.tool_registry.list_categories():
        tools = agent.tool_registry.list_tools(cat)
        print(f"  {cat}: {', '.join(tools)}")
    
    # Get query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("请输入查询:")
        query = input()
    
    print(f"\nUser request: {query}")
    print("-" * 50)
    print("模式: 服务模式 (输入 Ctrl+C 退出)")
    print("-" * 50)
    
    # Run service mode
    from src.bus.events import InboundMessage, OutboundMessage
    
    async def run_service_mode():
        # Start CronService
        await cron_service.start()
        
        # Start agent processing in background
        agent_task = asyncio.create_task(agent.run())
        
        # Wait a bit for agent to start
        await asyncio.sleep(0.5)
        
        # Send initial query to bus
        initial_msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=query,
        )
        await agent.bus.publish_inbound(initial_msg)
        
        # Wait for results
        last_result_time = None
        consecutive_no_result = 0
        
        try:
            while agent._running:
                try:
                    msg = await asyncio.wait_for(agent.bus.consume_outbound(), timeout=2.0)
                    if msg and isinstance(msg, OutboundMessage):
                        print("\n" + "=" * 50)
                        print("Result:")
                        print("=" * 50)
                        print(msg.content)
                        print()
                        last_result_time = asyncio.get_event_loop().time()
                        consecutive_no_result = 0
                except asyncio.TimeoutError:
                    if last_result_time:
                        consecutive_no_result += 1
                        # If no new results for 10 times (20s), exit
                        if consecutive_no_result >= 10:
                            break
                    if not agent._running:
                        break
        finally:
            agent.stop()
            cron_service.stop()
    
    await run_service_mode()


if __name__ == "__main__":
    asyncio.run(main())

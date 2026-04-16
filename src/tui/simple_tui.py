#!/usr/bin/env python
"""简单的命令行 TUI - 支持模式切换和命令"""

import asyncio
import os
import sys
from pathlib import Path


def print_welcome(mode="auto"):
    """打印欢迎界面"""
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print("\033[1;36m" + "          SciDataBot TUI v2.0" + "\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print()
    print(f"\033[1;32m欢迎使用 SciDataBot 文本界面！\033[0m")
    print(f"\n自动识别任务类型和执行策略")
    print()
    print("\033[1;33m功能:\033[0m")
    print("  - 自动识别任务类型")
    print("  - 简单任务直接执行")
    print("  - 复杂任务并行处理")
    print()
    print("\033[1;33m快捷键/命令:\033[0m")
    print("  /channel - 配置通道 (飞书/Telegram等)")
    print("  /connect - 配置 API 设置")
    print("  /help   - 显示帮助")
    print("  Ctrl+C  - 退出程序")
    print()
    print("\033[1;36m" + "-" * 60 + "\033[0m")
    print()


def get_prefix():
    """获取命令前缀 - 新架构不需要前缀"""
    return ""


async def run_simple_tui(scheduler, config_path=None):
    """运行 TUI (服务模式 - 支持 subagent)"""
    from src.bus.events import InboundMessage, OutboundMessage
    
    print_welcome("auto")
    
    print("\033[1;32m系统就绪！输入您的问题 (输入 exit 退出):\033[0m")
    print("\033[90m模式: 服务模式 (支持后台任务)\033[0m")
    print()
    
    # 用于标记是否正在处理任务
    is_processing = False
    agent_running = False
    
    async def output_listener():
        """监听 outbound bus 并显示结果"""
        nonlocal is_processing
        while True:
            try:
                msg = await asyncio.wait_for(scheduler.bus.consume_outbound(), timeout=0.5)
                if msg and isinstance(msg, OutboundMessage):
                    print()
                    print("\033[1;33m回复:\033[0m")
                    print(msg.content)
                    print()
                    print("\033[1;36m" + "-" * 60 + "\033[0m")
                    print()
                    is_processing = False
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
    
    # 启动输出监听任务
    output_task = asyncio.create_task(output_listener())
    
    # 启动 agent 在后台任务
    async def run_agent():
        nonlocal agent_running
        agent_running = True
        await scheduler.run()
    
    agent_task = asyncio.create_task(run_agent())
    
    # 等待 agent 启动
    await asyncio.sleep(0.5)
    
    try:
        while True:
            # 启用 readline 以改善输入体验
            try:
                import readline
            except ImportError:
                pass
            
            # 设置中文编码
            import sys
            import io
            if sys.stdout.encoding != 'utf-8':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            
            # 等待用户输入（在线程中执行，避免阻塞事件循环）
            if is_processing:
                prompt = "\033[90m[处理中...] 输入新消息或等待回复\033[0m "
            else:
                prompt = "[SciDataBot] "
            
            try:
                loop = asyncio.get_event_loop()
                message = await loop.run_in_executor(None, input, prompt)
            except asyncio.CancelledError:
                # 捕获异步任务被取消的异常
                break
            except (KeyboardInterrupt, EOFError):
                print("\n\033[1;31m退出程序\033[0m")
                break
            
            if not message.strip():
                continue
            
            # 处理命令
            msg = message.strip().lower()
            
            if msg == "/help":
                os.system('clear' if os.name == 'posix' else 'cls')
                print_welcome()
                continue
            
            if msg == "/connect":
                os.system('clear' if os.name == 'posix' else 'cls')
                print_welcome()
                print("\033[1;33m正在启动配置向导...\033[0m\n")
                new_config = await run_connect(config_path)
                # Hot-reload: 用新配置重建 provider 并更新 scheduler
                if new_config:
                    try:
                        from src.cli import create_llm_provider
                        new_provider = create_llm_provider(new_config)
                        scheduler.provider = new_provider
                        new_llm = new_config.get("llm", {})
                        new_prov_type = new_llm.get("provider", "minimax")
                        new_model = new_llm.get(new_prov_type, {}).get("model", scheduler.model)
                        scheduler.model = new_model
                        # 同步更新 subagent_manager 中的 provider / model
                        if hasattr(scheduler, "subagent_manager"):
                            scheduler.subagent_manager.provider = new_provider
                            scheduler.subagent_manager.model = new_model
                        print(f"\033[1;32m✓ 新配置已生效：{new_prov_type} / {new_model}\033[0m\n")
                    except Exception as _e:
                        print(f"\033[1;31m热重载失败，请重启 scidatabot：{_e}\033[0m\n")
                continue

            if msg == "/channel":
                os.system('clear' if os.name == 'posix' else 'cls')
                print_welcome()
                print("\033[1;33m正在启动通道配置向导...\033[0m\n")
                await run_channel_config(config_path)
                continue
            
            if msg in ["exit", "quit", "/quit", "/exit"]:
                print("\n\033[1;31m再见！\033[0m")
                break
            
            # 等待当前任务完成
            if is_processing:
                print("\033[90m等待当前任务完成...\033[0m")
                continue
            
            # 发送到 bus
            is_processing = True
            inbound_msg = InboundMessage(
                channel="tui",
                sender_id="user",
                chat_id="direct",
                content=message,
            )
            await scheduler.bus.publish_inbound(inbound_msg)

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\n\033[1;31m退出程序\033[0m")
        sys.stdout.flush()
        scheduler.stop()
        output_task.cancel()
        agent_task.cancel()
        os._exit(0)


async def run_connect(config_path=None):
    """运行连接配置向导 - 支持动态加载的providers"""
    import yaml
    from src.providers import get_registry
    
    # 获取provider registry
    registry = get_registry()
    providers_list = registry.list_providers_for_tui()
    
    # 加载现有配置
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    config_data = {}
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
    
    if "llm" not in config_data:
        config_data["llm"] = {}
    
    # 显示当前配置
    current_provider = config_data.get("llm", {}).get("provider", "minimax")
    print(f"当前 provider: {current_provider}\n")
    
    # 显示可用 providers
    print("可用 providers:")
    for i, (provider_name, metadata) in enumerate(providers_list, 1):
        print(f"  {i}. {metadata.display_name:20} - {metadata.description}")
    print()
    
    # 获取选择
    max_choice = len(providers_list)
    default_choice = "5" if max_choice >= 5 else str(max_choice)
    provider_choice = input(f"选择 provider (1-{max_choice}) [默认 {default_choice}]: ").strip() or default_choice
    
    # 验证选择
    try:
        choice_idx = int(provider_choice) - 1
        if 0 <= choice_idx < len(providers_list):
            provider, metadata = providers_list[choice_idx]
        else:
            provider, metadata = providers_list[int(default_choice) - 1]
    except (ValueError, IndexError):
        provider, metadata = providers_list[int(default_choice) - 1]
    
    print(f"\n已选择: {metadata.display_name}")
    
    # 获取 API key
    current_key = os.environ.get(metadata.env_var, "")
    provider_key = config_data.get("llm", {}).get(provider, {}).get("api_key", "")
    saved_key = provider_key or current_key

    # 显示已保存 key 的前8位和后4位，帮助用户确认是否正确
    if saved_key and len(saved_key) > 12:
        key_hint = f"{saved_key[:8]}...{saved_key[-4:]}"
    elif saved_key:
        key_hint = saved_key
    else:
        key_hint = "未设置"
    
    if metadata.requires_api_key:
        print(f"当前已保存: {key_hint}")
        api_key = input(f"输入新 API Key (直接回车=保留当前值): ").strip()
        if not api_key:
            api_key = saved_key
            print(f"已保留当前 key: {key_hint}")
    else:
        api_key = saved_key or ""
        print(f"此 provider 无需 API Key")
    
    # 获取模型
    default_model = metadata.default_model or "gpt-4o"
    
    if provider == "proxy":
        model = input("Model name [默认 gpt-4o]: ").strip() or "gpt-4o"
        base_url = input("Base URL (如 https://api.openai.com/v1): ").strip() or metadata.base_url
    else:
        model = input(f"Model [默认 {default_model}]: ").strip() or default_model
        base_url = metadata.base_url
    
    # 获取温度
    temp_input = input("Temperature (0.0-1.0) [默认 0.7]: ").strip() or "0.7"
    try:
        temperature = float(temp_input)
    except ValueError:
        temperature = 0.7
    
    # 获取 max tokens
    tokens_input = input("Max tokens [默认 4096]: ").strip() or "4096"
    try:
        max_tokens = int(tokens_input)
    except ValueError:
        max_tokens = 4096
    
    # 更新配置
    config_data["llm"]["provider"] = provider
    config_data["llm"][provider] = {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # 保存配置
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n\033[1;32m✓ 配置已保存到 {config_path}\033[0m")
    print(f"\n{metadata.display_name}")
    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")

    return config_data


async def run_channel_config(config_path=None):
    """运行通道配置向导"""
    import yaml
    from src.channels import ChannelType

    CHANNEL_TYPES = {
        "1": ("console", "Console (本地终端)", None),
        "2": ("feishu", "飞书 (HTTP API - 仅发送)", {"app_id": "", "app_secret": ""}),
        "3": ("feishu_ws", "飞书 WebSocket (收发消息)", {"app_id": "", "app_secret": ""}),
        "4": ("telegram", "Telegram Bot", {"token": ""}),
        "5": ("webhook", "Webhook (HTTP回调)", {"host": "0.0.0.0", "port": 8080, "path": "/webhook"}),
    }

    # 加载现有配置
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    config_data = {}
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    if "channel" not in config_data:
        config_data["channel"] = {}

    # 显示当前配置
    current_type = config_data.get("channel", {}).get("type", "console")
    print(f"当前通道类型: {current_type}\n")

    # 显示选项
    print("可用通道:")
    for key, (channel_id, desc, _) in CHANNEL_TYPES.items():
        print(f"  {key}. {desc}")
    print()

    # 获取选择
    choice = input("选择通道类型 (1-5) [默认 1]: ").strip() or "1"
    channel_id, channel_desc, default_config = CHANNEL_TYPES.get(choice, ("console", "Console", None))

    print(f"\n已选择: {channel_desc}")

    # 根据通道类型获取配置
    if channel_id == "console":
        config_data["channel"]["type"] = "console"
        if "feishu" in config_data["channel"]:
            del config_data["channel"]["feishu"]
        if "feishu_ws" in config_data["channel"]:
            del config_data["channel"]["feishu_ws"]
        if "telegram" in config_data["channel"]:
            del config_data["channel"]["telegram"]

    elif channel_id in ("feishu", "feishu_ws"):
        config_key = "feishu" if channel_id == "feishu" else "feishu_ws"
        current_config = config_data["channel"].get(config_key, {})

        print(f"\n配置 {channel_desc}:")

        # 获取 app_id
        current_app_id = current_config.get("app_id", "")
        app_id = input(f"App ID [当前: {current_app_id or '未设置'}]: ").strip()
        if not app_id:
            app_id = current_app_id

        # 获取 app_secret
        current_secret = current_config.get("app_secret", "")
        secret_display = current_secret[:8] + "..." if current_secret and len(current_secret) > 8 else (current_secret or "未设置")
        app_secret = input(f"App Secret [当前: {secret_display}]: ").strip()
        if not app_secret:
            app_secret = current_secret

        if app_id and app_secret:
            config_data["channel"]["type"] = config_key
            config_data["channel"][config_key] = {
                "app_id": app_id,
                "app_secret": app_secret,
            }
            print(f"\n\033[1;32m✓ 已配置 {channel_desc}\033[0m")
        else:
            print(f"\n\033[1;31m✗ app_id 和 app_secret 都是必填项\033[0m")
            return

    elif channel_id == "telegram":
        current_config = config_data["channel"].get("telegram", {})
        print(f"\n配置 Telegram Bot:")

        current_token = current_config.get("token", "")
        token = input(f"Bot Token [当前: {current_token or '未设置'}]: ").strip()
        if not token:
            token = current_token

        if token:
            config_data["channel"]["type"] = "telegram"
            config_data["channel"]["telegram"] = {"token": token}
            print(f"\n\033[1;32m✓ 已配置 Telegram Bot\033[0m")
        else:
            print(f"\n\033[1;31m✗ Token 是必填项\033[0m")
            return

    elif channel_id == "webhook":
        current_config = config_data["channel"].get("webhook", {})
        print(f"\n配置 Webhook:")

        current_host = current_config.get("host", "0.0.0.0")
        current_port = current_config.get("port", 8080)
        current_path = current_config.get("path", "/webhook")

        host = input(f"Host [当前: {current_host}]: ").strip() or current_host
        try:
            port = int(input(f"Port [当前: {current_port}]: ").strip() or current_port)
        except ValueError:
            port = current_port
        webhook_path = input(f"Path [当前: {current_path}]: ").strip() or current_path

        config_data["channel"]["type"] = "webhook"
        config_data["channel"]["webhook"] = {
            "host": host,
            "port": port,
            "path": webhook_path,
        }
        print(f"\n\033[1;32m✓ 已配置 Webhook\033[0m")

    # 保存配置
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n\033[1;32m✓ 配置已保存到 {config_path}\033[0m")
    print(f"通道类型: {config_data['channel']['type']}")

    return config_data


async def main():
    """主函数 (直接运行)"""
    import sys
    from pathlib import Path

    # 添加路径
    _src_path = Path(__file__).parent.parent
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    os.chdir(Path(__file__).parent.parent)

    # 打印欢迎
    print_welcome("simple")

    # 加载配置
    from src.cli import load_config
    config = load_config(str(Path(__file__).parent.parent / "config.yaml"))

    # 创建 scheduler，传入配置
    from src.main import create_app
    
    async def auto_confirm(tool_name: str, arguments: dict) -> bool:
        print(f"⚠️  危险工具请求: {tool_name}")
        response = input("   确认执行? (y/n): ")
        return response.lower() in ("y", "yes")
    
    scheduler = create_app(config)

    await run_simple_tui(scheduler, str(Path(__file__).parent.parent / "config.yaml"))


if __name__ == "__main__":
    asyncio.run(main())

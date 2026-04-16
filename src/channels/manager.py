"""Channel manager for handling multiple channels."""
import asyncio
from typing import Any, Callable, Dict, List, Optional

from .base import Channel, ChannelType, InboundMessage, OutboundMessage


class ChannelManager:
    """Manages multiple channel instances."""

    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._channel_configs: Dict[str, dict] = {}
        self._handlers: Dict[str, Callable] = {}

    def register_channel(self, name: str, channel: Channel) -> None:
        """Register a channel instance."""
        self._channels[name] = channel

    def add_channel(
        self,
        name: str,
        channel_type: ChannelType,
        config: dict,
    ) -> Channel:
        """Add and configure a channel."""
        if name in self._channels:
            raise ValueError(f"Channel already exists: {name}")

        channel = self._create_channel(channel_type, config)
        self._channels[name] = channel
        self._channel_configs[name] = {"type": channel_type, "config": config}

        return channel

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get channel by name."""
        return self._channels.get(name)

    def remove_channel(self, name: str) -> bool:
        """Remove a channel."""
        if name in self._channels:
            del self._channels[name]
            del self._channel_configs[name]
            return True
        return False

    def set_handler(self, name: str, handler: Callable[[InboundMessage], Any]) -> None:
        """Set message handler for a channel."""
        channel = self._channels.get(name)
        if channel:
            channel.set_message_handler(handler)
            self._handlers[name] = handler

    def set_global_handler(self, handler: Callable[[InboundMessage], Any]) -> None:
        """Set same handler for all channels."""
        for name, channel in self._channels.items():
            channel.set_message_handler(handler)
        self._handlers["_global"] = handler

    async def start_channel(self, name: str) -> None:
        """Start a specific channel."""
        channel = self._channels.get(name)
        if channel:
            await channel.start()

    async def stop_channel(self, name: str) -> None:
        """Stop a specific channel."""
        channel = self._channels.get(name)
        if channel:
            await channel.stop()

    async def start_all(self) -> None:
        """Start all channels."""
        tasks = []
        for name, channel in self._channels.items():
            tasks.append(self._start_channel_safe(name, channel))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _start_channel_safe(self, name: str, channel: Channel) -> None:
        """Start channel with error handling."""
        try:
            await channel.start()
        except Exception as e:
            print(f"Error starting channel {name}: {e}")

    async def stop_all(self) -> None:
        """Stop all channels."""
        tasks = []
        for name, channel in self._channels.items():
            tasks.append(self._stop_channel_safe(name, channel))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _stop_channel_safe(self, name: str, channel: Channel) -> None:
        """Stop channel with error handling."""
        try:
            await channel.stop()
        except Exception as e:
            print(f"Error stopping channel {name}: {e}")

    async def send_message(
        self,
        channel_name: str,
        message: OutboundMessage,
    ) -> str:
        """Send message through a channel."""
        channel = self._channels.get(channel_name)
        if not channel:
            raise ValueError(f"Channel not found: {channel_name}")
        return await channel.send_message(message)

    def list_channels(self) -> List[str]:
        """List all channel names."""
        return list(self._channels.keys())

    def get_channel_info(self, name: str) -> Optional[dict]:
        """Get channel information."""
        config = self._channel_configs.get(name)
        if not config:
            return None
        return {
            "name": name,
            "type": config["type"].value if isinstance(config["type"], ChannelType) else config["type"],
            "running": True,  # Would need more state tracking
        }

    def _create_channel(self, channel_type: ChannelType, config: dict) -> Channel:
        """Create channel instance based on type."""
        from .console import ConsoleChannel
        from .feishu import FeishuChannel, FeishuWebHookChannel
        from .feishu_ws import FeishuWSChannel
        from .wechat import WeChatChannel, WeChatWebhookChannel
        from .webhook import WebhookChannel

        if channel_type == ChannelType.CONSOLE:
            return ConsoleChannel(config)
        elif channel_type == ChannelType.FEISHU:
            if "webhook_url" in config:
                return FeishuWebHookChannel(config)
            return FeishuChannel(config)
        elif channel_type == ChannelType.FEISHU_WS:
            return FeishuWSChannel(config)
        elif channel_type == ChannelType.WECHAT:
            if "webhook_url" in config:
                return WeChatWebhookChannel(config)
            return WeChatChannel(config)
        elif channel_type == ChannelType.WEBHOOK:
            return WebhookChannel(config)
        else:
            raise ValueError(f"Unsupported channel type: {channel_type}")

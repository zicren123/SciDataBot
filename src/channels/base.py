"""Channel base classes and interfaces."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from enum import Enum

from src.bus.events import InboundMessage as BusInboundMessage, OutboundMessage as BusOutboundMessage


class ChannelType(Enum):
    """Channel types."""
    CONSOLE = "console"
    FEISHU = "feishu"
    FEISHU_WS = "feishu_ws"
    WECHAT = "wechat"
    LINE = "line"
    WEB = "web"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


InboundMessage = BusInboundMessage
OutboundMessage = BusOutboundMessage


class Channel(ABC):
    """Base class for channel implementations."""

    def __init__(self, channel_type: ChannelType, config: dict):
        """Initialize channel.

        Args:
            channel_type: Type of channel
            config: Channel configuration
        """
        self.channel_type = channel_type
        self.config = config
        self._message_handler: Optional[Callable] = None

    def set_message_handler(self, handler: Callable[[InboundMessage], Any]):
        """Set handler for incoming messages."""
        self._message_handler = handler

    @abstractmethod
    async def start(self) -> None:
        """Start the channel (connect/listen)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel (disconnect)."""
        pass

    @abstractmethod
    async def send_message(self, message: OutboundMessage) -> str:
        """Send a message to the channel.

        Returns:
            Message ID of sent message.
        """
        pass

    async def handle_inbound(self, message: InboundMessage) -> Any:
        """Handle inbound message with registered handler."""
        if self._message_handler:
            return await self._message_handler(message)
        raise NotImplementedError("No message handler registered")

    @property
    def channel_id(self) -> str:
        """Get unique channel identifier."""
        return self.channel_type.value

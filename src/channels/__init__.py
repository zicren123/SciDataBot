"""Channels module for various platform integrations."""
from .base import Channel, ChannelType, InboundMessage, OutboundMessage
from .console import ConsoleChannel
from .feishu import FeishuChannel, FeishuWebHookChannel
from .feishu_ws import FeishuWSChannel
from .wechat import WeChatChannel, WeChatWebhookChannel
from .webhook import WebhookChannel
from .manager import ChannelManager

__all__ = [
    # Base
    "Channel",
    "ChannelType",
    "InboundMessage",
    "OutboundMessage",
    # Channels
    "ConsoleChannel",
    "FeishuChannel",
    "FeishuWebHookChannel",
    "FeishuWSChannel",
    "WeChatChannel",
    "WeChatWebhookChannel",
    "WebhookChannel",
    # Manager
    "ChannelManager",
]

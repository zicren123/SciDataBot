"""MiniMax LLM Provider - 使用 Anthropic 兼容格式."""
import asyncio
import json
import os
import uuid
from typing import Any, AsyncIterator, Optional

import aiohttp
import certifi
from loguru import logger
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


class MiniMaxProvider(LLMProvider):
    """MiniMax LLM provider - 使用 Anthropic 兼容格式."""

    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimaxi.com/anthropic",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
        verify_ssl: bool = True,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        """Initialize MiniMax provider.

        Args:
            model: Model name (e.g., "MiniMax-M2.5").
            api_key: MiniMax API key.
            base_url: Anthropic 兼容 API 端点.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
            max_retries: Maximum number of retries for rate limit errors.
            retry_delay: Delay between retries in seconds.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = max(timeout, 120)  # 至少120秒
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 使用信号量限制并发数,防止触发速率限制
        self._semaphore = asyncio.Semaphore(20)

        if not self.api_key:
            raise ValueError("MiniMax API key is required. Set ANTHROPIC_AUTH_TOKEN env var or pass api_key.")

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request - 使用 Anthropic 兼容格式."""
        
        # 使用信号量限制并发数
        async with self._semaphore:
            return await self._chat_with_retry(messages, tools, **kwargs)
    
    async def _chat_with_retry(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """带重试的聊天请求"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self._do_chat(messages, tools, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是速率限制错误
                if "rate limit" in error_str.lower() or "500" in error_str or "当前请求量较高" in error_str:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"MiniMax API 速率限制，{wait_time}秒后重试 ({attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # 非速率限制错误，直接抛出
                    raise e
        
        # 所有重试都失败
        raise last_error
    
    async def _do_chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """实际的聊天请求"""
        url = f"{self.base_url}/v1/messages"

        # 使用基类方法转换消息和工具
        anthropic_messages, system_prompt = self.convert_messages_anthropic(messages)
        anthropic_tools = self.convert_tools_anthropic(tools)

        logger.info(f"[MiniMax] Sending request with {len(anthropic_messages)} messages, tools: {len(anthropic_tools) if anthropic_tools else 0}")

        # Build request parameters
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": anthropic_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_prompt:
            payload["system"] = system_prompt

        if anthropic_tools:
            payload["tools"] = anthropic_tools

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        # 创建 TCPConnector 来控制 SSL 验证
        connector = aiohttp.TCPConnector(ssl=False if not self.verify_ssl else None)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                logger.info(f"[MiniMax] Response status: {response.status}")
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[MiniMax] Error response: {error_text}")
                    raise Exception(f"MiniMax API error: {response.status} - {error_text}")

                result = await response.json()
                logger.info(f"[MiniMax] Response received: {result.get('content', [])}")

        # Parse response
        content = ""
        tool_calls = None
        has_tool_calls = False

        for block in result.get("content", []):
            block_type = block.get("type", "")
            if block_type == "text":
                content += block.get("text", "")
            elif block_type == "thinking":
                # Skip thinking blocks
                pass
            elif block_type == "tool_use":
                has_tool_calls = True
                if tool_calls is None:
                    tool_calls = []
                tool_id = block.get("id", "") or f"toolu_{uuid.uuid4().hex[:8]}"
                tool_calls.append(ToolCall(
                    id=tool_id,
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        return LLMResponse(
            content=content,
            has_tool_calls=has_tool_calls,
            tool_calls=tool_calls,
            model=result.get("model", self.model),
            usage=result.get("usage", {}),
            finish_reason=result.get("stop_reason"),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completion responses."""
        raise NotImplementedError("Streaming not supported yet in MiniMax provider.")

    async def close(self):
        """Close the client connection."""
        pass

    def get_default_model(self) -> str:
        """Get default model name."""
        return self.model

    @property
    def name(self) -> str:
        return f"minimax-{self.model}"


# Backwards compatibility alias
class MiniMaxAnthropicProvider(MiniMaxProvider):
    """Alias for MiniMaxProvider using Anthropic format."""
    pass
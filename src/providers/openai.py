"""OpenAI LLM Provider."""
import os
from typing import Any, AsyncIterator, Optional

from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the OpenAI SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Custom base URL for API-compatible services.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key.")

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tools to enable.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            LLMResponse object with content and tool calls.
        """
        # 使用基类方法转换消息和工具
        openai_messages = self.convert_messages_openai(messages)
        openai_tools = self.convert_tools_openai(tools)

        # Build request parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if openai_tools:
            params["tools"] = openai_tools

        if self.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        # Make request
        response = await self._client.chat.completions.create(**params)

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                import json as _json
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            model=response.model,
            usage=dict(response.usage) if response.usage else {},
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completion responses.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tools to enable.
            **kwargs: Additional parameters.

        Yields:
            Content chunks as they arrive.
        """
        openai_messages = []
        for msg in messages:
            if msg.role == "user":
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })

        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                })

        params = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if openai_tools:
            params["tools"] = openai_tools

        if self.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        stream = await self._client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self):
        """Close the client connection."""
        await self._client.close()

    def get_default_model(self) -> str:
        """Get default model name."""
        return self.model

    @property
    def name(self) -> str:
        return f"openai-{self.model}"

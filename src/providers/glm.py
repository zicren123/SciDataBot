"""GLM (ZhipuAI) LLM Provider - 使用 OpenAI 兼容格式."""
import json
import os
from typing import Any, AsyncIterator, Optional

import aiohttp
import certifi
from loguru import logger
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


class GLMProvider(LLMProvider):
    """GLM (ZhipuAI) LLM provider - 使用 OpenAI 兼容格式."""


    def __init__(
        self,
        model: str = "glm-4-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize GLM provider.

        Args:
            model: Model name (e.g., "glm-4-flash").
            api_key: GLM API key.
            base_url: API 端点.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = timeout
        self.name = model

        if not self.api_key:
            raise ValueError("GLM API key is required. Set ZHIPU_API_KEY or GLM_API_KEY env var or pass api_key.")

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request - 使用 OpenAI 兼容格式."""
        url = f"{self.base_url}/chat/completions"

        # 使用基类方法转换消息和工具
        openai_messages = self.convert_messages_openai(messages)
        openai_tools = self.convert_tools_openai(tools)

        # Build request parameters
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = max_tokens

        if openai_tools:
            payload["tools"] = openai_tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"GLM API error: {response.status}\n"
                            f"URL: {url}\n"
                            f"Details: {error_text}"
                        )

                    result = await response.json()
        except aiohttp.ClientError as e:
            raise Exception(
                f"GLM API connection error: {str(e)}\n"
                f"Base URL: {self.base_url}\n"
                f"Check your API key and network connection."
            )

        # Parse response
        content = ""
        tool_calls = None
        has_tool_calls = False

        message = result.get("choices", [{}])[0].get("message", {})
        
        # Handle content
        msg_content = message.get("content", "")
        if msg_content:
            content = msg_content

        # Handle tool calls
        tc_list = message.get("tool_calls", [])
        if tc_list:
            has_tool_calls = True
            tool_calls = []
            for tc in tc_list:
                func = tc.get("function", {})
                tc_args = func.get("arguments", {})
                
                # Parse arguments if it's a string
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except:
                        tc_args = {}
                
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=tc_args,
                ))

        return LLMResponse(
            content=content,
            has_tool_calls=has_tool_calls,
            tool_calls=tool_calls,
            model=result.get("model", self.model),
            usage=result.get("usage", {}),
            finish_reason=result.get("choices", [{}])[0].get("finish_reason"),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completion responses."""
        raise NotImplementedError("Streaming not supported yet in GLM provider.")

    async def close(self):
        """Close the client connection."""
        pass

    def get_default_model(self) -> str:
        """Get default model name."""
        return self.model

    @property
    def name(self) -> str:
        return f"glm-{self.model}"

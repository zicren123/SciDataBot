"""DeepSeek LLM Provider - OpenAI Compatible."""
import json
import os
from typing import Any, AsyncIterator, Optional
import aiohttp
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider."""


    def __init__(
        self,
        model: str = "deepseek-coder",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize DeepSeek provider.

        Args:
            model: Model name (e.g., "deepseek-chat", "deepseek-coder").
            api_key: DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.
            base_url: Custom API base URL (default: DeepSeek official API).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = timeout
        self.name = "deepseek"

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY env var or pass api_key."
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
            **kwargs: Additional parameters.

        Returns:
            LLMResponse object with content and tool calls.
        """
        url = f"{self.base_url}/chat/completions"

        # Convert messages and tools to OpenAI format
        openai_messages = self.convert_messages_openai(messages)
        openai_tools = self.convert_tools_openai(tools)

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

        if not aiohttp:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"DeepSeek API error: {response.status}\n"
                            f"URL: {url}\n"
                            f"Details: {error_text}"
                        )

                    result = await response.json()
        except aiohttp.ClientError as e:
            raise Exception(
                f"DeepSeek API connection error: {str(e)}\n"
                f"Base URL: {self.base_url}\n"
                f"Check your API key and network connection."
            )

        # Parse response
        choice = result["choices"][0]
        message = choice["message"]

        content = message.get("content", "") or ""
        tool_calls = None

        # Extract tool calls if present
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        return LLMResponse(
            content=content,
            has_tool_calls=bool(tool_calls),
            tool_calls=tool_calls,
            model=result.get("model", self.model),
            usage=result.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
        )

    def _sanitize_function_name(self, name: str, used_names: set) -> str:
        """Sanitize function name to meet API requirements.
        API requires function names to:
        - Start with a letter
        - Contain only letters, numbers, underscores, and dashes
        - Be unique within the request
        """
        import re

        original = name

        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'func_' + sanitized

        # Clean up consecutive and trailing underscores
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')

        # If empty or just 'func', use fallback
        if not sanitized or sanitized == 'func':
            alphanum = ''.join(c for c in original if c.isalnum())
            sanitized = 'func_' + (alphanum[:20] or 'tool')

        # Truncate to leave room for uniqueness suffix
        base_name = sanitized[:55]

        # Ensure uniqueness
        final_name = base_name
        counter = 0
        while final_name in used_names:
            suffix = f"_{counter}"
            final_name = base_name[:64 - len(suffix)] + suffix
            counter += 1

        used_names.add(final_name)

        # Debug: log if name was changed
        if final_name != original:
            print(f"[DeepSeekProvider] Sanitized: '{original}' -> '{final_name}'")

        return final_name

    def convert_tools_openai(self, tools: list) -> list[dict] | None:
        """Convert tools to OpenAI format with proper JSON Schema for parameters."""
        if not tools:
            return None

        used_names = set()
        openai_tools = []

        for i, tool in enumerate(tools):
        # 获取工具属性
            if isinstance(tool, dict):
                tool_name = tool.get("name", "") or f"tool_{i}"
                description = tool.get("description", "") or ""
                parameters = tool.get("parameters", {})
            else:
                tool_name = getattr(tool, "name", "") or f"tool_{i}"
                description = getattr(tool, "description", "") or ""
                parameters = getattr(tool, "parameters", {})

            # 清理函数名
            sanitized_name = self._sanitize_function_name(tool_name, used_names)

            # 修复 parameters: 确保是有效的 JSON Schema，且 type 为 "object"
            if not parameters:
                # 如果没有参数，创建空的 object schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            elif isinstance(parameters, dict):
                # 确保有 type: object
                if "type" not in parameters:
                    parameters = {
                        "type": "object",
                        "properties": parameters.get("properties", {}),
                        "required": parameters.get("required", [])
                    }
                # 如果 type 为 None 或其他非 object 值，强制设为 object
                elif parameters.get("type") != "object":
                    parameters["type"] = "object"
                
                # 整理 properties 中的每个参数: 确保 description 不是 None
                if "properties" in parameters and isinstance(parameters["properties"], dict):
                    for prop_name, prop_schema in parameters["properties"].items():
                        if isinstance(prop_schema, dict):
                            # 确保 description 不是 None
                            if "description" in prop_schema and prop_schema["description"] is None:
                                prop_schema["description"] = f"Parameter: {prop_name}"
        
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": description or f"Function {sanitized_name}",
                    "parameters": parameters,
                },
            })

        return openai_tools

    def get_default_model(self) -> str:
        """Get default model."""
        return self.model

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
        url = f"{self.base_url}/chat/completions"

        openai_messages = self.convert_messages_openai(messages)
        openai_tools = self.convert_tools_openai(tools)

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
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

        if not aiohttp:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"DeepSeek API error: {response.status} - {error_text}"
                    )

                async for line in response.content.iter_any():
                    if not line:
                        continue

                    line_str = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
                    if not line_str.startswith("data: "):
                        continue

                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError as e:
                        # Log but continue on JSON parsing errors
                        import sys
                        print(f"Failed to parse SSE line: {data_str}", file=sys.stderr)
                        continue
                    except Exception as e:
                        import sys
                        print(f"Error processing stream: {e}", file=sys.stderr)
                        continue

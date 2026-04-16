"""OpenAI Compatible Provider - Supports most API Relays/Proxies."""
import json
import os
from typing import Any, AsyncIterator, Optional
import aiohttp
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class ProxyProvider(LLMProvider):
    """通用 OpenAI 兼容代理提供商，支持各类中转站 API。"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """初始化中转站提供商。

        Args:
            model: 模型名称 (例如 "gpt-4o")。
            api_key: 中转站提供的 API Key。
            base_url: 中转站的 API 基础地址 (例如 "https://api.example.com/v1")。
            temperature: 采样温度。
            max_tokens: 最大生成长度。
            timeout: 请求超时时间。
        """
        self.model = model
        # 尝试从环境变量获取，也可以在初始化时传入
        self.api_key = api_key or os.environ.get("ANY_API_KEY")
        self.base_url = base_url or os.environ.get("ANY_BASE_URL", "https://api.openai.com/v1")
        
        # 确保 base_url 不以斜杠结尾，方便拼接
        self.base_url = self.base_url.rstrip('/')
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = model
        self._tool_name_map: dict[str, str] = {}
        if not self.api_key:
            raise ValueError("API key is required for OpenAICompatibleProvider.")

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
        self._tool_name_map = {}

        for i, tool in enumerate(tools):
        # 获取工具属性
            if isinstance(tool, dict):
                if "function" in tool:
                    func = tool.get("function", {}) or {}
                    tool_name = func.get("name", "") or f"tool_{i}"
                    description = func.get("description", "")
                    parameters = func.get("parameters", {})
                else:
                    tool_name = tool.get("name", "") or f"tool_{i}"
                    description = tool.get("description", "")
                    parameters = tool.get("parameters", {})
            else:
                tool_name = getattr(tool, "name", "") or f"tool_{i}"
                description = getattr(tool, "description", "")
                parameters = getattr(tool, "parameters", {})

            # 清理函数名
            sanitized_name = self._sanitize_function_name(tool_name, used_names)
            self._tool_name_map[sanitized_name] = tool_name

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
        
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": sanitized_name,
                    "description": description or f"Function {sanitized_name}",
                    "parameters": parameters,
                },
            })

        return openai_tools

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        openai_messages = self.convert_messages_openai(messages)
        openai_tools = self.convert_tools_openai(tools)

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if self.max_tokens:
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        if openai_tools:
            payload["tools"] = openai_tools
            # 某些中转站对 tool_choice 有特殊要求，默认 auto
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

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
                            f"API Relay Error: {response.status}\n"
                            f"URL: {url}\n"
                            f"Details: {error_text}"
                        )

                    result = await response.json()
        except aiohttp.ClientError as e:
            raise Exception(
                f"API Relay connection error: {str(e)}\n"
                f"Base URL: {self.base_url}\n"
                f"Check your API key and network connection."
            )

        choice = result["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        
        # 处理工具调用
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                try:
                    args = tc["function"]["arguments"]
                    args_json = json.loads(args) if isinstance(args, str) else args
                    raw_name = tc["function"]["name"]
                    mapped_name = self._tool_name_map.get(raw_name, raw_name)
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=mapped_name,
                        arguments=args_json,
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

        return LLMResponse(
            content=content,
            has_tool_calls=bool(tool_calls),
            tool_calls=tool_calls if tool_calls else None,
            model=result.get("model", self.model),
            usage=result.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
        )
    
    def get_default_model(self) -> str:
        """Get default model."""
        return self.model

    async def stream(
        self,
        messages: list[LLMMessage],
        tools: Optional[list[LLMTool]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": self.convert_messages_openai(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API Relay Stream Error: {response.status}\n"
                        f"URL: {url}\n"
                        f"Details: {error_text}"
                    )

                async for line in response.content.iter_any():
                    line_str = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def get_default_model(self) -> str:
        return self.model
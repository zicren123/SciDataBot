"""Kimi (Moon) LLM Provider - OpenAI Compatible."""
import json
import os
from typing import Any, AsyncIterator, Optional
import aiohttp
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class KimiProvider(LLMProvider):
    """Kimi (Moon) LLM provider."""

    def __init__(
        self,
        model: str = "moonshot-v1-8k",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize Kimi provider.

        Args:
            model: Model name (e.g., "moonshot-v1-8k", "moonshot-v1-128k").
            api_key: Kimi API key. Falls back to KIMI_API_KEY or MOONSHOT_API_KEY env var.
            base_url: Custom API base URL (default: Moon official API).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.base_url = base_url or "https://api.moonshot.cn/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = timeout
        self.name = "kimi"

        if not self.api_key:
            raise ValueError(
                "Kimi API key is required. Set KIMI_API_KEY or MOONSHOT_API_KEY env var or pass api_key."
            )

    def _sanitize_function_name(self, name: str, used_names: set) -> str:
        """Sanitize function name to meet Kimi API requirements.
    
        Kimi requires function names to:
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
            print(f"[KimiProvider] Sanitized: '{original}' -> '{final_name}'")
    
        return final_name

    def convert_tools_openai(self, tools: list) -> list[dict]:
        """Convert tools to OpenAI format with Kimi-specific sanitization."""
        if not tools:
            return None
    
        used_names = set()  # Track used names for this request
        openai_tools = []
    
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                tool_name = tool.get("name", "") or f"tool_{i}"
                sanitized_name = self._sanitize_function_name(tool_name, used_names)
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": sanitized_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
            else:
                tool_name = getattr(tool, "name", "") or f"tool_{i}"
                sanitized_name = self._sanitize_function_name(tool_name, used_names)
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": sanitized_name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "parameters", {}),
                    },
                })
        return openai_tools

    def _ensure_reasoning_for_tool_calls(self, messages: list[dict]) -> list[dict]:
        """Ensure reasoning_content exists on assistant messages with tool calls for Kimi."""
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if "reasoning_content" not in msg:
                    msg["reasoning_content"] = ""
        return messages

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
        openai_messages = self._ensure_reasoning_for_tool_calls(openai_messages)
        openai_tools = self.convert_tools_openai(tools)

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "enable_thinking": kwargs.get("enable_thinking", False),
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
                            f"Kimi API error: {response.status}\n"
                            f"URL: {url}\n"
                            f"Details: {error_text}"
                        )

                    result = await response.json()
        except aiohttp.ClientError as e:
            raise Exception(
                f"Kimi API connection error: {str(e)}\n"
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
        openai_messages = self._ensure_reasoning_for_tool_calls(openai_messages)
        openai_tools = self.convert_tools_openai(tools)

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
            "enable_thinking": kwargs.get("enable_thinking", False),
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
                        f"Kimi API error: {response.status}\nDetails: {error_text}"
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

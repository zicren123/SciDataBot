"""Shanghai Ailab Intern-S1 LLM Provider."""
import json
import os
from typing import Any, AsyncIterator, Optional
import aiohttp
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class InternS1Provider(LLMProvider):
    """Shanghai Ailab Intern-S1 LLM provider.
    
    Supports Intern-S1 models through OpenAI-compatible API.
    """

    SUPPORTED_MODELS = [
        "internS1",
        "intern-s1",
    ]

    def __init__(
        self,
        model: str = "internS1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize Intern-S1 provider.

        Args:
            model: Model name (e.g., "internS1", "intern-s1").
            api_key: Intern-S1 API key. Falls back to INTERN_S1_API_KEY env var.
            base_url: Custom API base URL (default: Shanghai Ailab official API).
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("INTERN_S1_API_KEY")
        self.base_url = base_url or "https://api.shanghai-ailab.com/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Intern-S1 API key is required. Set INTERN_S1_API_KEY env var or pass api_key."
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
                            f"Intern-S1 API error: {response.status}\n"
                            f"URL: {url}\n"
                            f"Details: {error_text}"
                        )

                    result = await response.json()
        except aiohttp.ClientError as e:
            raise Exception(
                f"Intern-S1 API connection error: {str(e)}\n"
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
        return "internS1"

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
                        f"Intern-S1 API error: {response.status} - {error_text}"
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

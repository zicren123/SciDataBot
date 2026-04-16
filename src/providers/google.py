"""Google Gemini LLM Provider."""
import json
import os
from typing import Any, AsyncIterator, Optional
import aiohttp
from .base import LLMProvider, LLMMessage, LLMTool, LLMResponse, ToolCall


class GoogleGeminiProvider(LLMProvider):
    """Google Gemini LLM provider using official Google SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize Google Gemini provider.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required. Install with: pip install google-generativeai"
            )

        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY env var or pass api_key.")

        genai.configure(api_key=self.api_key)

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
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")

            if role == "system":
                continue  # System messages handled separately in Gemini
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            gemini_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    gemini_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": {
                                "type": "object",
                                "properties": tool.get("parameters", {}),
                            },
                        },
                    })

        model = genai.GenerativeModel(self.model)

        # Make request
        response = model.generate_content(
            gemini_messages,
            tools=gemini_tools,
            generation_config={
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            },
        )

        # Parse response
        content = response.text or ""
        tool_calls = None

        # Extract tool calls if present
        if hasattr(response, "tool_calls"):
            tool_calls = []
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("args", {}),
                ))

        return LLMResponse(
            content=content,
            has_tool_calls=bool(tool_calls),
            tool_calls=tool_calls,
            model=self.model,
            finish_reason=response.candidates[0].finish_reason if response.candidates else None,
        )

    def get_default_model(self) -> str:
        """Get default model."""
        return "gemini-2.0-flash"

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
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")

            if role == "system":
                continue
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})

        model = genai.GenerativeModel(self.model)

        # Stream response
        response = model.generate_content(
            gemini_messages,
            stream=True,
            generation_config={
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            },
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def close(self):
        """Close the client connection."""
        pass

    @property
    def name(self) -> str:
        return f"google-gemini-{self.model}"

"""Web 工具 - 移植自 nanobot"""

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from .base import Tool
from src.bus.events import OutboundMessage

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using Brave Search API."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets. Use this when you need up-to-date information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }

    def __init__(self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None, bus = None):
        self._init_api_key = api_key
        self.max_results = max_results
        self.proxy = proxy
        self.bus = bus  # 事件总线，用于向 TUI 发送消息

    @property
    def api_key(self) -> str:
        """Resolve API key at call time so env/config changes are picked up."""
        return self._init_api_key or os.environ.get("BRAVE_API_KEY", "")
    
    async def _notify_tui(self, message: str) -> None:
        """向 TUI 发送提示消息。"""
        if self.bus:
            try:
                await self.bus.publish_outbound(OutboundMessage(
                    channel="tui",
                    chat_id="console",
                    content=f"🔍 {message}"
                ))
            except Exception:
                # 如果发送失败，静默处理
                pass

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        # 如果没有 Brave API key，尝试使用 DuckDuckGo
        if not self.api_key:
            return await self._ddg_search(query, count)
        
        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Install with: pip install httpx"

        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])[:n]
            if not results:
                return f"No results for: {query}"

            # 向 TUI 通知搜索成功
            await self._notify_tui(f"网络搜索成功：找到 {len(results)} 条关于 '{query}' 的结果")

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    async def _ddg_search(self, query: str, count: int | None = None) -> str:
        """使用 DuckDuckGo Instant Answer API (免费)"""
        import aiohttp
        
        n = min(max(count or self.max_results, 1), 10)
        url = "https://api.duckduckgo.com/"
        
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url, params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1
                }) as r:
                    r.raise_for_status()
                    data = await r.json()
            
            # 获取即时答案
            if data.get("AnswerType") == "calc" or data.get("AbstractText"):
                answer = data.get("AbstractText", "")
                if answer:
                    # 向 TUI 通知搜索成功
                    await self._notify_tui(f"网络搜索成功：获取到 DuckDuckGo 即时答案")
                    return f"Answer: {answer}\n\nRelated: {data.get('AbstractURL', '')}"
            
            # 获取相关主题
            results = []
            for topic in data.get("RelatedTopics", [])[:n]:
                if "Text" in topic and "FirstURL" in topic:
                    results.append(f"- {topic['Text']}: {topic['FirstURL']}")
            
            if results:
                # 向 TUI 通知搜索成功
                await self._notify_tui(f"网络搜索成功：找到 {len(results)} 个相关主题")
                return f"Results for '{query}':\n" + "\n".join(results)
            
            return f"No results for: {query}"
            
        except Exception as e:
            return f"Error: {e}"


class KimiWebSearchTool(WebSearchTool):
    """Search the web using Kimi Search API directly."""

    name = "kimi_web_search"
    description = "Search the web via Kimi Search API. Returns titles, URLs, and snippets."

    def __init__(
        self,
        max_results: int = 5,
        proxy: str | None = None,
        kimi_api_key: str | None = None,
        kimi_base_url: str | None = None,
        kimi_search_path: str | None = None,
        timeout: float = 10.0,
        bus = None,
    ):
        super().__init__(api_key=None, max_results=max_results, proxy=proxy, bus=bus)
        self.kimi_api_key = kimi_api_key
        self.kimi_base_url = (kimi_base_url or "https://api.moonshot.cn/v1").rstrip("/")
        self.timeout = timeout

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        """Use Kimi builtin $web_search via chat/completions to avoid /search 404."""
        import aiohttp

        if not self.kimi_api_key:
            return "Error: KIMI_API_KEY or MOONSHOT_API_KEY not set"

        n = min(max(count or self.max_results, 1), 10)
        url = f"{self.kimi_base_url}/chat/completions"

        messages = [
            {"role": "system", "content": "你是 Kimi。"},
            {"role": "user", "content": f"请搜索以下内容并返回{n}条结果，包含标题、URL与摘要：\n{query}"},
        ]

        tools = [
            {
                "type": "builtin_function",
                "function": {"name": "$web_search"},
            }
        ]

        payload = {
            "model": "kimi-k2.5",
            "messages": messages,
            "tools": tools,
            "temperature": 0.6,
            "max_tokens": 2048,
            "enable_thinking": False,
            "thinking": {"type": "disabled"},
        }

        headers = {
            "Authorization": f"Bearer {self.kimi_api_key}",
            "Content-Type": "application/json",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                max_tool_rounds = 3
                for _ in range(max_tool_rounds):
                    for msg in messages:
                        if msg.get("role") == "assistant" and msg.get("tool_calls") and "reasoning_content" not in msg:
                            msg["reasoning_content"] = ""
                    async with session.post(url, json=payload, headers=headers, proxy=self.proxy) as r:
                        if r.status != 200:
                            error_text = await r.text()
                            return (
                                f"Kimi search error: {r.status}\n"
                                f"URL: {url}\n"
                                f"Details: {error_text}"
                            )
                        data = await r.json()

                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason")

                    if finish_reason != "tool_calls":
                        # 向 TUI 通知搜索成功
                        await self._notify_tui(f"Kimi 网络搜索成功：已获取 '{query}' 的搜索结果")
                        return message.get("content", "") or "No content"

                    tool_calls = message.get("tool_calls") or []
                    if message.get("role") == "assistant" and "reasoning_content" not in message:
                        message["reasoning_content"] = ""
                    messages.append(message)

                    for tc in tool_calls:
                        function = tc.get("function", {})
                        args = function.get("arguments", "{}")
                        try:
                            args = json.loads(args) if isinstance(args, str) else args
                        except Exception:
                            args = {}

                        tool_name = function.get("name", "$web_search")
                        tool_query = args.get("query") or args.get("q") or query
                        tool_count = args.get("count") or args.get("n") or n

                        if tool_name in ("$web_search", "web_search"):
                            content = await self._call_kimi_search_api(
                                session=session,
                                query=tool_query,
                                count=tool_count,
                            )
                        else:
                            content = json.dumps({"error": f"Unsupported tool: {tool_name}"}, ensure_ascii=False)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "name": tool_name,
                            "content": content,
                        })

                    payload["messages"] = messages

                return "Error: tool_calls exceeded limit"

        except aiohttp.ClientError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    async def _call_kimi_search_api(self, session, query: str, count: int | None) -> str:
        """Call Kimi Search API and normalize results; fallback to DDG if needed."""
        n = min(max(count or self.max_results, 1), 10)
        url = f"{self.kimi_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.kimi_api_key}",
            "Content-Type": "application/json",
        }

        payload = {"query": query, "count": n}

        try:
            async with session.post(url, json=payload, headers=headers, proxy=self.proxy) as r:
                if r.status == 200:
                    data = await r.json()
                    return self._format_search_results(query, data, n)
        except Exception:
            pass

        try:
            async with session.get(url, params={"query": query, "count": n}, headers=headers, proxy=self.proxy) as r:
                if r.status == 200:
                    data = await r.json()
                    return self._format_search_results(query, data, n)
        except Exception:
            pass

        return await self._ddg_search(query, n)

    def _format_search_results(self, query: str, data: dict, n: int) -> str:
        """Normalize various search API payloads to a readable string."""
        results = (
            data.get("results")
            or data.get("data")
            or data.get("web", {}).get("results")
            or data.get("items")
            or []
        )

        if not isinstance(results, list):
            return f"Results for: {query}\n\n{json.dumps(data, ensure_ascii=False)}"

        results = results[:n]
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results, 1):
            title = item.get("title") or item.get("name") or ""
            url = item.get("url") or item.get("link") or ""
            snippet = item.get("snippet") or item.get("description") or item.get("summary") or ""
            lines.append(f"{i}. {title}\n   {url}")
            if snippet:
                lines.append(f"   {snippet}")

        return "\n".join(lines)

class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }

    def __init__(self, max_chars: int = 50000, proxy: str | None = None):
        self.max_chars = max_chars
        self.proxy = proxy

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        import aiohttp
        
        max_chars = maxChars or self.max_chars
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=False)
            text = ""
            ctype = ""
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url, headers={"User-Agent": USER_AGENT}) as r:
                    r.raise_for_status()
                    text = await r.text()
                    ctype = r.headers.get("content-type", "")

            if "application/json" in ctype:
                text, extractor = json.dumps(json.loads(text), indent=2, ensure_ascii=False), "json"
            elif "text/html" in ctype or text[:256].lower().startswith(("<!doctype", "<html")):
                try:
                    from readability import Document
                    doc = Document(text)
                    content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                    text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                    extractor = "readability"
                except ImportError:
                    text, extractor = _strip_tags(text), "raw"
            else:
                text, extractor = text, "raw"

            truncated = len(text) > max_chars
            if truncated: text = text[:max_chars]

            return json.dumps({
                "url": url, 
                "status": r.status,
                "extractor": extractor, 
                "truncated": truncated, 
                "length": len(text), 
                "text": text
            }, ensure_ascii=False)
        except aiohttp.ClientError as e:
            return json.dumps({"error": f"Client error: {e}", "url": url}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))


class BrowserTool(Tool):
    """Browser automation tool using Playwright."""

    name = "browser"
    description = "Browser automation with Playwright - screenshot, click, fill, evaluate"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["screenshot", "click", "fill", "evaluate"]},
                "url": {"type": "string", "description": "URL to interact with"},
                "selector": {"type": "string", "description": "CSS selector"},
                "value": {"type": "string", "description": "Value to fill (for fill operation)"},
                "script": {"type": "string", "description": "JavaScript to evaluate (for evaluate operation)"}
            },
            "required": ["operation", "url"]
        }

    async def execute(self, operation: str, url: str, selector: str = None, value: str = None, script: str = None, **kwargs) -> str:
        """Execute browser operation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return json.dumps({"error": "Playwright not installed. Install with: pip install playwright"})

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)

                if operation == "screenshot":
                    if selector:
                        element = await page.query_selector(selector)
                        if element:
                            screenshot = await element.screenshot()
                        else:
                            return json.dumps({"error": "Selector not found"})
                    else:
                        screenshot = await page.screenshot()
                    import base64
                    b64 = base64.b64encode(screenshot).decode()
                    return json.dumps({"url": url, "screenshot": f"data:image/png;base64,{b64}"})

                elif operation == "click":
                    if not selector:
                        return json.dumps({"error": "selector required for click"})
                    await page.click(selector)
                    await browser.close()
                    return json.dumps({"clicked": selector})

                elif operation == "fill":
                    if not selector or not value:
                        return json.dumps({"error": "selector and value required for fill"})
                    await page.fill(selector, value)
                    await browser.close()
                    return json.dumps({"filled": selector, "value": value})

                elif operation == "evaluate":
                    if not script:
                        return json.dumps({"error": "script required for evaluate"})
                    result = await page.evaluate(script)
                    await browser.close()
                    return json.dumps({"result": result})

                await browser.close()
                return json.dumps({"error": f"Unknown operation: {operation}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

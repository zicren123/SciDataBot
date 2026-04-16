"""LLM Provider 接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class LLMMessage:
    """LLM 消息"""
    role: str  # system, user, assistant, tool
    content: str
    tool_call_id: Optional[str] = None


@dataclass
class LLMTool:
    """LLM 工具定义"""
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    has_tool_calls: bool
    tool_calls: List[ToolCall] = None
    thinking_blocks: list = None
    model: str = None
    usage: dict = None
    finish_reason: str = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.has_tool_calls is None:
            self.has_tool_calls = bool(self.tool_calls)


class LLMProvider(ABC):
    """LLM Provider 基类"""

    @abstractmethod
    async def chat(
        self,
        messages: list,
        tools: list = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """调用 LLM"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """获取默认模型"""
        pass

    def convert_messages_openai(self, messages: list) -> list[dict]:
        """将消息转换为OpenAI格式"""
        import json
        
        openai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                tool_call_id = getattr(msg, "tool_call_id", None)
        
            # 严格确保 content 是字符串，不能是 None 或其他类型
            if content is None:
                content = ""
            elif isinstance(content, list):
                # 如果是列表（某些富文本格式），转换为 JSON 字符串
                content = json.dumps(content, ensure_ascii=False)
            elif not isinstance(content, str):
                # 其他类型转换为字符串
                content = str(content)
        
            if role == "user":
                openai_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(msg, dict) and msg.get("tool_calls"):
                    tool_calls = []
                    for tc in msg.get("tool_calls", []):
                        if isinstance(tc, dict):
                            if "type" not in tc:
                                tc = {**tc, "type": "function"}
                            elif not tc.get("type"):
                                tc["type"] = "function"
                            tool_calls.append(tc)
                        else:
                            tool_calls.append(tc)
                    openai_messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    })
                else:
                    openai_messages.append({"role": "assistant", "content": content})
            elif role == "system":
                openai_messages.append({"role": "system", "content": content})
            elif role == "tool":
                # tool 消息必须有 content 和 tool_call_id
                # Ensure tool_call_id is never empty for OpenAI-compatible APIs
                final_tool_call_id = tool_call_id or "unknown"
                openai_messages.append({
                    "role": "tool",
                    "content": content if content else "",  # 确保不是 None
                    "tool_call_id": final_tool_call_id,
                })
        return openai_messages

    def convert_messages_anthropic(self, messages: list) -> tuple[list[dict], str]:
        """将消息转换为Anthropic格式
        
        返回: (messages, system_prompt)
        Anthropic将system消息单独提取为参数
        """
        anthropic_messages = []
        system_prompt = None

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                tool_call_id = getattr(msg, "tool_call_id", None)
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                if isinstance(content, list):
                    processed_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            processed_content.append({
                                "type": "tool_result",
                                "tool_use_id": item.get("tool_use_id", tool_call_id or "unknown"),
                                "content": item.get("content", ""),
                            })
                        else:
                            processed_content.append(item)
                    anthropic_messages.append({"role": "user", "content": processed_content})
                else:
                    anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(msg, dict) and "tool_calls" in msg:
                    msg_content = content or ""
                    tool_calls_list = msg.get("tool_calls", [])
                    
                    blocks = []
                    if msg_content:
                        blocks.append({"type": "text", "text": msg_content})
                    
                    for tc in tool_calls_list:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            # Debug: log tc_id
                            if not tc_id:
                                import uuid
                                tc_id = f"toolu_{uuid.uuid4().hex[:8]}"
                            func = tc.get("function", {})
                            tc_name = func.get("name", "")
                            tc_args = func.get("arguments", "")
                        else:
                            tc_id = getattr(tc, "id", "")
                            tc_name = getattr(tc, "function", {}).get("name", "")
                            tc_args = getattr(tc, "function", {}).get("arguments", "")
                        
                        # 确保 input 是字典类型
                        if isinstance(tc_args, str):
                            try:
                                import json
                                tc_args = json.loads(tc_args) if tc_args else {}
                            except json.JSONDecodeError:
                                tc_args = {}
                        
                        blocks.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": tc_args,
                        })
                    
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool result message - should be converted to user message with tool_result content
                tool_use_id = msg.get("tool_use_id") or msg.get("tool_call_id", "unknown")
                tool_content = content or ""
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_content,
                    }]
                })
        
        return anthropic_messages, system_prompt or ""

    def convert_tools_openai(self, tools: list) -> list[dict]:
        """将工具转换为OpenAI格式"""
        if not tools:
            return None
        
        openai_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
            else:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "parameters", {}),
                    },
                })
        return openai_tools

    def convert_tools_anthropic(self, tools: list) -> list[dict]:
        """将工具转换为Anthropic格式
        
        支持两种输入格式:
        1. OpenAI格式: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        2. 直接格式: {"name": ..., "description": ..., "parameters": ...}
        """
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # 检查是否是OpenAI格式 (包含 "function" 键)
                if "function" in tool:
                    func = tool.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    })
                else:
                    # 直接格式
                    anthropic_tools.append({
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {}),
                    })
            else:
                anthropic_tools.append({
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "parameters", {}),
                })
        
        # 过滤掉空工具
        anthropic_tools = [t for t in anthropic_tools if t.get("name")]
        return anthropic_tools if anthropic_tools else None


class MockProvider(LLMProvider):
    """模拟 Provider - 用于测试"""

    def __init__(self, model: str = "mock"):
        self.model = model
        self.name = "mock"

    async def chat(
        self,
        messages: list,
        tools: list = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """模拟调用 - 返回简单的响应"""
        # 获取最后一条用户消息
        last_msg = messages[-1].get("content", "") if messages else ""

        # 检查是否有工具可用
        has_tools = tools and len(tools) > 0

        # 简单判断：如果消息包含特定关键词，模拟调用工具
        tool_keywords = ["检测", "提取", "分析", "处理", "转换", "清洗", "统计", "对齐", "导出"]
        should_use_tool = any(kw in last_msg for kw in tool_keywords)

        if has_tools and should_use_tool and tools:
            # 返回一个工具调用
            return LLMResponse(
                content="我将使用工具来处理您的请求。",
                has_tool_calls=True,
                tool_calls=[
                    ToolCall(
                        id="mock_call_1",
                        name=tools[0]["function"]["name"],
                        arguments={"file_path": "/path/to/data.csv"}
                    )
                ]
            )

        return LLMResponse(
            content=f"我收到了您的请求：{last_msg[:100]}...\n\n这是一个模拟响应。在配置好 LLM API 后，我将能够真正处理您的科学数据请求。",
            has_tool_calls=False
        )

    def get_default_model(self) -> str:
        return self.model

"""SubAgent管理 - 借鉴NanoBot"""
import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable
from pathlib import Path
from datetime import datetime

from loguru import logger

from src.bus.queue import MessageBus
from src.bus.events import InboundMessage
from src.core.prompt_builder import PromptBuilder


@dataclass
class SubAgentTask:
    """子Agent任务"""
    task_id: str
    agent_name: str
    input_data: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    completed_at: Optional[float] = None
    session_key: Optional[str] = None


class TaskPlannerSubagent:
    """Task Planner Subagent - 生成执行计划"""
    
    def __init__(self, provider, workspace: Path, model: str):
        self.provider = provider
        self.workspace = workspace
        self.model = model

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        """Extract possible JSON payloads from free-form text safely."""
        import re

        if not text:
            return []

        candidates: list[str] = []

        # 1) Prefer fenced json blocks
        for m in re.finditer(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE):
            block = (m.group(1) or "").strip()
            if block:
                candidates.append(block)

        # 2) Then generic fenced blocks (in case model forgot json tag)
        for m in re.finditer(r"```\s*([\s\S]*?)```", text):
            block = (m.group(1) or "").strip()
            if block:
                candidates.append(block)

        # 3) Balanced JSON extraction for inline payloads
        pair = {"{": "}", "[": "]"}
        for i, ch in enumerate(text):
            if ch not in pair:
                continue

            stack = [pair[ch]]
            in_string = False
            escape = False

            for j in range(i + 1, len(text)):
                c = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                    continue

                if c == '"':
                    in_string = True
                    continue

                if c in pair:
                    stack.append(pair[c])
                elif c in ("}", "]"):
                    if not stack or c != stack[-1]:
                        break
                    stack.pop()
                    if not stack:
                        block = text[i:j + 1].strip()
                        if block:
                            candidates.append(block)
                        break

        # 4) Last resort: whole text
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        # Dedupe while preserving order
        deduped: list[str] = []
        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        return deduped

    @staticmethod
    def _parse_plan_payload(text: str) -> dict | None:
        """Parse task planner output into normalized dict format."""
        for candidate in TaskPlannerSubagent._extract_json_candidates(text):
            try:
                data = json.loads(candidate)
            except Exception:
                continue

            if isinstance(data, list):
                return {
                    "task_type": "processing",
                    "execution_strategy": "parallel",
                    "pipelines": data,
                    "result_handling": {"mode": "context"},
                }

            if isinstance(data, dict):
                return data

        return None
    
    async def execute(self, user_request: str) -> dict:
        """执行任务规划，返回执行计划"""
        prompt = f"""根据用户请求，生成数据处理执行计划。

用户请求: {user_request}

请输出JSON格式的执行计划:
{{
    "task_type": "processing|integration|data_prep",
    "execution_strategy": "parallel|serial",
    "pipelines": [
        {{
            "pipeline_id": 1,
            "tasks": [
                {{
                    "task_id": 1,
                    "tool": "使用的工具",
                    "inputs": "输入描述",
                    "outputs": "输出描述"
                }}
            ]
        }}
    ],
    "result_handling": {{
        "mode": "file|context",
        "save_format": "json|markdown"
    }}
}}

可用工具: read_file, list_dir, write_file, data processing tools
只输出JSON。"""

        result = await self.provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )

        parsed = self._parse_plan_payload(result.content or "")
        if parsed is not None:
            return parsed

        logger.warning("TaskPlanner parse failed: no valid JSON payload")
        
        return {
            "task_type": "processing",
            "execution_strategy": "parallel",
            "pipelines": [],
            "result_handling": {"mode": "context"}
        }


class ProcessorSubagent:
    """Processor Subagent - 执行单条 pipeline"""
    
    def __init__(self, provider, workspace: Path, model: str, tools: list):
        self.provider = provider
        self.workspace = workspace
        self.model = model
        self.tools = tools
    
    async def execute(self, pipeline: dict) -> dict:
        """执行单条 pipeline"""
        pipeline_id = pipeline.get("pipeline_id", 0)
        tasks = pipeline.get("tasks", [])
        
        logger.info(f"[ProcessorSubagent] Pipeline {pipeline_id}: {len(tasks)} tasks")
        
        results = []
        for task in tasks:
            tool_name = task.get("tool", "")
            inputs = task.get("inputs", "")
            outputs = task.get("outputs", "")
            
            prompt = f"""执行数据处理任务:

工具: {tool_name}
输入: {inputs}
输出: {outputs}

请执行任务并返回结果。"""
            
            result = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a data processor. Execute tasks using appropriate tools."},
                    {"role": "user", "content": prompt}
                ],
                tools=[t.to_schema() for t in self.tools],
                model=self.model,
            )
            
            results.append({
                "task_id": task.get("task_id", 0),
                "tool": tool_name,
                "result": result.content
            })
        
        return {"pipeline_id": pipeline_id, "results": results}


class IntegratorSubagent:
    """Integrator Subagent - 整合处理结果并保存到文件"""
    
    def __init__(self, provider, workspace: Path, model: str, tools: list = None):
        self.provider = provider
        self.workspace = workspace
        self.model = model
        self.tools = tools or []
    
    async def execute(self, processor_results: list, task_spec: dict) -> str:
        """整合多个 processor 的结果并保存到文件"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.workspace / f"integrated_result_{timestamp}.json"
        
        prompt = f"""整合以下处理结果，输出为 JSON 格式并保存到文件。

处理结果:
{json.dumps(processor_results, indent=2, ensure_ascii=False)}

任务规格:
{json.dumps(task_spec, indent=2)}

输出要求:
1. 整合所有处理结果
2. 输出 JSON 格式
3. 文件保存路径: {output_file}

请提供整合后的最终结果（JSON格式）。"""
        
        result = await self.provider.chat(
            messages=[
                {"role": "system", "content": "You are a data integrator. Merge and summarize processing results. Output JSON format."},
                {"role": "user", "content": prompt}
            ],
            tools=[t.to_schema() for t in self.tools] if self.tools else [],
            model=self.model,
        )
        
        if result.has_tool_calls:
            # If there are tool calls, execute them (e.g., write_file)
            for tool_call in result.tool_calls:
                if tool_call.name == "write_file":
                    return f"结果已保存到文件"
        
        # If no tool calls, return the content
        return f"整合结果:\n{result.content or '整合完成'}\n\n结果文件: {output_file}"


class SubagentManager:
    """子Agent管理器 - 支持创建、取消、查询子Agent任务"""

    def __init__(
        self,
        provider=None,
        workspace: Path = None,
        bus: MessageBus = None,
        model: str = None,
        tools: list = None,
        tool_registry=None,
    ):
        self.provider = provider
        self.workspace = workspace or Path(".")
        self.bus = bus or MessageBus()
        self.model = model or "anthropic/claude-opus-4-5"
        self.tools = tools or []
        self.tool_registry = tool_registry
        self._tasks: Dict[str, SubAgentTask] = {}
        self._active_tasks: Dict[str, List[asyncio.Task]] = {}
        self._lock = asyncio.Lock()
        # Per-instance task tracking (must be instance vars, not class vars)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._session_tasks: Dict[str, set] = {}
        self._session_selected_skills: Dict[str, list[str]] = {}

        # Initialize PromptBuilder for subagent prompts
        self.prompt_builder = PromptBuilder(workspace=self.workspace)

    def create_task_planner(self) -> TaskPlannerSubagent:
        """创建 TaskPlanner 子 agent"""
        return TaskPlannerSubagent(
            provider=self.provider,
            workspace=self.workspace,
            model=self.model,
        )

    def create_processor(self, tools: list = None) -> ProcessorSubagent:
        """创建 Processor 子 agent"""
        return ProcessorSubagent(
            provider=self.provider,
            workspace=self.workspace,
            model=self.model,
            tools=tools or self.tools,
        )

    def create_integrator(self, tools: list = None) -> IntegratorSubagent:
        """创建 Integrator 子 agent"""
        return IntegratorSubagent(
            provider=self.provider,
            workspace=self.workspace,
            model=self.model,
            tools=tools or self.tools,
        )

    def set_session_selected_skills(self, session_key: str, selected_skills: list[str] | None) -> None:
        """Store MainAgent-selected skills for subagent inheritance."""
        if not session_key:
            return
        skills = [s for s in (selected_skills or []) if s]
        self._session_selected_skills[session_key] = skills

    async def spawn(
        self,
        task: str,
        label: str = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str = None,
        subagent_type: str = "general",
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, subagent_type, session_key)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict,
        subagent_type: str = "general",
        session_key: str | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            inherited_skills = self._session_selected_skills.get(session_key or "", [])

            # Build subagent prompt using PromptBuilder directly
            if subagent_type == "task_planner":
                system_prompt = self.prompt_builder.build_task_planner_prompt(task, inherited_skills)
            elif subagent_type == "processor":
                system_prompt = self.prompt_builder.build_processor_prompt(task, inherited_skills)
            elif subagent_type == "integrator":
                system_prompt = self.prompt_builder.build_integrator_prompt(task, inherited_skills)
            else:
                system_prompt = self.prompt_builder.build_processor_prompt(task, inherited_skills)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # task_planner only needs a few explore iterations before outputting JSON
            max_iterations = 8 if subagent_type == "task_planner" else 20
            iteration = 0
            final_result = None

            # Tool access policy by subagent type
            _READONLY_TOOL_NAMES = {"list_dir", "read_file", "exec"}
            _NON_SPAWN_TOOL_NAMES = {
                t.name for t in (self.tools or []) if t.name != "spawn"
            }

            def to_dict(tc):
                return {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments)
                    }
                }

            while iteration < max_iterations:
                iteration += 1

                # Tool access policy per subagent type
                if subagent_type == "task_planner":
                    # Explore-only: list_dir, read_file, exec (read-only shell commands)
                    tools_to_use = [
                        t.to_schema() for t in self.tools
                        if t.name in _READONLY_TOOL_NAMES
                    ] if self.tools else []
                elif subagent_type in {"processor", "integrator"}:
                    # Prevent nested planning loops from processor/integrator
                    tools_to_use = [
                        t.to_schema() for t in self.tools
                        if t.name in _NON_SPAWN_TOOL_NAMES
                    ] if self.tools else []
                else:
                    # default: full access
                    tools_to_use = [t.to_schema() for t in self.tools] if self.tools else []

                logger.info("[Subagent:{}][{}] iter={} tools={}", subagent_type, task_id, iteration, len(tools_to_use))

                try:
                    response = await self.provider.chat(
                        messages=messages,
                        tools=tools_to_use,
                        model=self.model,
                    )
                except Exception as e:
                    logger.error("[Subagent:{}][{}] provider error: {}", subagent_type, task_id, e)
                    final_result = f"Error: {str(e)}"
                    break

                if response.has_tool_calls:
                    tool_call_dicts = [to_dict(tc) for tc in response.tool_calls]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Build allowed-name set for this subagent type (enforce at execution time)
                    if subagent_type == "task_planner":
                        allowed_names = _READONLY_TOOL_NAMES
                    elif subagent_type in {"processor", "integrator"}:
                        allowed_names = _NON_SPAWN_TOOL_NAMES
                    else:
                        allowed_names = {t.name for t in self.tools}

                    for tool_call in response.tool_calls:
                        logger.info("[Subagent:{}][{}] tool={} args={}", subagent_type, task_id, tool_call.name, str(tool_call.arguments)[:120])
                        if tool_call.name not in allowed_names:
                            tool_result = (
                                f"Error: tool '{tool_call.name}' is not permitted for "
                                f"{subagent_type} subagent. "
                                f"Allowed tools: {sorted(allowed_names)}"
                            )
                            logger.warning("[Subagent:{}][{}] blocked disallowed tool '{}'", subagent_type, task_id, tool_call.name)
                        elif self.tool_registry:
                            tool_result = await self.tool_registry.execute(
                                tool_call.name,
                                tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                            )
                        else:
                            tool_result = "Tool registry not available"

                        logger.info("[Subagent:{}][{}] tool result: {}", subagent_type, task_id, str(tool_result)[:200])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id or "unknown",
                            "content": str(tool_result),
                        })
                else:
                    final_result = response.content
                    logger.info("[Subagent:{}][{}] final result ({} chars)", subagent_type, task_id, len(final_result or ""))
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok", subagent_type)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error", subagent_type)

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict,
        status: str,
        subagent_type: str = "general",
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        import json
        pipeline_id = None
        if subagent_type == "processor":
            try:
                parsed_task = json.loads(task)
                if isinstance(parsed_task, dict):
                    if "pipeline" in parsed_task and isinstance(parsed_task["pipeline"], dict):
                        pipeline_id = parsed_task["pipeline"].get("pipeline_id")
                    else:
                        pipeline_id = parsed_task.get("pipeline_id")
            except Exception:
                pipeline_id = None

        message_data = {
            "subagent_type": subagent_type,
            "task_id": task_id,
            "label": label,
            "task": task,
            "pipeline_id": pipeline_id,
            "status": status,
            "result": result,
            "origin_channel": origin["channel"],
            "origin_chat_id": origin["chat_id"],
        }
        
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=json.dumps(message_data, ensure_ascii=False),
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced {} result to {}:{}", task_id, subagent_type, origin['channel'], origin['chat_id'])

    # [DEPRECATED] Use PromptBuilder directly
    # def _build_task_planner_prompt(self, user_request: str) -> str:
    #     """Build system prompt for TaskPlanner with user request."""
    #     return self.prompt_builder.build_task_planner_prompt(user_request)

    # [DEPRECATED] Use PromptBuilder directly
    # def _build_subagent_prompt(self, subagent_type: str = "general") -> str:
    #     """Build a focused system prompt for the subagent."""
    #     if subagent_type == "processor":
    #         return self.prompt_builder.build_processor_prompt()
    #     elif subagent_type == "integrator":
    #         return self.prompt_builder.build_integrator_prompt()
    #     else:
    #         return self.prompt_builder.build_processor_prompt()

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    async def create_task(
        self,
        agent_name: str,
        input_data: str,
        session_key: Optional[str] = None,
    ) -> str:
        """创建子Agent任务"""
        async with self._lock:
            task_id = f"subagent-{uuid.uuid4().hex[:8]}"
            task = SubAgentTask(
                task_id=task_id,
                agent_name=agent_name,
                input_data=input_data,
                session_key=session_key,
            )
            self._tasks[task_id] = task
            logger.info(f"Created subagent task: {task_id} for {agent_name}")
            return task_id

    async def run_task(
        self,
        task_id: str,
        agent_factory: Callable,
        on_progress: Optional[Callable[[str, bool], None]] = None,
    ) -> str:
        """运行子Agent任务"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = "running"
        
        try:
            # 创建Agent实例
            agent = agent_factory(task.agent_name)
            
            # 执行任务
            logger.info(f"Running subagent task: {task_id}")
            result = await agent.execute(
                task.input_data,
                on_progress=on_progress,
            )
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now().timestamp()
            logger.info(f"Subagent task completed: {task_id}")
            return result
            
        except asyncio.CancelledError:
            task.status = "cancelled"
            task.error = "Task cancelled"
            raise
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Subagent task failed: {task_id} - {e}")
            raise

    async def cancel_task(self, task_id: str) -> bool:
        """取消子Agent任务"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            if task.status == "running":
                # 取消正在运行的任务
                if task_id in self._active_tasks:
                    for t in self._active_tasks[task_id]:
                        if not t.done():
                            t.cancel()
                task.status = "cancelled"
                task.error = "Cancelled by user"
                return True
            elif task.status == "pending":
                task.status = "cancelled"
                return True
            
            return False

    async def cancel_by_session(self, session_key: str) -> int:
        """取消指定会话的所有任务"""
        cancelled = 0
        async with self._lock:
            for task in self._tasks.values():
                if task.session_key == session_key and task.status == "running":
                    await self.cancel_task(task.task_id)
                    cancelled += 1
        return cancelled

    def get_task(self, task_id: str) -> Optional[SubAgentTask]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def get_tasks_by_session(self, session_key: str) -> List[SubAgentTask]:
        """获取指定会话的所有任务"""
        return [
            t for t in self._tasks.values()
            if t.session_key == session_key
        ]

    def get_active_tasks(self, session_key: Optional[str] = None) -> List[SubAgentTask]:
        """获取活跃任务"""
        if session_key:
            return [
                t for t in self._tasks.values()
                if t.status == "running" and t.session_key == session_key
            ]
        return [t for t in self._tasks.values() if t.status == "running"]

    async def cleanup_completed(self, older_than_seconds: int = 3600) -> int:
        """清理已完成的任务"""
        now = datetime.now().timestamp()
        cleaned = 0
        
        async with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in ("completed", "failed", "cancelled"):
                    if task.completed_at and (now - task.completed_at) > older_than_seconds:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
                cleaned += 1
        
        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total": len(self._tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        
        for task in self._tasks.values():
            stats[task.status] = stats.get(task.status, 0) + 1
        
        return stats



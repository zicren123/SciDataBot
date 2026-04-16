"""MainAgent - Core agent with subagent spawning capability."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from src.bus.queue import MessageBus
from src.bus.events import InboundMessage, OutboundMessage
from src.session.manager import SessionManager, Session
from src.core.subagent import SubagentManager
from src.core.prompt_builder import PromptBuilder


class MainAgent:
    """Core Agent - Main entry point for SciDataBot.
    
    Responsibilities:
    1. Process user requests directly for simple tasks
    2. Identify data processing tasks via intent recognition
    3. Spawn subagents (task_planner, processor, integrator) for complex tasks
    4. Aggregate results and return to user
    
    Architecture: Follows nanobot's single-threaded agent loop pattern.
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        provider,
        workspace: Path,
        model: str = "anthropic/claude-opus-4-5",
        max_iterations: int = 40,
        tool_registry: Any | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry
        
        # Global lock to serialize message processing (nanobot pattern)
        self._processing_lock = asyncio.Lock()

        # Subagent state tracking — keyed by session_key to avoid cross-session corruption
        self._subagent_states: dict[str, dict] = {}
        
        self.bus = MessageBus()
        self.session_manager = SessionManager(workspace)
        self.prompt_builder = PromptBuilder(workspace=workspace)
        
        # Sync templates on first run
        self.prompt_builder.sync_templates()
        
        tools = list(tool_registry._tools.values()) if tool_registry else []
        self.subagent_manager = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=self.bus,
            model=model,
            tools=tools,
            tool_registry=tool_registry,
        )
        
        self.tools = tool_registry
        self._running = False

    def register_tool(self, tool: Any) -> None:
        """Register a tool."""
        if self.tool_registry:
            self.tool_registry.register(tool)

    async def execute(self, user_request: str) -> dict:
        """Execute user request via dispatch (nanobot pattern)."""
        logger.info(f"[MainAgent] Processing: {user_request[:]}...")
        
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=user_request,
        )
        
        response = await self._dispatch(msg)
        
        return {
            "result": response.content if response else "No response",
            "type": "direct",
        }

    async def _dispatch(self, msg: InboundMessage) -> OutboundMessage | None:
        """Process a message under the global lock (nanobot pattern)."""
        async with self._processing_lock:
            try:
                return await self._process_message(msg)
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                )

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """Process a single inbound message."""
        session_key = msg.session_key
        session = self.session_manager.get_or_create(session_key)
        state = self._get_subagent_state(session_key)

        # Resolve selected skills once in MainAgent; subagents inherit this selection directly.
        selected_skills: list[str] = []
        try:
            skill_loader = getattr(self.prompt_builder, "skill_loader", None)
            if skill_loader:
                selected_skills = skill_loader.load_skills_for_request(
                    request_text=msg.content,
                    explicit_names=None,
                    include_always=True,
                )
        except Exception as e:
            logger.warning("[MainAgent] Failed to resolve selected skills: {}", e)

        state["selected_skills"] = selected_skills
        state["user_request"] = msg.content
        state["origin_channel"] = msg.channel
        state["origin_chat_id"] = msg.chat_id

        if self.subagent_manager:
            self.subagent_manager.set_session_selected_skills(session_key, selected_skills)

        # Update SpawnTool origin context so subagent replies return to the right channel
        if self.tools:
            spawn_tool = self.tools._tools.get("spawn")
            if spawn_tool:
                spawn_tool.set_context(channel=msg.channel, chat_id=msg.chat_id)

        # Send immediate acknowledgment so user knows the request was received.
        preview = msg.content[:30] + ("…" if len(msg.content) > 30 else "")
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=f"「{preview}」已收到，处理中……",
            reply_to=msg.metadata.get("message_id"),
        ))

        history = session.get_history(max_messages=40)
        messages = self.prompt_builder.build_messages(
            history=history,
            current_message=msg.content,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        final_content, tools_used, all_msgs = await self._run_agent_loop(messages, session_key)

        # Save only the current user turn and final assistant response.
        # Saving intermediate tool call messages (role=tool) would lose tool_use_id
        # linkage and corrupt future API calls.
        session.add_message("user", msg.content)
        if final_content:
            session.add_message("assistant", final_content)
        self.session_manager.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content or "I've completed processing but have no response to give.",
            reply_to=msg.metadata.get("message_id"),
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        session_key: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop (nanobot pattern)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        dynamic_loaded_skills: set[str] = set()

        while iteration < self.max_iterations:
            iteration += 1

            # Principle 3: load additional skills during ReAct based on evolving context
            try:
                recent_text = "\n\n".join(
                    str(m.get("content", ""))
                    for m in messages[-6:]
                    if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
                )
                dynamic_ctx, new_names = self.prompt_builder.build_dynamic_skills_context(
                    recent_text,
                    dynamic_loaded_skills,
                )
                if dynamic_ctx:
                    messages.append({"role": "system", "content": dynamic_ctx})
                    dynamic_loaded_skills.update(new_names)
                    if session_key and new_names:
                        state = self._get_subagent_state(session_key)
                        merged = list(dict.fromkeys((state.get("selected_skills") or []) + new_names))
                        state["selected_skills"] = merged
                        if self.subagent_manager:
                            self.subagent_manager.set_session_selected_skills(session_key, merged)
            except Exception as e:
                logger.warning("[MainAgent] Dynamic skill load skipped: {}", e)
            
            tool_defs = self.tool_registry.get_definitions() if self.tool_registry else []

            # Use chat (retry is handled internally by MiniMaxProvider)
            response = await self.provider.chat(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                # Add assistant message with tool calls
                messages = self._add_assistant_message(
                    messages,
                    response.content,
                    response.tool_calls,
                )

                # Sequential tool execution (not parallel!)
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    
                    if not self.tool_registry:
                        result = f"Error: No tool registry available"
                    else:
                        result = await self.tool_registry.execute(
                            tool_call.name,
                            tool_call.arguments,
                        )
                    
                    messages = self._add_tool_result(
                        messages,
                        tool_call.id,
                        tool_call.name,
                        result,
                    )
            else:
                # No tool calls - return content
                clean = self._strip_think(response.content)
                
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                
                messages = self._add_assistant_message(messages, clean, [])
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    def _add_assistant_message(
        self,
        messages: list[dict],
        content: str | None,
        tool_calls: list[Any],
    ) -> list[dict]:
        """Add assistant message to messages list."""
        import uuid as uuid_module
        
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            prepared_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id") or f"toolu_{uuid_module.uuid4().hex[:8]}"
                    tc["id"] = tc_id
                    tc_name = tc.get("name") or tc.get("function", {}).get("name", "")
                    tc_args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                else:
                    tc_id = getattr(tc, "id", None) or f"toolu_{uuid_module.uuid4().hex[:8]}"
                    tc.id = tc_id
                    tc_name = getattr(tc, "name", "")
                    tc_args = getattr(tc, "arguments", {})

                prepared_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": json.dumps(tc_args, ensure_ascii=False),
                    }
                })

            msg["tool_calls"] = prepared_calls
        
        messages.append(msg)
        return messages

    def _add_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict]:
        """Add tool result to messages list.
        
        Supports both Anthropic (tool_use_id) and OpenAI (tool_call_id) formats.
        Always stores as tool_call_id internally, provider's convert_messages_* 
        will map to the correct format when sending to API.
        """
        # Ensure tool_call_id is not empty
        final_tool_call_id = tool_call_id if tool_call_id else "unknown"
        
        # Truncate if too long
        if len(result) > self._TOOL_RESULT_MAX_CHARS:
            result = result[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
        
        # Build tool result message - always use tool_call_id internally
        # The provider's convert_messages_* methods will handle the mapping
        tool_result_msg: dict[str, Any] = {
            "role": "tool",
            "content": result,
            "tool_call_id": final_tool_call_id,
        }
        
        messages.append(tool_result_msg)
        return messages

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Strip think blocks from content."""
        if not text:
            return None
        return re.sub(r"<Blocks[\s\S]*?>", "", text).strip() or None

    @staticmethod
    def _extract_skill_paths(text: str | None) -> list[str]:
        """Extract SKILL.md paths from free text."""
        if not text:
            return []
        pattern = r"(?:/[^\s\"']*SKILL\.md|(?:\.|\./|\.\./)?[^\s\"']*SKILL\.md)"
        return re.findall(pattern, text)

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        """Extract possible JSON payloads from model output safely."""
        if not text:
            return []

        candidates: list[str] = []

        # 1) Prefer fenced json blocks
        for m in re.finditer(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE):
            block = (m.group(1) or "").strip()
            if block:
                candidates.append(block)

        # 2) Generic fenced blocks
        for m in re.finditer(r"```\s*([\s\S]*?)```", text):
            block = (m.group(1) or "").strip()
            if block:
                candidates.append(block)

        # 3) Balanced inline JSON extraction
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

        # 4) Last resort: whole output
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        deduped: list[str] = []
        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        return deduped

    @staticmethod
    def _parse_task_planner_plan(result_text: str) -> dict | None:
        """Parse TaskPlanner result into a normalized plan dict."""
        for candidate in MainAgent._extract_json_candidates(result_text):
            try:
                data = json.loads(candidate)
            except Exception:
                continue

            if isinstance(data, list):
                return {"pipelines": data}

            if isinstance(data, dict):
                if isinstance(data.get("pipelines"), list):
                    return data
                nested = data.get("plan")
                if isinstance(nested, dict) and isinstance(nested.get("pipelines"), list):
                    return nested
        return None

    def _get_subagent_state(self, session_key: str) -> dict:
        """Get or create per-session subagent state."""
        if session_key not in self._subagent_states:
            self._subagent_states[session_key] = {
                "task_planner_done": False,
                "plan": None,
                "processor_results": [],
                "expected_processors": 0,
                "processors_done": 0,
                "integrator_done": False,
                "final_result": None,
                "user_request": None,
                "skill_paths": [],
                "selected_skills": [],
                "origin_channel": None,
                "origin_chat_id": None,
            }
        return self._subagent_states[session_key]

    async def _handle_subagent_result(self, msg: InboundMessage) -> None:
        """Handle subagent result message from bus."""
        try:
            data = json.loads(msg.content)
            subagent_type = data.get("subagent_type", "general")
            status = data.get("status", "error")
            result = data.get("result", "")

            origin_channel = data.get("origin_channel", "cli")
            origin_chat_id = data.get("origin_chat_id", "direct")
            session_key = msg.chat_id

            logger.info("[MainAgent] Received {} result, status={}, result_len={}", subagent_type, status, len(result))

            if status == "error":
                logger.error("[MainAgent] Subagent error: {}", result)
                origin = self._get_subagent_state(session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=origin.get("origin_channel") or origin_channel,
                    chat_id=origin.get("origin_chat_id") or origin_chat_id,
                    content=f"Error: {result}",
                ))
                return

            if subagent_type == "task_planner":
                logger.info("[MainAgent] Calling _handle_task_planner_result...")
                await self._handle_task_planner_result(
                    msg,
                    result,
                    session_key,
                    origin_channel,
                    origin_chat_id,
                    data.get("task", ""),
                )
            elif subagent_type == "processor":
                logger.info("[MainAgent] Calling _handle_processor_result...")
                await self._handle_processor_result(msg, result, data.get("pipeline_id"), session_key, origin_channel, origin_chat_id)
            elif subagent_type == "integrator":
                logger.info("[MainAgent] Calling _handle_integrator_result...")
                await self._handle_integrator_result(msg, result, session_key, origin_channel, origin_chat_id)

        except Exception as e:
            logger.error("[MainAgent] Error handling subagent result: {}", e)
    
    async def _handle_task_planner_result(
        self,
        msg: InboundMessage,
        result: str,
        session_key: str,
        origin_channel: str,
        origin_chat_id: str,
        original_task: str = "",
    ) -> None:
        """Handle TaskPlanner result - parse plan and spawn processors."""
        logger.info("[MainAgent] Processing TaskPlanner result...")
        state = self._get_subagent_state(session_key)

        plan = None
        parse_error: str | None = None
        try:
            plan = self._parse_task_planner_plan(result)
            if plan is not None:
                logger.info("[MainAgent] Parsed task plan, pipelines={}", len(plan.get("pipelines", [])))
        except Exception as e:
            parse_error = str(e)
            logger.warning("[MainAgent] Failed to parse plan: {}", e)

        pipelines = plan.get("pipelines", []) if isinstance(plan, dict) else []

        if not pipelines:
            error_payload = {
                "error": {
                    "code": "TASK_PLANNER_PLAN_INVALID",
                    "message": "TaskPlanner 未生成可执行的 pipelines，已终止执行。",
                    "retryable": True,
                    "details": {
                        "parse_error": parse_error,
                        "raw_result_preview": (result or "")[:500],
                    },
                    "suggestion": [
                        "请重试一次当前请求。",
                        "若仍失败，请将任务拆分成更明确的子任务后重试（重规划）。",
                    ],
                }
            }
            logger.warning("[MainAgent] Invalid task plan, ask user to retry/replan")
            await self.bus.publish_outbound(OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=(
                    "任务规划失败，未生成可执行流水线。\n"
                    "请重试，或将任务拆分后重新发起。\n\n"
                    f"{json.dumps(error_payload, ensure_ascii=False, indent=2)}"
                ),
            ))
            self._subagent_states.pop(session_key, None)
            return
        
        skill_paths = self._extract_skill_paths(original_task)
        selected_skills = state.get("selected_skills", [])

        valid_pipelines: list[dict] = []
        for idx, pipeline in enumerate(pipelines, start=1):
            if not isinstance(pipeline, dict):
                logger.warning("[MainAgent] Skip invalid pipeline entry (not dict): {}", pipeline)
                continue
            if "pipeline_id" not in pipeline:
                pipeline = {**pipeline, "pipeline_id": idx}
            valid_pipelines.append(pipeline)

        if not valid_pipelines:
            error_payload = {
                "error": {
                    "code": "TASK_PLANNER_NO_VALID_PIPELINES",
                    "message": "TaskPlanner 返回了 pipelines，但没有可执行的有效 pipeline。",
                    "retryable": True,
                    "details": {
                        "pipelines_count": len(pipelines),
                        "valid_pipelines_count": 0,
                    },
                    "suggestion": [
                        "请重试一次当前请求。",
                        "若仍失败，请让 TaskPlanner 输出严格 JSON，并确保 pipelines 每一项都是对象。",
                    ],
                }
            }
            logger.warning("[MainAgent] No valid pipeline entries after validation")
            await self.bus.publish_outbound(OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=(
                    "任务规划失败，未发现可执行的有效 pipeline。\n"
                    "请重试，或将任务拆分后重新发起。\n\n"
                    f"{json.dumps(error_payload, ensure_ascii=False, indent=2)}"
                ),
            ))
            self._subagent_states.pop(session_key, None)
            return
        # 606-661 可以注释掉，主要是为了在日志中清晰展示 TaskPlanner 输出的 pipelines 内容，方便调试和验证解析逻辑。
        logger.info(
            "[MainAgent] TaskPlanner produced {} pipelines:\n{}",
            len(valid_pipelines),
            json.dumps(valid_pipelines, ensure_ascii=False, indent=2),
        )

        logger.info("[MainAgent] Spawning {} valid Processors...", len(valid_pipelines))

        spawned_pipelines: list[dict] = []
        for pipeline in valid_pipelines:
            pipeline_id = pipeline.get("pipeline_id", 0)
            processor_payload = {
                "pipeline": pipeline,
                "global_context": {
                    "original_user_request": original_task,
                    "skill_paths": skill_paths,
                    "selected_skills": selected_skills,
                },
            }
            try:
                await self.subagent_manager.spawn(
                    task=json.dumps(processor_payload, ensure_ascii=False),
                    label=f"Processor-{pipeline_id}",
                    origin_channel=origin_channel,
                    origin_chat_id=origin_chat_id,
                    session_key=session_key,
                    subagent_type="processor",
                )
                spawned_pipelines.append(pipeline)
            except Exception as e:
                logger.error("[MainAgent] Failed to spawn Processor-{}: {}", pipeline_id, e)

        if not spawned_pipelines:
            error_payload = {
                "error": {
                    "code": "PROCESSOR_SPAWN_FAILED",
                    "message": "未能成功启动任何 Processor，已终止执行。",
                    "retryable": True,
                    "details": {
                        "valid_pipelines_count": len(valid_pipelines),
                        "spawned_pipelines_count": 0,
                    },
                    "suggestion": [
                        "请重试一次当前请求。",
                        "如仍失败，请检查子代理服务状态后重试。",
                    ],
                }
            }
            await self.bus.publish_outbound(OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content=(
                    "任务执行失败，未能启动任何处理流水线。\n"
                    "请重试。\n\n"
                    f"{json.dumps(error_payload, ensure_ascii=False, indent=2)}"
                ),
            ))
            self._subagent_states.pop(session_key, None)
            return

        normalized_plan = dict(plan) if isinstance(plan, dict) else {}
        normalized_plan["pipelines"] = spawned_pipelines

        state["task_planner_done"] = True
        state["plan"] = normalized_plan
        state["expected_processors"] = len(spawned_pipelines)
        state["processor_results"] = []
        state["processors_done"] = 0
        state["integrator_spawned"] = False
        state["origin_channel"] = origin_channel
        state["origin_chat_id"] = origin_chat_id
        state["user_request"] = original_task
        state["skill_paths"] = skill_paths
        state["selected_skills"] = selected_skills

        logger.info(
            "[MainAgent] Processors spawned: {}/{} (expected={})",
            len(spawned_pipelines),
            len(valid_pipelines),
            state["expected_processors"],
        )
        logger.info("[MainAgent] All Processors spawned, waiting for results...")

    async def _handle_processor_result(self, msg: InboundMessage, result: str, pipeline_id: int = None, session_key: str = None, origin_channel: str = None, origin_chat_id: str = None) -> None:
        """Handle Processor result - collect and check if all done."""
        logger.info("[MainAgent] Processor {} completed, result: {}", pipeline_id, result[:300] if result else "empty")
        state = self._get_subagent_state(session_key or msg.chat_id)

        state["processor_results"].append({
            "pipeline_id": pipeline_id,
            "result": result,
        })
        state["processors_done"] += 1

        done = state["processors_done"]
        expected = state["expected_processors"]

        logger.info("[MainAgent] Processor {}/{} completed", done, expected)

        if done >= expected and expected > 0 and not state.get("integrator_spawned"):
            logger.info("[MainAgent] All Processors done, spawning Integrator...")
            state["integrator_spawned"] = True

            await self.subagent_manager.spawn(
                task=json.dumps({
                    "processor_results": state["processor_results"],
                    "plan": state["plan"],
                    "global_context": {
                        "original_user_request": state.get("user_request", ""),
                        "skill_paths": state.get("skill_paths", []),
                        "selected_skills": state.get("selected_skills", []),
                    },
                }, ensure_ascii=False),
                label="Integrator",
                origin_channel=state["origin_channel"],
                origin_chat_id=state["origin_chat_id"],
                session_key=session_key,
                subagent_type="integrator",
            )

    async def _handle_integrator_result(self, msg: InboundMessage, result: str, session_key: str = None, origin_channel: str = None, origin_chat_id: str = None) -> None:
        """Handle Integrator result - send final result to user."""
        logger.info("[MainAgent] Integrator completed")
        state = self._get_subagent_state(session_key or msg.chat_id)

        await self.bus.publish_outbound(OutboundMessage(
            channel=origin_channel or state["origin_channel"],
            chat_id=origin_chat_id or state["origin_chat_id"],
            content=result,
        ))

        self._subagent_states.pop(session_key or msg.chat_id, None)

        logger.info("[MainAgent] Complex task completed, result sent to user")

    # [DEPRECATED] Use PromptBuilder instead
    # def _build_system_prompt(self) -> str:
    #     """Build the system prompt."""
    #     tool_descriptions = []
    #     if self.tool_registry:
    #         for tool in self.tool_registry._tools.values():
    #             tool_descriptions.append(f"- {tool.name}: {tool.description}")
    #     
    #     tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
    #     
    #     return f"""# SciDataBot - Scientific Data Assistant

    # You are SciDataBot, an AI assistant specialized in scientific data processing.

    # ## Available Tools
    # You have access to the following tools. Use them when needed:

    # {tools_text}

    # ## Guidelines
    # - Use available tools to help user requests
    # - For weather queries, use the 'weather' tool
    # - For file operations, use read_file, write_file, list_dir tools
    # - For data processing, use the appropriate data processing tools
    # - Report results clearly
    # - If uncertain, ask for clarification"""

    async def run(self) -> None:
        """Run the agent (consuming from message bus)."""
        self._running = True
        logger.info("MainAgent started")
        
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                logger.info("[MainAgent.run] Received inbound message: channel={}, sender={}", msg.channel, msg.sender_id)
                
                # Check if this is a subagent result message
                if msg.channel == "system" and msg.sender_id == "subagent":
                    logger.info("[MainAgent.run] Handling subagent result...")
                    await self._handle_subagent_result(msg)
                else:
                    response = await self._dispatch(msg)
                    
                    if response:
                        await self.bus.publish_outbound(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("MainAgent error: {}", e)

    def stop(self) -> None:
        """Stop the agent."""
        self._running = False

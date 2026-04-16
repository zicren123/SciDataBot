"""Prompt builder - modular system prompt construction."""

import json
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class PromptBuilder:
    """Builds the system prompt from modular components."""
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    
    def __init__(self, workspace: Path, skill_loader: Any = None):
        self.workspace = workspace
        self.skill_loader = skill_loader
        
        # Try to load skill loader if not provided
        if self.skill_loader is None:
            try:
                from src.skills.manager import SkillLoader
                self.skill_loader = SkillLoader(workspace=workspace)
            except ImportError:
                logger.warning("SkillLoader not available, skills will be disabled")
                self.skill_loader = None
    
    def build_system_prompt(self, skill_names: list[str] | None = None, user_request: str | None = None) -> str:
        """Build the complete system prompt."""
        request_skills = self._get_preloaded_request_skills(user_request, skill_names)
        parts = [
            self._get_identity(),
            self._load_bootstrap_files(),
            self._get_memory_context(),
            self._get_active_skills(),
            request_skills,
            self._build_skills_summary(),
        ]
        # Filter empty parts
        parts = [p for p in parts if p]
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section - dynamic runtime info only."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}"
        
        platform_policy = ""
        if system == "Windows":
            platform_policy = """## Platform Policy (Windows)
- Running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable.
"""
        else:
            platform_policy = """## Platform Policy (POSIX)
- Running on POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
"""
        
        return f"""# Runtime
{runtime}

## Workspace
{workspace_path}
- Memory: memory/MEMORY.md
- History: memory/HISTORY.md
- Skills: skills/<name>/SKILL.md

{platform_policy}

Reply directly for conversation. Use 'message' tool to send to chat channels."""
    
    def _load_bootstrap_files(self) -> str:
        """Load bootstrap files from templates/ or workspace."""
        parts = []
        
        # Check templates directory first (project-relative)
        templates_dir = Path(__file__).parent.parent.parent / "templates"
        
        for filename in self.BOOTSTRAP_FILES:
            # Try workspace first (allows user overrides)
            file_path = self.workspace / filename
            if not file_path.exists():
                # Fall back to templates
                if templates_dir.exists():
                    file_path = templates_dir / filename
            
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    parts.append(f"## {filename}\n\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to load bootstrap file {filename}: {e}")
        
        return "\n\n".join(parts) if parts else ""
    
    def _get_memory_context(self) -> str:
        """Get memory context from workspace."""
        memory_file = self.workspace / "memory" / "MEMORY.md"
        if memory_file.exists():
            try:
                content = memory_file.read_text(encoding="utf-8")
                return f"# Memory\n\n{content}"
            except Exception as e:
                logger.warning(f"Failed to load memory file: {e}")
        return ""
    
    def _get_active_skills(self) -> str:
        """Get skills marked as always=true."""
        if not self.skill_loader:
            return ""
        
        try:
            always_skills = self.skill_loader.get_always_skills()
            if always_skills:
                content = self.skill_loader.load_skills_for_context(always_skills)
                if content:
                    return f"# Active Skills\n\n{content}"
        except Exception as e:
            logger.warning(f"Failed to load active skills: {e}")
        
        return ""
    
    def _build_skills_summary(self) -> str:
        """Build skills summary for progressive loading."""
        if not self.skill_loader:
            return ""
        
        try:
            summary = self.skill_loader.build_skills_summary()
            if summary:
                return f"""# Skills

Builtin skills are preloaded from {self.workspace}/src/skills/builtin.
User skills are loaded only when explicitly referenced by skill name (e.g. "CLIMATE SKILL") or by SKILL.md path.

{summary}"""
        except Exception as e:
            logger.warning(f"Failed to build skills summary: {e}")
        
        return ""
    
    def _get_preloaded_request_skills(
        self,
        user_request: str | None,
        skill_names: list[str] | None = None,
    ) -> str:
        """Preload request-related skills before ReAct (principles 1 & 2)."""
        if not self.skill_loader:
            return ""

        try:
            names = self.skill_loader.load_skills_for_request(
                request_text=user_request or "",
                explicit_names=skill_names,
                include_always=False,
            )
            if not names:
                return ""
            content = self.skill_loader.load_skills_for_context(names)
            if not content:
                return ""
            return f"# Request Skills (Preloaded)\n\n{content}"
        except Exception as e:
            logger.warning(f"Failed to preload request skills: {e}")
            return ""

    def build_dynamic_skills_context(
        self,
        text: str,
        already_loaded: set[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Load additional skills during ReAct based on evolving context (principle 3)."""
        if not self.skill_loader:
            return "", []

        loaded_set = already_loaded or set()
        try:
            candidate_names = self.skill_loader.load_skills_for_request(
                request_text=text or "",
                explicit_names=None,
                include_always=False,
            )
            new_names = [n for n in candidate_names if n not in loaded_set]
            if not new_names:
                return "", []

            content = self.skill_loader.load_skills_for_context(new_names)
            if not content:
                return "", []
            return f"# Dynamic Skills (ReAct)\n\n{content}", new_names
        except Exception as e:
            logger.warning(f"Failed to build dynamic skills context: {e}")
            return "", []

    @staticmethod
    def build_runtime_context(channel: str | None = None, chat_id: str | None = None) -> str:
        """Build runtime metadata block for injection before the user message."""
        import time
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        
        return PromptBuilder.RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build complete message list for LLM call."""
        runtime_ctx = self.build_runtime_context(channel, chat_id)
        
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names, current_message)},
            *history,
            {"role": "user", "content": f"{runtime_ctx}\n\n{current_message}"},
        ]
    
    def sync_templates(self) -> list[str]:
        """Sync template files to workspace if missing. Returns list of synced files."""
        synced = []
        templates_dir = Path(__file__).parent.parent.parent / "templates"
        
        if not templates_dir.exists():
            return synced
        
        for filename in self.BOOTSTRAP_FILES:
            src = templates_dir / filename
            dst = self.workspace / filename
            
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                synced.append(filename)
                logger.info(f"Synced template file: {filename}")
        
        return synced

    # ============== Subagent Prompt Methods ==============

    def get_workspace_info(self) -> str:
        """Get workspace information for subagents."""
        workspace_path = str(self.workspace.expanduser().resolve())
        return f"""## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md"""

    def get_platform_policy(self) -> str:
        """Get platform-specific policy."""
        system = platform.system()
        if system == "Windows":
            return """## Platform Policy (Windows)
- Running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable."""
        else:
            return """## Platform Policy (POSIX)
- Running on POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands."""

    def _load_subagent_template(self, subagent_type: str) -> str:
        """Load subagent template from templates/subagents/ or return empty."""
        templates_dir = Path(__file__).parent.parent.parent / "templates" / "subagents"
        template_file = templates_dir / f"{subagent_type.upper()}.md"
        
        if template_file.exists():
            try:
                return template_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to load subagent template {subagent_type}: {e}")
        return ""

    @staticmethod
    def _extract_skill_paths_from_text(text: str) -> list[str]:
        """Extract SKILL.md paths from free text."""
        if not text:
            return []
        pattern = r"(?:/[^\s\"']*SKILL\.md|(?:\.|\./|\.\./)?[^\s\"']*SKILL\.md)"
        return re.findall(pattern, text)

    @staticmethod
    def _extract_skill_names_from_text(text: str) -> list[str]:
        """Extract names like 'CLIMATE SKILL' from text."""
        if not text:
            return []
        pattern = r"\b([A-Za-z0-9_\-]+)\s+skill\b"
        return re.findall(pattern, text, re.IGNORECASE)

    @staticmethod
    def _dedupe_skill_names(names: list[str] | None) -> list[str]:
        """Dedupe skill names while preserving order."""
        if not names:
            return []
        out: list[str] = []
        seen = set()
        for n in names:
            key = (n or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _extract_inherited_skill_names(
        self,
        task_input: str | None,
        inherited_skill_names: list[str] | None = None,
    ) -> list[str]:
        """Extract selected skills inherited from MainAgent context."""
        result: list[str] = []
        if inherited_skill_names:
            result.extend(inherited_skill_names)

        if task_input:
            try:
                data = json.loads(task_input)
                if isinstance(data, dict):
                    gc = data.get("global_context")
                    if isinstance(gc, dict):
                        selected = gc.get("selected_skills")
                        if isinstance(selected, list):
                            result.extend([str(x) for x in selected])
            except Exception:
                pass

        return self._dedupe_skill_names(result)

    def _build_subagent_skills_context(
        self,
        task_input: str | None = None,
        inherited_skill_names: list[str] | None = None,
    ) -> str:
        """Build reusable skills context for subagent prompts."""
        workspace_path = str(self.workspace.expanduser().resolve())

        lines = [
            "## Skills Context",
            f"- Skill root (workspace): {workspace_path}/skills",
            f"- Builtin skill root: {workspace_path}/src/skills/builtin",
            "- Rule: Subagents MUST inherit skill constraints selected by MainAgent.",
            "- Rule: Do NOT re-select skills by traversing user skill folders.",
        ]

        lines.append("- Rule: Builtin skills are preloaded before planning/execution.")
        lines.append("- Rule: Execute strictly under inherited skill constraints.")

        inherited_names = self._extract_inherited_skill_names(task_input, inherited_skill_names)
        if self.skill_loader:
            try:
                if inherited_names:
                    referenced_content = self.skill_loader.load_skills_for_context(inherited_names)
                    if referenced_content:
                        lines.append("\n## Inherited Skill Content (must follow)")
                        lines.append(referenced_content)
                    else:
                        lines.append("\n## Inherited Skills")
                        lines.append("- " + ", ".join(inherited_names))
                else:
                    lines.append("\n## Inherited Skills")
                    lines.append("- (none provided by MainAgent)")
            except Exception as e:
                logger.warning(f"Failed to preload subagent skill content: {e}")

        return "\n".join(lines)

    def build_task_planner_prompt(self, user_request: str, inherited_skill_names: list[str] | None = None) -> str:
        """Build system prompt for TaskPlanner subagent."""
        template = self._load_subagent_template("task_planner")
        if template:
            rendered = template.replace("{user_request}", user_request).replace("{workspace}", str(self.workspace))
            skills_ctx = self._build_subagent_skills_context(user_request, inherited_skill_names)
            return (
                f"{rendered}\n\n"
                "## Planning Constraint\n"
                "You MUST preserve user-declared SKILL constraints in every pipeline task.\n"
                "If a SKILL.md is referenced, make sure each pipeline/task explicitly requires following that SKILL.\n\n"
                f"{skills_ctx}"
            )
        return ""

    def build_processor_prompt(
        self,
        task_input: str | None = None,
        inherited_skill_names: list[str] | None = None,
    ) -> str:
        """Build system prompt for Processor subagent."""
        template = self._load_subagent_template("processor")
        if template:
            rendered = template.replace("{workspace}", str(self.workspace))
            skills_ctx = self._build_subagent_skills_context(task_input, inherited_skill_names)
            task_ctx = f"\n\n## Original Task Input\n{task_input}" if task_input else ""
            return f"{rendered}\n\n{skills_ctx}{task_ctx}"
        return ""

    def build_integrator_prompt(
        self,
        task_input: str | None = None,
        inherited_skill_names: list[str] | None = None,
    ) -> str:
        """Build system prompt for Integrator subagent."""
        template = self._load_subagent_template("integrator")
        if template:
            rendered = template.replace("{workspace}", str(self.workspace))
            skills_ctx = self._build_subagent_skills_context(task_input, inherited_skill_names)
            task_ctx = f"\n\n## Original Task Input\n{task_input}" if task_input else ""
            return f"{rendered}\n\n{skills_ctx}{task_ctx}"
        return ""

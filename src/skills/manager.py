"""Skill loader - load and manage skills from SKILL.md files"""

import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict

from loguru import logger

from .loader import Skill, SkillMetadata


class SkillLoader:
    """
    Skill loader - manages skill installation and loading

    Skills are loaded from:
    - Built-in skills: ~/.scidatabot/src/skills/builtin/
    - User skills: ~/.scidatabot/skills/
    - External skills: can be installed via CLI
    """

    SKILL_PATH_PATTERN = re.compile(r"(?:/[\w\-./]+SKILL\.md|(?:\.|\./|\.\./)?[\w\-./]+SKILL\.md)", re.IGNORECASE)
    SKILL_NAME_PATTERN = re.compile(r"\b([A-Za-z0-9_\-]+)\s+skill\b", re.IGNORECASE)

    def __init__(self, skills_dir: Optional[Path] = None, workspace: Optional[Path] = None):
        """Initialize skill loader"""
        self.workspace = (workspace or Path.cwd()).expanduser().resolve()

        # User skills directory (workspace-local by default)
        self.skills_dir = (
            skills_dir.expanduser().resolve()
            if skills_dir
            else (self.workspace / "skills")
        )

        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Built-in skills directory (prefer workspace source path)
        self.builtin_dir = self.workspace / "src" / "skills" / "builtin"
        if not self.builtin_dir.exists():
            self.builtin_dir = Path(__file__).parent / "builtin"

        # Loaded skills cache
        self._skills: Dict[str, Skill] = {}
        self._loaded = False

    def load_all(self) -> Dict[str, Skill]:
        """Load all preloaded skills (builtin only)."""
        if self._loaded:
            return self._skills

        self._skills = {}

        # Principle 1: preload built-in skills only.
        if self.builtin_dir.exists():
            self._load_from_dir(self.builtin_dir, is_builtin=True)

        self._loaded = True
        logger.info(f"Preloaded {len(self._skills)} builtin skills")
        return self._skills

    def _load_from_dir(self, skills_path: Path, is_builtin: bool = False):
        """Load skills from a directory"""
        if not skills_path.exists():
            return

        for item in skills_path.iterdir():
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = Skill.from_file(skill_file)
                        skill.path = skill_file
                        self._skills[skill.name] = skill
                        if is_builtin:
                            self._skills[f"builtin:{skill.name}"] = skill
                        logger.debug(f"Loaded skill: {skill.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load skill {item.name}: {e}")

    def _load_single_skill_file(self, skill_file: Path, is_builtin: bool = False) -> Optional[Skill]:
        """Load a single SKILL.md file into cache."""
        if not skill_file.exists() or not skill_file.is_file():
            return None

        try:
            skill = Skill.from_file(skill_file)
            skill.path = skill_file
            self._skills[skill.name] = skill
            if is_builtin:
                self._skills[f"builtin:{skill.name}"] = skill
            return skill
        except Exception as e:
            logger.warning(f"Failed to load skill file {skill_file}: {e}")
            return None

    @staticmethod
    def _normalize_skill_token(name: str) -> str:
        token = (name or "").strip()
        token = re.sub(r"\s+", "_", token)
        token = re.sub(r"_+", "_", token)
        token = token.strip("_")
        return token

    def _resolve_user_skill_path(self, ref: str) -> Optional[Path]:
        """Resolve user skill by explicit path or by name without directory traversal."""
        raw = (ref or "").strip().strip('"\'')
        if not raw:
            return None

        # Explicit path form
        if "SKILL.md" in raw or "/" in raw or raw.startswith("."):
            p = Path(raw)
            if not p.is_absolute():
                p = (self.workspace / p).resolve()
            if p.is_dir():
                p = p / "SKILL.md"
            if p.name != "SKILL.md":
                return None
            try:
                p.relative_to(self.skills_dir.resolve())
            except Exception:
                return None
            return p if p.exists() else None

        # Name form: <name> -> {workspace}/skills/<name>/SKILL.md
        token = self._normalize_skill_token(raw)
        token = re.sub(r"(?i)_?skill$", "", token)
        if not token:
            return None

        candidates = [
            token,
            token.lower(),
            token.replace("-", "_").lower(),
            token.replace("_", "-").lower(),
        ]

        for c in candidates:
            p = (self.skills_dir / c / "SKILL.md").resolve()
            if p.exists() and p.is_file():
                return p
        return None

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name"""
        if not self._loaded:
            self.load_all()

        key = (name or "").strip()

        # Try exact match
        if key in self._skills:
            return self._skills[key]

        # Try with builtin: prefix
        if key.startswith("builtin:"):
            plain = key.replace("builtin:", "", 1)
            if plain in self._skills:
                return self._skills[plain]

        # Try loading user skill lazily by explicit name/path
        user_skill_path = self._resolve_user_skill_path(key)
        if user_skill_path:
            loaded = self._load_single_skill_file(user_skill_path, is_builtin=False)
            if loaded:
                return loaded

        return None

    def list(self) -> List[Skill]:
        """List all available skills"""
        if not self._loaded:
            self.load_all()
        dedup: Dict[str, Skill] = {}
        for s in self._skills.values():
            dedup[s.name] = s
        return list(dedup.values())

    def install(self, skill_path: Path) -> Skill:
        """
        Install a skill from a local directory

        Args:
            skill_path: Path to skill directory containing SKILL.md

        Returns:
            The installed skill
        """
        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"SKILL.md not found in {skill_path}")

        # Load the skill
        skill = Skill.from_file(skill_file)

        # Copy to user skills directory
        dest_dir = self.skills_dir / skill.name
        if dest_dir.exists():
            logger.warning(f"Skill {skill.name} already exists, overwriting")

        # Copy directory
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(skill_path, dest_dir)

        # Reload skills
        self._loaded = False
        self.load_all()

        logger.info(f"Installed skill: {skill.name}")
        return self.get(skill.name)

    def uninstall(self, name: str) -> bool:
        """Uninstall a user skill (not builtin)"""
        skill = self.get(name)
        if not skill:
            logger.warning(f"Skill not found: {name}")
            return False

        # Check if builtin
        if name.startswith("builtin:") or str(skill.path).startswith(str(self.builtin_dir)):
            logger.warning(f"Cannot uninstall builtin skill: {name}")
            return False

        # Remove from filesystem
        skill_dir = skill.path.parent
        if skill_dir.exists():
            shutil.rmtree(skill_dir)

        # Reload
        self._loaded = False
        self.load_all()

        logger.info(f"Uninstalled skill: {name}")
        return True

    def get_skill_prompt(self, name: str) -> Optional[str]:
        """Get the full prompt content for a skill"""
        skill = self.get(name)
        if skill:
            return skill.content
        return None

    def build_skills_summary(self) -> str:
        """生成所有 skills 的摘要（供 Agent 使用）"""
        skills = self.list()
        if not skills:
            return "暂无可用 skills"
        
        lines = []
        for s in skills:
            emoji = s.metadata.emoji or "📦"
            always_mark = " (常用)" if s.metadata.always else ""
            lines.append(f"- {emoji} **{s.name}**: {s.description}{always_mark}")
        
        return "\n".join(lines)

    def get_always_skills(self) -> List[str]:
        """获取始终加载的 skills"""
        result = []
        for s in self.list():
            is_builtin = False
            try:
                is_builtin = str(s.path.resolve()).startswith(str(self.builtin_dir.resolve()))
            except Exception:
                is_builtin = False
            if s.metadata.always and is_builtin:
                result.append(s.name)
        return result

    def load_skills_for_context(self, names: List[str]) -> str:
        """加载指定 skills 的完整内容"""
        parts = []
        for name in names:
            content = self.get_skill_prompt(name)
            if content:
                # 移除 frontmatter
                if content.startswith("---"):
                    parts_content = content.split("---", 2)
                    if len(parts_content) >= 3:
                        content = parts_content[2].strip()
                parts.append(f"### Skill: {name}\n\n{content}")
        
        return "\n\n---\n\n".join(parts) if parts else ""

    def extract_references_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract skill names and skill paths from user text."""
        if not text:
            return {"names": [], "paths": []}

        names = [m.group(1) for m in self.SKILL_NAME_PATTERN.finditer(text)]
        paths = self.SKILL_PATH_PATTERN.findall(text)

        # 去重并保持顺序
        def _dedupe(items: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for it in items:
                k = (it or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(k)
            return out

        return {
            "names": _dedupe(names),
            "paths": _dedupe(paths),
        }

    def load_skills_for_request(
        self,
        request_text: str,
        explicit_names: Optional[List[str]] = None,
        include_always: bool = True,
    ) -> List[str]:
        """Load skills required by the request (builtin + explicitly referenced user skills)."""
        if not self._loaded:
            self.load_all()

        loaded_names: List[str] = []

        # Principle 1: always skills from builtin catalog
        if include_always:
            loaded_names.extend(self.get_always_skills())

        refs = self.extract_references_from_text(request_text or "")

        # Principle 2: explicit path references
        for raw_path in refs["paths"]:
            p = self._resolve_user_skill_path(raw_path)
            if not p:
                continue
            sk = self._load_single_skill_file(p, is_builtin=False)
            if sk:
                loaded_names.append(sk.name)

        # Principle 2: explicit skill names (e.g., CLIMATE SKILL)
        candidate_names = []
        if explicit_names:
            candidate_names.extend(explicit_names)
        candidate_names.extend(refs["names"])

        for n in candidate_names:
            s = self.get(n)
            if s:
                loaded_names.append(s.name)

        # 去重并保持顺序
        deduped: List[str] = []
        seen = set()
        for n in loaded_names:
            if n in seen:
                continue
            seen.add(n)
            deduped.append(n)
        return deduped


# Global skill loader instance
_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get global skill loader instance"""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader

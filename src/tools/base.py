"""Tool base class."""

from dataclasses import dataclass
from typing import Any, List, Optional
from enum import Enum


class ToolCategory(Enum):
    """Tool category."""
    DATA_ACCESS = "data_access"
    DATA_PROCESSING = "data_processing"
    DATA_INTEGRATION = "data_integration"
    GENERAL = "general"


@dataclass
class ToolResult:
    """Tool execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.error:
            self.success = False


@dataclass
class ToolSet:
    """Tool set."""
    name: str
    description: str
    tools: List["Tool"] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

    def add_tool(self, tool: "Tool"):
        """Add a tool."""
        self.tools.append(tool)

    def get_tool(self, name: str) -> Optional["Tool"]:
        """Get a tool."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class Tool:
    """Base class for agent tools."""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    category: str = "general"

    def __init__(self, name: str = None, description: str = None, category: str = None):
        """Initialize tool with optional properties."""
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if category is not None:
            self.category = category

    @property
    def name(self) -> str:
        return getattr(self, '_name', 'tool')
    
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def description(self) -> str:
        return getattr(self, '_description', 'A tool')
    
    @description.setter
    def description(self, value):
        self._description = value

    @property
    def parameters(self) -> dict:
        return getattr(self, '_parameters', {"type": "object", "properties": {}})
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters."""
        return "Not implemented"

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply safe schema-driven casts before validation."""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """Cast an object (dict) according to schema."""
        if not isinstance(obj, dict):
            return obj

        props = schema.get("properties", {})
        result = {}

        for key, value in obj.items():
            if key in props:
                result[key] = self._cast_value(value, props[key])
            else:
                result[key] = value

        return result

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        """Cast a single value according to schema."""
        target_type = schema.get("type")

        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val

        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val

        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val

        if target_type == "string":
            return val if val is None else str(val)

        if target_type == "boolean" and isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes"):
                return True
            if val_lower in ("false", "0", "no"):
                return False
            return val

        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val

        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict) -> bool:
        """Validate parameters."""
        required = self.parameters.get("required", [])
        for key in required:
            if key not in params:
                return False
        return True

    def _validate(self, val: Any, schema: dict, path: str) -> list:
        """Full parameter validation - internal method."""
        errors = []
        t = schema.get("type")
        label = path or "parameter"
        
        if t in self._TYPE_MAP and not isinstance(val, self._TYPE_MAP[t]):
            errors.append(f"{label} should be {t}")
        
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {k}")
            for k, v in val.items():
                if k in props:
                    errors.extend(self._validate(v, props[k], f"{path}.{k}" if path else k))
        
        if t == "array" and "items" in schema:
            for i, item in enumerate(val):
                errors.extend(self._validate(item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"))
        
        return errors

    def validate_params_full(self, params: dict) -> list:
        """Full parameter validation - returns error list."""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict:
        """Convert to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def get_name_variants(self) -> list:
        """Get all name variants."""
        variants = [self.name]
        variants.append("".join(word.capitalize() for word in self.name.split("_")))
        variants.append(f"functions.{self.name}")
        return variants

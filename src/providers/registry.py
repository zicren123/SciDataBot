from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib
import inspect
from .base import LLMProvider


class ProviderMetadata:
    """Provider元数据"""
    def __init__(
        self, 
        name: str,
        display_name: str,
        description: str,
        provider_class: type,
        env_var: str = None,
        default_model: str = None,
        base_url: str = None,
        requires_api_key: bool = True,
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.provider_class = provider_class
        self.env_var = env_var or f"{name.upper()}_API_KEY"
        self.default_model = default_model
        self.base_url = base_url
        self.requires_api_key = requires_api_key


class ProviderRegistry:
    """Provider注册表 - 动态发现和管理providers"""
    
    # 预定义的provider元数据
    PROVIDER_METADATA = {
        "anthropic": ProviderMetadata(
            name="anthropic",
            display_name="Anthropic",
            description="Anthropic Claude API",
            provider_class=None,  # 稍后加载
            env_var="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com",
        ),
        "minimax": ProviderMetadata(
            name="minimax",
            display_name="MiniMax",
            description="MiniMax API",
            provider_class=None,
            env_var="MINIMAX_API_KEY",
            default_model="MiniMax-M2.5",
            base_url="https://api.minimaxi.com/anthropic", # 国际版 MiniMax 请替换为 https://api.minimaxi.com/v1
        ),
        "glm": ProviderMetadata(
            name="glm",
            display_name="ZHIPU AI",
            description="ZHIPU AI GLM API",
            provider_class=None,
            env_var="ZHIPU_API_KEY",
            default_model="glm-4-flash",
            base_url="https://open.bigmodel.cn/api/paas/v4", # 国际版 GLM 请替换为 https://api.z.ai/api/paas/v4
        ),
        "openai": ProviderMetadata(
            name="openai",
            display_name="OpenAI",
            description="OpenAI GPT API",
            provider_class=None,
            env_var="OPENAI_API_KEY",
            default_model="gpt-4o",
            base_url="https://api.openai.com/v1",
        ),
        "deepseek": ProviderMetadata(
            name="deepseek",
            display_name="DeepSeek",
            description="DeepSeek API",
            provider_class=None,
            env_var="DEEPSEEK_API_KEY",
            default_model="deepseek-coder",
            base_url="https://api.deepseek.com/v1",
        ),
        "kimi": ProviderMetadata(
            name="kimi",
            display_name="Moonshot",
            description="Moonshot Kimi API（部分模型温度仅能设置为1）",
            provider_class=None,
            env_var="MOONSHOT_API_KEY",
            default_model="moonshot-v1-128k",
            base_url="https://api.moonshot.cn/v1",
        ),
        "qwen": ProviderMetadata(
            name="qwen",
            display_name="Alibaba",
            description="Alibaba Qwen API",
            provider_class=None,
            env_var="QWEN_API_KEY",
            default_model="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "grok": ProviderMetadata(
            name="grok",
            display_name="xAI",
            description="xAI Grok API",
            provider_class=None,
            env_var="XAI_API_KEY",
            default_model="grok-3",
            base_url="https://api.x.ai/v1",
        ),
        "google": ProviderMetadata(
            name="google",
            display_name="Google",
            description="Google Gemini API",
            provider_class=None,
            env_var="GOOGLE_API_KEY",
            default_model="gemini-2.0-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        ),
        "intern_s1": ProviderMetadata(
            name="intern_s1",
            display_name="Shanghai Ailab",
            description="Shanghai Ailab Intern-S1 API",
            provider_class=None,
            env_var="DATABRICKS_API_KEY",
            default_model="intern-s1",
            base_url="https://chat.intern-ai.org.cn/api/v1",
        ),
        "proxy": ProviderMetadata(
            name="proxy",
            display_name="中转站",
            description="中转站 API (通用 OpenAI 兼容代理提供商)",
            provider_class=None,
            env_var="PROXY_API_KEY",
            default_model="gpt-4o",
            base_url="https://api.example.com/v1",
        ),
    }
    
    def __init__(self):
        """初始化registry"""
        self._providers: Dict[str, ProviderMetadata] = {}
        self._provider_modules: Dict[str, Any] = {}
        self._load_providers()
    
    def _load_providers(self):
        """动态加载所有providers"""
        providers_dir = Path(__file__).parent
        
        for metadata in self.PROVIDER_METADATA.values():
            try:
                # 尝试导入provider模块
                module_name = f"src.providers.{metadata.name}"
                module = importlib.import_module(module_name)
                self._provider_modules[metadata.name] = module
                
                # 查找provider类
                if metadata.provider_class is None:
                    # 尝试自动查找provider类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, LLMProvider) and 
                            obj is not LLMProvider and
                            name.endswith("Provider") and
                            not name.endswith("BaseProvider")):
                            # 优先选择主provider类（不包含特殊后缀如Azure等）
                            if "Azure" not in name and "Anthropic" not in name:
                                metadata.provider_class = obj
                                break
                    
                    # 如果没找到，尝试首字母大写的provider类
                    if metadata.provider_class is None:
                        class_name = f"{metadata.name.title().replace('_', '')}Provider"
                        if hasattr(module, class_name):
                            metadata.provider_class = getattr(module, class_name)
                
                # 尝试获取默认模型
                if metadata.provider_class and metadata.default_model is None:
                    try:
                        instance = metadata.provider_class(api_key="dummy")
                        metadata.default_model = instance.get_default_model()
                    except:
                        pass
                
                self._providers[metadata.name] = metadata
                
            except ImportError as e:
                # 某些providers可能不可用，忽略
                pass
            except Exception as e:
                # 记录加载错误但继续
                pass
    
    def get_provider(self, name: str) -> ProviderMetadata:
        """获取provider元数据"""
        return self._providers.get(name)
    
    def get_all_providers(self) -> Dict[str, ProviderMetadata]:
        """获取所有可用providers"""
        return self._providers
    
    def list_providers_for_tui(self) -> List[Tuple[str, ProviderMetadata]]:
        """为TUI返回providers列表 - 按推荐顺序排序"""
        # 推荐顺序
        preferred_order = ["anthropic", "openai", "google", "grok", "minimax", "glm", "deepseek", "kimi", "qwen", "intern_s1", "proxy"]
        
        result = []
        for name in preferred_order:
            if name in self._providers:
                result.append((name, self._providers[name]))
        
        # 添加任何其他不在preferred_order中的providers
        for name, metadata in self._providers.items():
            if name not in preferred_order:
                result.append((name, metadata))
        
        return result
    
    def get_provider_description(self, name: str) -> str:
        """获取provider的简短描述"""
        metadata = self._providers.get(name)
        if metadata:
            return f"{metadata.display_name} - {metadata.description}"
        return f"{name} (未知)"
    
    def validate_provider_config(self, name: str, config: dict) -> Tuple[bool, str]:
        """验证provider配置"""
        metadata = self._providers.get(name)
        if not metadata:
            return False, f"Provider '{name}' 不存在"
        
        if metadata.requires_api_key:
            if not config.get("api_key"):
                return False, f"{metadata.display_name} 需要 API Key"
        
        if not config.get("model"):
            if metadata.default_model:
                config["model"] = metadata.default_model
            else:
                return False, f"{metadata.display_name} 需要指定模型"
        
        return True, ""


# 全局registry实例
_registry: ProviderRegistry = None

def get_registry() -> ProviderRegistry:
    """获取全局provider registry"""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry

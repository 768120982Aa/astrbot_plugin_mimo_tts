# config.py — 复刻 GPT-SoVITS 的 ConfigNode 模式，精简适配 MiMo
from __future__ import annotations

import base64
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.context import Context
from astrbot.core.star.star_tools import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_path


class ConfigNode:
    """配置节点，把 dict 变成强类型对象。"""

    _SCHEMA_CACHE: dict[type, dict[str, type]] = {}
    _FIELDS_CACHE: dict[type, set[str]] = {}

    @classmethod
    def _schema(cls) -> dict[str, type]:
        return cls._SCHEMA_CACHE.setdefault(cls, get_type_hints(cls))

    @classmethod
    def _fields(cls) -> set[str]:
        return cls._FIELDS_CACHE.setdefault(
            cls,
            {k for k in cls._schema() if not k.startswith("_")},
        )

    @staticmethod
    def _is_optional(tp: type) -> bool:
        if get_origin(tp) in (Union, UnionType):
            return type(None) in get_args(tp)
        return False

    def __init__(self, data: MutableMapping[str, Any]):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_children", {})
        for key, tp in self._schema().items():
            if key.startswith("_"):
                continue
            if key in data:
                continue
            if hasattr(self.__class__, key):
                continue
            if self._is_optional(tp):
                continue
            logger.warning(f"[config:{self.__class__.__name__}] 缺少字段: {key}")

    def __getattr__(self, key: str) -> Any:
        if key in self._fields():
            value = self._data.get(key)
            tp = self._schema().get(key)
            if isinstance(tp, type) and issubclass(tp, ConfigNode):
                children: dict[str, ConfigNode] = self.__dict__["_children"]
                if key not in children:
                    if not isinstance(value, MutableMapping):
                        raise TypeError(
                            f"[config:{self.__class__.__name__}] "
                            f"字段 {key} 期望 dict，实际是 {type(value).__name__}"
                        )
                    children[key] = tp(value)
                return children[key]
            return value
        if key in self.__dict__:
            return self.__dict__[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._fields():
            self._data[key] = value
            return
        object.__setattr__(self, key, value)

    def raw_data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data)

    def save_config(self) -> None:
        if not isinstance(self._data, AstrBotConfig):
            raise RuntimeError(
                f"{self.__class__.__name__}.save_config() 只能在根配置节点上调用"
            )
        self._data.save_config()


# ============ 插件配置节点 ==================

class AutoConfig(ConfigNode):
    only_llm_result: bool
    tts_prob: float
    max_msg_len: int
    style_prompt: str


class ClientConfig(ConfigNode):
    api_key: str
    base_url: str
    timeout: int


class ModelConfig(ConfigNode):
    model: str
    voice: str
    format: str


class CloneConfig(ConfigNode):
    voice_sample_path: str


class DesignConfig(ConfigNode):
    voice_design_prompt: str
    optimize_text_preview: bool = False


class CacheConfig(ConfigNode):
    enabled: bool
    expire_hours: int
    path: str


class ToolConfig(ConfigNode):
    usage_guide: str


# V2.5 系列模型的自然语言指令支持
MODELS_WITH_NL_INSTRUCTION = {
    "mimo-v2.5-tts",
    "mimo-v2.5-tts-voicedesign",
    "mimo-v2.5-tts-voiceclone",
}

# ⚠️ mimo-v2-tts 即将废弃
# 小米官方公告：V2 系列将于 2026 年 6 月 1 日自动路由到 V2.5（含 V2.5 计费），6 月 30 日完全下线。
# 建议尽快迁移到 mimo-v2.5-tts。
DEPRECATED_MODELS = {"mimo-v2-tts"}

# mimo-v2-tts 将于 2026 年 6 月 30 日完全下线，已从可选列表移除
# 详见: https://platform.xiaomimimo.com/docs/updates/deprecate


class PluginConfig(ConfigNode):
    enabled: bool
    auto: AutoConfig
    client: ClientConfig
    model: ModelConfig
    clone: CloneConfig
    design: DesignConfig
    cache: CacheConfig
    tool: ToolConfig

    _plugin_name: str = "astrbot_plugin_mimo_tts"

    def __init__(self, cfg: AstrBotConfig, context: Context):
        super().__init__(cfg)
        self.context = context
        self.data_dir = StarTools.get_data_dir(self._plugin_name)
        self.plugin_dir = Path(get_astrbot_plugin_path()) / self._plugin_name

        # 规范化路径
        self.clone.voice_sample_path = self.normalize_path(self.clone.voice_sample_path)
        self.cache.path = self.normalize_path(self.cache.path)

        # 音频缓存目录
        self.audio_dir = (
            Path(self.cache.path) if self.cache.path else self.data_dir / "audio"
        )
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        # 确保新增字段有默认值（兼容旧配置文件）
        design_data = self._data.setdefault("design", {})
        design_data.setdefault("optimize_text_preview", False)

        self.save_config()

    @staticmethod
    def normalize_path(p: str) -> str:
        if not p:
            return p
        return str(Path(p.strip()).expanduser().resolve())

    @property
    def has_nl_instruction(self) -> bool:
        """当前模型是否支持自然语言风格指令（通过 user message）。"""
        return self.model.model in MODELS_WITH_NL_INSTRUCTION

    def build_voice_param(self) -> str | None:
        """根据当前模型配置构建 voice 参数。

        - voiceclone: 读取音频文件 → Base64 data URI
        - voicedesign: 返回 None（不传 voice）
        - 其他: 直接使用配置的 voice 字符串
        """
        model = self.model.model

        if model == "mimo-v2.5-tts-voiceclone":
            path = self.clone.voice_sample_path
            if path and Path(path).exists():
                ext = Path(path).suffix.lower()
                mime = "audio/mpeg" if ext == ".mp3" else "audio/wav"
                raw = Path(path).read_bytes()
                b64 = base64.b64encode(raw).decode()
                logger.info(f"[MiMo] 已加载参考音频: {path} ({len(raw)} bytes)")
                return f"data:{mime};base64,{b64}"
            # voice 字段如果是 data URI 直接使用
            if self.model.voice.startswith("data:"):
                return self.model.voice
            # 两种都没有 → 不传 voice，让 API 报明确的错，不给用户静默传错值
            logger.error("语音克隆模式未配置参考音频，请在 WebUI 中设置 clone.voice_sample_path")
            return None

        if model == "mimo-v2.5-tts-voicedesign":
            return None  # 此模型不支持 voice 参数

        return self.model.voice or None

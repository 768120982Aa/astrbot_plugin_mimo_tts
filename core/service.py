# service.py — TTS 推理编排：缓存检查 → API调用 → 缓存保存
from __future__ import annotations

import base64

from astrbot.api import logger
from astrbot.core.message.components import Record

from .client import MiMoApiClient, MiMoRequestResult
from .config import PluginConfig
from .local_data import LocalDataManager


class MiMoTTSService:
    def __init__(
        self,
        config: PluginConfig,
        client: MiMoApiClient,
        local_data: LocalDataManager,
    ):
        self.cfg = config
        self.client = client
        self.local_data = local_data

    async def inference(
        self,
        text: str,
        *,
        style_prompt: str = "",
        user_instruction: str = "",
        voice: str | None = None,
        model: str | None = None,
        fmt: str | None = None,
    ) -> MiMoRequestResult:
        """TTS 推理，带缓存。

        Args:
            text: 待合成文本
            style_prompt: 风格提示（V2-tts 用，注入 user 消息）
            user_instruction: 自然语言指令（V2.5 系列用，优先于 style_prompt）
            voice: 覆盖配置的音色
            model: 覆盖配置的模型
            fmt: 覆盖配置的输出格式
        """
        model = model or self.cfg.model.model
        fmt = fmt or self.cfg.model.format
        voice = voice if voice is not None else self.cfg.build_voice_param()

        # 对于 VoiceDesign 模式，优先使用配置的音色描述文本作为 user 指令
        if model == "mimo-v2.5-tts-voicedesign":
            design_prompt = self.cfg.design.voice_design_prompt
            if design_prompt and not user_instruction:
                # 设计提示完全替换 style_prompt（两条路线不叠加）
                user_instruction = design_prompt
                style_prompt = ""

        # 缓存 key
        cache_params = {
            "text": text,
            "model": model,
            "voice": voice or "",
            "format": fmt,
            "style": user_instruction or style_prompt,
        }

        # 查缓存
        cached = self.local_data.get_cached_audio(cache_params)
        if cached:
            cache_path, cached_data = cached
            return MiMoRequestResult(
                ok=True,
                data=cached_data,
                text=text,
                file_path=str(cache_path),
            )

        # 调 API
        logger.debug(f"[MiMo] 推理: model={model}, text_len={len(text)}")
        result = await self.client.tts(
            text,
            model=model,
            voice=voice,
            fmt=fmt,
            style_prompt=style_prompt,
            user_instruction=user_instruction,
        )

        if result and result.data:
            cache_path = self.local_data.save_audio(result.data, cache_params)
            if cache_path:
                result.file_path = str(cache_path)

        return result

    @staticmethod
    def to_record(res: MiMoRequestResult) -> Record:
        """将推理结果转为 AstrBot Record 组件。"""
        if res.file_path:
            try:
                return Record.fromFileSystem(res.file_path)
            except Exception:
                logger.warning(f"[MiMo] 无法读取文件: {res.file_path}")

        if not res.data:
            raise ValueError("无音频数据")

        b64 = base64.urlsafe_b64encode(res.data).decode()
        return Record.fromBase64(b64)

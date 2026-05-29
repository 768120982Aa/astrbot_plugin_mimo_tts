# client.py — MiMo API 客户端（OpenAI 兼容格式，异步）
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from openai import AsyncOpenAI

from .config import PluginConfig


@dataclass
class MiMoRequestResult:
    ok: bool
    data: bytes | None = None
    error: str = ""
    text: str = ""
    file_path: str = ""

    @property
    def size(self) -> int:
        return len(self.data) if self.data else 0

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def __bool__(self) -> bool:
        return self.ok and not self.is_empty


class MiMoApiClient:
    """MiMo TTS API 客户端，使用 OpenAI Async SDK。"""

    def __init__(self, config: PluginConfig):
        self.cfg = config
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.cfg.client.api_key,
                base_url=self.cfg.client.base_url,
                timeout=float(self.cfg.client.timeout),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    @staticmethod
    def _pcm16_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """将裸 PCM16 音频数据包装为 WAV（加上 RIFF 文件头）。"""
        import struct
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(pcm_data)
        header_size = 44
        file_size = data_size + header_size - 8

        wav = bytearray()
        wav.extend(b'RIFF')
        wav.extend(struct.pack('<I', file_size))
        wav.extend(b'WAVE')
        wav.extend(b'fmt ')
        wav.extend(struct.pack('<I', 16))           # chunk size
        wav.extend(struct.pack('<H', 1))             # PCM format
        wav.extend(struct.pack('<H', channels))
        wav.extend(struct.pack('<I', sample_rate))
        wav.extend(struct.pack('<I', byte_rate))
        wav.extend(struct.pack('<H', block_align))
        wav.extend(struct.pack('<H', bits_per_sample))
        wav.extend(b'data')
        wav.extend(struct.pack('<I', data_size))
        wav.extend(pcm_data)
        return bytes(wav)

    async def tts(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        fmt: str | None = None,
        style_prompt: str = "",
        user_instruction: str = "",
    ) -> MiMoRequestResult:
        """调用 MiMo TTS API 合成语音。

        Args:
            text: 待合成文本
            model: 模型 ID，默认取配置
            voice: 音色参数，默认取配置
            fmt: 输出格式 (wav/mp3/pcm16)，默认取配置
            style_prompt: 风格提示，注入 user 消息
            user_instruction: V2.5 系列的自然语言指令（优先级高于 style_prompt）

        Returns:
            MiMoRequestResult
        """
        model = model or self.cfg.model.model
        fmt = fmt or self.cfg.model.format
        if voice is None:
            voice = self.cfg.build_voice_param()

        try:
            client = self._get_client()

            # 构建 messages
            messages: list[dict[str, str]] = []

            # V2.5 系列模型：用 user message 传自然语言指令
            nl_models = {"mimo-v2.5-tts", "mimo-v2.5-tts-voicedesign", "mimo-v2.5-tts-voiceclone"}
            if model in nl_models:
                nl = user_instruction or style_prompt
                if nl:
                    messages.append({"role": "user", "content": nl})
            elif style_prompt:
                messages.append({"role": "user", "content": style_prompt})

            # 待合成文本放在 assistant 消息中（MiMo 强制要求）
            messages.append({"role": "assistant", "content": text})

            # 构建 audio 参数
            audio_params: dict[str, Any] = {"format": fmt}
            if model != "mimo-v2.5-tts-voicedesign" and voice:
                audio_params["voice"] = voice
            # VoiceDesign 模型支持 optimize_text_preview 参数
            if model == "mimo-v2.5-tts-voicedesign" and self.cfg.design.optimize_text_preview:
                audio_params["optimize_text_preview"] = True

            logger.debug(f"[MiMo] 请求: model={model}, voice={voice}, fmt={fmt}, text_len={len(text)}")

            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=["text", "audio"],
                audio=audio_params,
            )

            audio_b64 = resp.choices[0].message.audio.data
            if not audio_b64:
                return MiMoRequestResult(ok=False, error="API 返回空音频数据", text=text)

            audio_bytes = base64.b64decode(audio_b64)

            # PCM16 → WAV 转换（PCM16 是裸音频流，无文件头，大部分平台无法播放）
            if fmt == "pcm16":
                audio_bytes = self._pcm16_to_wav(audio_bytes)
                fmt = "wav"
                logger.debug(f"[MiMo] PCM16 已转换为 WAV: {len(audio_bytes)} bytes")

            logger.debug(f"[MiMo] 合成成功: {len(audio_bytes)} bytes, format={fmt}")
            return MiMoRequestResult(ok=True, data=audio_bytes, text=text)

        except Exception as e:
            logger.error(f"[MiMo] 请求失败: {e}")
            return MiMoRequestResult(ok=False, error=str(e), text=text)

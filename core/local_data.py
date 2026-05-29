# local_data.py — 音频文件缓存，复刻 GPT-SoVITS 的同名模块
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .config import PluginConfig


class LocalDataManager:
    def __init__(self, config: PluginConfig):
        self.cfg = config.cache
        self.expire_seconds = self.cfg.expire_hours * 3600
        self.audio_dir: Path = config.audio_dir

    def _cache_path(self, params: dict[str, Any]) -> Path:
        payload = json.dumps(
            params,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        cache_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        # pcm16 在保存时已转为 wav，所以统一用 wav 后缀
        raw_ext = str(params.get("format", "mp3")).lower()
        if raw_ext == "pcm16":
            ext = "wav"
        else:
            ext = raw_ext if raw_ext in {"wav", "mp3"} else "mp3"
        filename = f"mimo_{cache_hash}.{ext}"
        return (self.audio_dir / filename).resolve()

    def _is_expired(self, file_path: Path) -> bool:
        if self.expire_seconds == 0:
            return False
        age = datetime.now().timestamp() - file_path.stat().st_mtime
        return age > self.expire_seconds

    def get_cached_audio(self, params: dict[str, Any]) -> tuple[Path, bytes] | None:
        if not self.cfg.enabled:
            return None
        try:
            path = self._cache_path(params)
            if not path.exists():
                return None
            if self._is_expired(path):
                path.unlink(missing_ok=True)
                logger.debug(f"[MiMo缓存] 过期删除: {path}")
                return None
            data = path.read_bytes()
            if not data:
                path.unlink(missing_ok=True)
                return None
            logger.debug(f"[MiMo缓存] 命中: {path}")
            return path, data
        except Exception as e:
            logger.warning(f"[MiMo缓存] 读取失败: {e}")
            return None

    def save_audio(
        self,
        data: bytes | None,
        params: dict[str, Any],
        overwrite: bool = True,
    ) -> Path | None:
        if not self.cfg.enabled:
            return None
        if not data:
            logger.error("[MiMo缓存] 保存失败: 无音频数据")
            return None
        try:
            path = self._cache_path(params)
            if path.exists() and not overwrite:
                logger.debug(f"[MiMo缓存] 已存在，跳过: {path}")
                return path
            path.write_bytes(data)
            logger.info(f"[MiMo缓存] 已保存: {path}")
            return path
        except Exception as e:
            logger.error(f"[MiMo缓存] 保存失败: {e}")
            return None

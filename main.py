# main.py — MiMo TTS 插件入口
import random
import time

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Plain, Record
from astrbot.core.platform import AstrMessageEvent

from .core.client import MiMoApiClient
from .core.config import PluginConfig
from .core.local_data import LocalDataManager
from .core.service import MiMoTTSService


@register(
    "astrbot_plugin_mimo_tts",
    "Soulfish",
    "对接小米MiMo API，为AstrBot提供文本转语音(TTS)服务，支持预置音色/音色设计/音色克隆",
    "0.1.2",
    "https://github.com/Soulfish/astrbot_plugin_mimo_tts",
)
class MiMoTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context)
        self.local_data = LocalDataManager(self.cfg)
        self.client = MiMoApiClient(self.cfg)
        self.service = MiMoTTSService(self.cfg, self.client, self.local_data)
        # 去重：记录工具/指令模式最近合成的文本，防止 auto 模式重复调用
        self._last_tts_text: str = ""
        self._last_tts_time: float = 0.0

    async def initialize(self):
        if self.cfg.enabled and not self.cfg.client.api_key:
            logger.warning("[MiMo] API Key 未配置，插件已启用但无法工作")
        elif self.cfg.enabled:
            logger.info(f"[MiMo] 已就绪: model={self.cfg.model.model}, voice={self.cfg.model.voice}")
            logger.info(f"[MiMo] auto 配置: prob={self.cfg.auto.tts_prob}, max_len={self.cfg.auto.max_msg_len}, only_llm={self.cfg.auto.only_llm_result}")

        # 动态拼接工具描述：把用户配置的调用指引注入到已注册工具的 description 中
        if self.cfg.enabled:
            usage_guide = self.cfg.tool.usage_guide
            if usage_guide:
                try:
                    tool_mgr = self.context.get_llm_tool_manager()
                    func_tool = tool_mgr.get_func("mimo_tts")
                    if func_tool:
                        func_tool.description = func_tool.description + "\n\n调用频率指引：" + usage_guide
                        logger.info("[MiMo] 已注入工具调用指引到 mimo_tts 描述")
                except Exception as e:
                    logger.warning(f"[MiMo] 注入工具调用指引失败: {e}")

    async def terminate(self):
        await self.client.close()

    # ==================== 入口一：自动模式 ====================

    @filter.on_decorating_result(priority=15)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """自动拦截 LLM 回复，按概率触发 TTS。"""
        if not self.cfg.enabled:
            return

        cfg = self.cfg.auto
        result = event.get_result()
        if not result:
            return
        chain = result.chain
        if not chain:
            return

        if cfg.only_llm_result and not result.is_llm_result():
            return

        # 收集纯文本（提前到概率判断之前，用于去重）
        plain_texts = []
        for seg in chain:
            if isinstance(seg, Plain):
                plain_texts.append(seg.text)

        if len(plain_texts) != len(chain):
            return  # 含非纯文本组件（图片等），跳过

        combined_text = "\n".join(plain_texts)
        if len(combined_text) > cfg.max_msg_len:
            return

        # 过滤空文本：避免对空消息合成无意义音频
        if not combined_text.strip():
            return

        # 去重：如果最近 15 秒内已被工具/指令模式合成过，跳过所有 auto 触发
        # 15 秒窗口覆盖了工具调用 → LLM 生成回复的完整链路
        time_since_last = time.time() - self._last_tts_time
        if time_since_last < 15:
            logger.debug(f"[MiMo] auto 去重：距离上次工具调用 {time_since_last:.1f}s，跳过")
            return
        else:
            logger.debug(f"[MiMo] auto 通过：距离上次工具调用 {time_since_last:.1f}s")

        if random.random() > cfg.tts_prob:
            return

        logger.info(f"[MiMo] auto 模式触发 TTS，文本长度: {len(combined_text)}")

        # 推理
        res = await self.service.inference(
            combined_text,
            style_prompt=cfg.style_prompt,
        )
        if not res:
            return

        # 语音作为独立消息发送（QQ 等平台文本+语音不能同消息）
        record = self.service.to_record(res)
        await event.send(event.chain_result([record]))

    # ==================== 入口二：手动指令 ====================

    @filter.command("mimo说", alias={"mimo", "MiMo说", "mimott说"})
    async def on_command(self, event: AstrMessageEvent):
        """/mimo说 <内容> — 手动调用 MiMo TTS。"""
        if not self.cfg.enabled:
            return

        text = event.message_str.partition(" ")[2]
        if not text.strip():
            yield event.plain_result("用法: /mimo说 <要合成的内容>")
            return

        res = await self.service.inference(
            text,
            style_prompt=self.cfg.auto.style_prompt,
        )

        if not res:
            yield event.plain_result(f"MiMo TTS 失败: {res.error}")
            return

        # 去重标记：记录本次合成的文本，防止后续 auto 模式重复
        self._last_tts_text = text
        self._last_tts_time = time.time()

        # 先语音，再回文本
        yield event.chain_result([self.service.to_record(res)])
        yield event.chain_result([Plain(text)])

    # ==================== 入口三：LLM 工具 ====================

    @filter.llm_tool()
    async def mimo_tts(
        self,
        event: AstrMessageEvent,
        message: str = "",
        style: str = "",
    ):
        """用 MiMo 语音输出要讲的话。

        Args:
            message(string): 要讲的话。支持在文本中嵌入细粒度控制：
              - 开头加风格标签：(开心)明天就是周五了！
              - 句中插音频标签：(叹气)这么多年来……(苦笑)呵，没如果了。
              - 开头加唱歌标签触发唱歌：(唱歌)原谅我这一生不羁放纵爱自由
              - 复合风格：(慵懒 俏皮)让我再睡五分钟
            style(string): 风格指令，支持三种写法：
              1. 简短描述："慵懒随意，像在跟老朋友聊天"
              2. 复杂描述："语气尖锐刻薄，带着狐假虎威的得意感，语速偏快"
              3. 导演模式（分角色/场景/指导三段）：
                 "角色：疲惫的中年上班族。场景：深夜加班后终于下班。指导：声音沙哑低沉，语速极慢，句间有长停顿。"
              留空使用默认风格。
        """
        try:
            style_prompt = style or self.cfg.auto.style_prompt
            res = await self.service.inference(
                message,
                style_prompt=style_prompt,
                user_instruction=style if style and self.cfg.has_nl_instruction else "",
            )
            if not res:
                return res.error

            seg = self.service.to_record(res)

            # 去重标记：记录本次合成的文本，防止后续 auto 模式重复
            self._last_tts_text = message
            self._last_tts_time = time.time()
            logger.info(f"[MiMo] 工具调用完成，记录时间戳用于去重")

            # 先发语音，再发文本（如果不为空）（分开两条，QQ 等平台不支持同消息含文本+语音）
            await event.send(event.chain_result([seg]))
            if message and message.strip():
                await event.send(event.chain_result([Plain(message)]))
            return "语音已发送，任务完成。"
        except Exception as e:
            return str(e)

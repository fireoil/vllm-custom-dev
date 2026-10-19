# -*- coding: utf-8 -*-
"""
io_processor_plugins 示例：引擎输入/输出的前后处理器。

接口（源码: vllm/plugins/io_processors/interface.py, v0.31.0）:
    class IOProcessor(ABC):
        __init__(self, vllm_config, renderer)   # vLLM 注入引擎配置与渲染器
        parse_data(data) -> 输入                  # 解析 HTTP 请求体里的自定义数据
        merge_sampling_params(params) -> params  # 强制修改采样参数的钩子
        pre_process(prompt, request_id) -> PromptType   # ★ 进引擎前
        post_process(outputs, request_id) -> 自定义输出  # ★ 出引擎后
        （各有一个 async 版本，默认委托给同步版）

加载时机：仅 API 前端进程(process0)。

本示例实现"敏感词过滤"：请求里的违禁词在进引擎前被替换；
（post_process 中保留扩展点，读者可自行加输出过滤）
"""
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Generic

from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor

logger = logging.getLogger(__name__)

BANNED_WORDS = {"secret_code": "【已过滤】"}


class CensorIOProcessor(IOProcessor[dict, str], Generic[dict, str]):
    def parse_data(self, data: object) -> dict:
        """约定请求里 data 字段是 dict（如 {"user_id": ...}），原样透传。"""
        return data if isinstance(data, dict) else {}

    def pre_process(self, prompt: dict, request_id=None, **kwargs):
        """进引擎前：把 prompt 文本中的敏感词替换掉。

        prompt 的实际类型是 vLLM 的 PromptType（字符串或 TokensPrompt 等），
        这里只演示字符串场景。
        """
        if isinstance(prompt, str):
            for word, repl in BANNED_WORDS.items():
                if word in prompt:
                    logger.info("[my_vllm_plugin] censored input for %s", request_id)
                    prompt = prompt.replace(word, repl)
        return prompt

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id=None,
        **kwargs,
    ) -> str:
        """出引擎后：把结果拼成纯文本返回（可在此做输出侧过滤/脱敏）。"""
        return "\n".join(str(o) for o in model_output)


def create_io_processor(vllm_config, renderer) -> IOProcessor:
    """entry point 指向的工厂函数：返回插件实例。"""
    return CensorIOProcessor(vllm_config, renderer)

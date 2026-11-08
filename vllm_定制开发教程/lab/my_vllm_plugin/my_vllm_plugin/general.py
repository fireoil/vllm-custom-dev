# -*- coding: utf-8 -*-
"""
general_plugins 示例：在【所有进程】中执行的注册函数。

调用时机（源码: vllm/plugins/__init__.py → load_general_plugins）:
    - entry point 指向的零参函数会被**直接调用**（func()）
    - 进程0、EngineCore 子进程、每个 Worker 进程都会各调一次
    - 因此插件必须是幂等的：加载多次不能出错（vLLM 明确警告了这一点）

典型用途：
    1. 向 ModelRegistry 注册自定义模型架构（配合教程第10章）
    2. 注册平台（配合 vllm-ascend 这类硬件插件，第12章）
    3. 注册 worker 侧 RPC 方法（collective_rpc）
    4. 环境变量/全局配置预处理
"""
import logging

logger = logging.getLogger(__name__)


def register_banner():
    """entry point 指向这个函数。vLLM 会在每个进程里调用它一次。"""
    # —— 用途演示 1：纯副作用日志（验证插件被加载的最简单方式）——
    logger.info("[my_vllm_plugin] general plugin loaded in this process ✔")

    # —— 用途演示 2：注册自定义模型架构（真实用法骨架，API 以你版本的
    #    vllm/model_executor/models/registry.py 为准）——
    # from vllm.model_executor.models.registry import ModelRegistry
    # ModelRegistry.register_model(
    #     "MyCloneForCausalLM",
    #     "my_vllm_plugin.my_model:MyCloneForCausalLM",
    # )

    # —— 用途演示 3：环境预处理 ——
    import os
    if os.environ.get("MY_PLUGIN_NOTE"):
        logger.info("[my_vllm_plugin] note: %s", os.environ["MY_PLUGIN_NOTE"])

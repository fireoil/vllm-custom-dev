# -*- coding: utf-8 -*-
"""
endpoint_plugins 示例：给 OpenAI API 服务添加自定义 HTTP 路由。

接口（源码: vllm/plugins/endpoint_plugins/interface.py, v0.31.0）:
    Protocol: EndpointPlugin
        name: str                     # VLLM_PLUGINS 白名单用的名字
        required_tasks: tuple | None  # 服务必须支持这些任务才加载；None=不限
        attach_router(app: FastAPI)   # Phase A: build_app 时注册路由(最后挂载, 可遮蔽同名核心路由!)
        async init_state(engine_client, state, args)  # Phase B: 注入引擎句柄

安全设计（vLLM 特意如此）：
    - 其他插件组"默认全加载"，endpoint 插件"必须显式列入 VLLM_PLUGINS 才加载"
      （因为它扩大 HTTP 攻击面）→ 运行本示例必须:
         export VLLM_PLUGINS=my_healthcheck

    - 插件只该做 HTTP 面; 需要动引擎时用 init_state 注入的 engine_client
      (如 collective_rpc), 不要自开引擎访问通道。

运行验证:
    vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 512 &
    curl http://localhost:8000/plugin/ping      # → {"ping": "pong", ...}
"""
import logging

from fastapi import FastAPI
from starlette.datastructures import State

logger = logging.getLogger(__name__)


class HealthCheckPlugin:
    def __init__(self):
        self.name = "my_healthcheck"
        self.required_tasks = None            # 不挑任务, 任何服务都可加载

    def attach_router(self, app: FastAPI) -> None:
        """Phase A: 注册路由。这里加一个 /plugin/ping 探活端点。"""

        @app.get("/plugin/ping")
        async def plugin_ping():
            # self.engine_ready 在 init_state 里被置位, 体现两阶段接线的先后
            return {
                "ping": "pong",
                "plugin": self.name,
                "engine_ready": getattr(self, "engine_ready", False),
            }

    async def init_state(self, engine_client, state: State, args) -> None:
        """Phase B: init_app_state 时被调用, engine_client 已就绪。"""
        logger.info("[my_vllm_plugin] endpoint plugin state initialized")
        self.engine_ready = engine_client is not None


def create_endpoint_plugin():
    """entry point 指向的零参工厂：返回满足 EndpointPlugin 协议的对象。"""
    return HealthCheckPlugin()

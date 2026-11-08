# -*- coding: utf-8 -*-
"""my_vllm_plugin —— 教程第 13 章配套的 vLLM 插件示例包。

安装（在装好 vLLM 的环境里）:
    pip install -e lab/my_vllm_plugin

验证:
    1) general_plugins: 任意方式启动 vLLM, 日志搜 "my_vllm_plugin"
    2) endpoint_plugins:
         export VLLM_PLUGINS=my_healthcheck
         vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 512 &
         curl http://localhost:8000/plugin/ping
"""

# 第 13 章 · vLLM 插件机制实现细节（附完整可运行示例）

> 本章目标：把 vLLM 的插件系统拆到"能自己写一个"的程度。
> 本章所有接口描述均摘自 **vLLM v0.31.0** 源码 `vllm/plugins/` 目录（一手资料），
> 配套完整示例包：`lab/my_vllm_plugin/`（已通过语法与配置验证，装上就能跑）。

---

## 13.0 全景：vLLM 有一张"插件入口点注册表"

vLLM 用 Python 标准的 **entry points 机制**（`importlib.metadata.entry_points`）发现插件。
源码 `vllm/plugins/__init__.py` 定义了 **5 个入口点组**，各有明确的加载进程与时机：

| 入口点组 | 加载的进程 | 时机 | 用途 |
|---|---|---|---|
| `vllm.general_plugins` | **所有进程**（process0 / EngineCore / 每个 Worker） | 进程初始化 | 注册模型/平台/RPC/全局副作用 |
| `vllm.platform_plugins` | 所有进程 | **`current_platform` 初始化时**（且尚未初始化） | 提供自定义硬件平台（昇腾插件走这里） |
| `vllm.io_processor_plugins` | 仅 process0（API 前端） | 引擎构建 | 引擎输入/输出的前后处理 |
| `vllm.endpoint_plugins` | 仅 API 前端进程 | `build_app()` | 给 FastAPI 加自定义 HTTP 路由 |
| `vllm.stat_logger_plugins` | 仅 process0 | async serve 模式 | 统计/日志上报 |

**加载的通用逻辑**（`load_plugins_by_group()`）：

```python
discovered = entry_points(group=group)          # 1. 扫描已安装包的 entry points
for plugin in discovered:
    if allowed is None or plugin.name in allowed:   # 2. VLLM_PLUGINS 白名单过滤
        try:
            func = plugin.load()                # 3. 导入并加载
        except Exception:
            logger.exception(...)               # 4. 失败只记日志, 不炸引擎(容错!)
```

三个工程细节（源码注释原话的提炼）：
1. **白名单**：环境变量 `VLLM_PLUGINS` 控制加载哪些插件（逗号分隔名字）；
   未设置 = 该组全部加载。⚠️ 特例：`VLLM_PLUGINS=""` 会解析成 `[""]`——
   一个谁都匹配不上的允许列表，等于"全禁"，**不是**"未设置"；
2. **幂等要求**：插件会在多个进程重复加载，必须设计成"加载多次不出事"
   （源码原话 WARNING：*plugins can be loaded for multiple times in different processes*）；
3. **失败容忍**：单个插件加载失败只打日志，不影响 vLLM 启动——所以插件坏了
   要去日志里翻 `Failed to load plugin ...`。

## 13.1 general_plugins：最常用的注册通道

entry point 指向一个**零参函数**，vLLM 在每个进程里**直接调用它**。
你的函数体里想干什么都行——这是"注册副作用"的通道：

```python
def register_banner():
    # 用途① 注册自定义模型架构(第10章 L3 的正规军方式, 优于改 registry.py 源文件)
    # 用途② 注册自定义平台
    # 用途③ 注册 worker 侧 RPC 方法(collective_rpc), 与 endpoint 插件配对(见13.3)
    # 用途④ 环境/配置预处理
    ...
```

完整可运行示例：`lab/my_vllm_plugin/my_vllm_plugin/general.py`。

## 13.2 IOProcessor：给引擎装"进水口/出水口滤网"

接口（`vllm/plugins/io_processors/interface.py`，泛型抽象类）：

```python
class IOProcessor(ABC, Generic[In, Out]):
    def __init__(self, vllm_config, renderer): ...   # vLLM 注入配置与渲染器
    def parse_data(self, data) -> In: ...            # 解析请求里的自定义 data 字段
    def merge_sampling_params(self, params): ...     # 钩子: 强制修改采样参数
    @abstractmethod
    def pre_process(self, prompt, request_id) -> PromptType   # ★ 进引擎前
    @abstractmethod
    def post_process(self, outputs, request_id) -> Out        # ★ 出引擎后
    # 每个方法都有 _async 版本, 默认委托同步版; async 版的 post_process
    # 会先把乱序到达的输出按 id 排序再交给你(源码注释: 顺序不保证!)
```

**适用场景**：敏感词过滤、私有协议适配（公司内部网关格式 ↔ vLLM 内部格式）、
请求审计、输出脱敏。示例实现见 `my_vllm_plugin/io_processor.py`（实现了敏感词替换）。

## 13.3 EndpointPlugin：给 API 服务加自定义路由

这是安全设计最讲究的一组（接口 `endpoint_plugins/interface.py`，Protocol 定义）：

```python
class EndpointPlugin(Protocol):
    name: str                                  # VLLM_PLUGINS 白名单里的名字
    required_tasks: tuple[SupportedTask, ...] | None   # 服务须支持这些任务才加载; None=不限
    def attach_router(self, app: FastAPI) -> None       # Phase A: 注册路由
    async def init_state(self, engine_client, state, args)  # Phase B: 引擎句柄注入
```

**三个必须理解的设计决策：**

1. **默认不加载**：其他组是"白名单外全加载"，本组反过来——
   `VLLM_PLUGINS` **未设置时一个都不加载**（因为插件扩大 HTTP 攻击面）。
   未加载却发现了插件时，日志会警告
   `Found endpoint plugin(s) ... but VLLM_PLUGINS is not set`；
2. **两阶段接线**：`attach_router`（build_app 阶段，最后挂载，**可遮蔽同名核心路由**）
   → `init_state`（init_app_state 阶段，此时 `engine_client` 就绪）。
   还有个细节：纯 CPU 渲染服务器没有引擎，`engine_client=None`，插件须自行容错；
3. **只做 HTTP 面**：插件不许自开引擎访问通道；要动引擎就与 general_plugins 配对——
   **general_plugins 安装引擎侧方法，endpoint_plugins 把它暴露成 HTTP**
   （两份注册互相独立，谁也不 implies 谁——源码文档原话）。

完整可运行示例：`lab/my_vllm_plugin/endpoint.py`（`/plugin/ping` 端点，演示两阶段状态）。

## 13.4 platform_plugins：硬件插件的接入口

`current_platform` 单例**首次初始化且尚未就绪时**，vLLM 会加载该组，
插件返回的平台类将**取代内置检测结果**（vllm-ascend 的接法，第 12 章 12.1 已图解）。
写硬件插件 = 实现 `vllm/platforms/interface.py` 中 `Platform` 抽象类的全部抽象方法
（设备类型、显存查询、graph 支持、内核选择策略……第 12 章实验 12-3 让你数过一遍）。

## 13.5 亲手做一个插件：完整五步（示例包已就绪）

```bash
cd lab/my_vllm_plugin
pip install -e .          # ① 安装: entry points 进入系统注册表
                          # ② general: 无需任何环境变量, 启动即加载

# ③ endpoint: 必须显式列入白名单(13.3 的安全设计)
export VLLM_PLUGINS=my_healthcheck
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 512 &

# ④ 验证 HTTP 端点
curl http://localhost:8000/plugin/ping
# → {"ping":"pong","plugin":"my_healthcheck","engine_ready":true}

# ⑤ 验证 general 插件在多进程加载: 启动日志里搜
grep -c "my_vllm_plugin" serve.log    # >1 证明它在多个进程各加载了一次
```

**调试插件的三板斧**：

```bash
export VLLM_LOGGING_LEVEL=DEBUG       # 看 "Available plugins for group ..." 列表
tail -f serve.log | grep -i plugin    # 加载/失败/跳过 全在这里
python3 -c "from importlib.metadata import entry_points; \
    print([ (e.name,e.value) for g in ['vllm.general_plugins','vllm.endpoint_plugins'] for e in entry_points(group=g)])"
                                      # 不启动 vLLM, 直接查注册表
```

## 13.6 插件 vs 改源码：决策指南

| 需求 | 正确姿势 |
|---|---|
| 加 HTTP 端点/审计/过滤 | endpoint / io_processor 插件 |
| 注册新模型、新平台 | general / platform 插件 |
| 改调度策略、KV 管理行为 | 只能改源码（fork + 可编辑安装），目前无插件通道 |
| 临时观察内部（不发布） | monkeypatch（第 06 章 instrument.py） |

> 优先插件的原因：上游升级零冲突。改源码的部分越少，跟着 vLLM 半月一个版本的节奏走越轻松。

## 13.7 动手实验

✅ **实验 13-1（必做）**：按 13.5 的五步，把示例插件跑通，截图三处证据：
`/plugin/ping` 返回 JSON、启动日志的 general 插件加载行、`VLLM_PLUGINS` 去掉后
endpoint 插件的警告行。

✅ **实验 13-2（推荐）**：给 `io_processor.py` 的 `BANNED_WORDS` 加两个词，
重启后验证过滤生效（请求前用 curl 发含敏感词的 prompt）。

🔬 **实验 13-3（硬核）**：把 13.1 注释里的"注册自定义模型"代码解开，
配合第 10 章实验 10-1 的克隆模型，走一遍**不改 vLLM 源码文件**的模型注册
——这才是生产级定制模型的正确发布方式。

## 13.8 本章小结 + 自测

1. 默写 5 个入口点组及各自"在哪个进程、什么时候"加载。
2. `VLLM_PLUGINS=""` 和不设置它，行为差异是什么？endpoint 插件的默认姿态为什么和其他组相反？
3. IOProcessor 的两个抽象方法？它的 async post_process 为什么要先排序？
4. EndpointPlugin 的两阶段接线各发生在什么时机？`engine_client` 什么时候是 None？
5. general_plugins 与 endpoint_plugins 如何配对实现"引擎侧能力 + HTTP 暴露"？

**上一章**：[第 12 章](12_异构硬件部署指南.md) | **下一章**：[附录 · 速查表与常见问题](附录_速查表与常见问题.md)

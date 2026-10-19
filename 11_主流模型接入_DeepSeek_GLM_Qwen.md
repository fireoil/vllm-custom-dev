# 第 11 章 · 主流模型接入：DeepSeek / GLM / Qwen 实现细节

> 本章目标：以国内三大主力模型家族为样本，讲清"一个模型家族在 vLLM 里是怎么被支持的"——
> 涉及注意力变体（GQA/MLA）、MoE、MTP 投机解码、专属量化。
> 学完你能独立为任意一家模型选对启动参数、定位它的源码实现、判断"能不能跑在我的卡上"。
> （文件路径均对照 vLLM **v0.31.0** 源码树逐一核实）

---

## 11.0 三大家族在源码里的"户口本"

vLLM 按模型家族分文件实现（`vllm/model_executor/models/`）。三家的核心文件：

| 家族 | 密集模型 | MoE 模型 | 多模态 | MTP/投机 |
|---|---|---|---|---|
| **Qwen** | `qwen2.py`（Qwen2/2.5）、`qwen3.py` | `qwen2_moe.py`、`qwen3_moe.py`、`qwen3_next.py` | `qwen2_5_vl.py`、`qwen2_vl.py`、`qwen3_vl.py` | `qwen3_next_mtp.py`、`qwen3_5_mtp.py` |
| **GLM** | `glm4.py`（GLM-4 系列）、`chatglm.py`（早期） | `glm4_moe.py`（GLM-4.5/4.6 系）、`glm4_moe_lite.py` | `glm4v.py`、`glm4_1v.py` | `glm4_moe_mtp.py`、`glm4_moe_lite_mtp.py` |
| **DeepSeek** | —（全线 MoE） | `deepseek_v2.py`（**V2/V3/R1 共用此实现**） | `deepseek_vl2.py` | `deepseek_mtp.py`、`deepseek_eagle3.py` |

**怎么知道自己手里的模型用的是哪个实现？** 看 HuggingFace 仓库 `config.json` 的
`architectures` 字段，例如：

| 模型 | config.json 里的 architectures | 对应 vLLM 实现 |
|---|---|---|
| Qwen2.5-7B-Instruct | `Qwen2ForCausalLM` | `qwen2.py` |
| Qwen3-30B-A3B | `Qwen3MoeForCausalLM` | `qwen3_moe.py` |
| GLM-4-9B-Chat | `Glm4ForCausalLM` | `glm4.py` |
| GLM-4.5-Air | `Glm4MoeForCausalLM` | `glm4_moe.py` |
| DeepSeek-R1 | `DeepseekV3ForCausalLM` | `deepseek_v2.py` |

> vLLM 启动时拿 `architectures` 去**模型注册表**（`registry.py`）查实现类——
> 这正是第 10 章"新模型四步接入"的另一半：三大家族都已经注册过了，你只要装对版本。

## 11.1 Qwen 家族：最"标准"的接入，五种形态

Qwen 系是学习 vLLM 模型实现的最佳教材，因为它覆盖了全部主流形态：

1. **Dense（Qwen2/2.5/3）**：标准 GQA 注意力 —— 第 01 章公式直接适用；
2. **MoE（Qwen3-MoE 等）**：FFN 换成路由 + 专家（第 07 章 EP 的应用场景）；
3. **混合架构（Qwen3-Next）**：部分层用**门控 DeltaNet（GDN，线性注意力）**替代标准注意力
   → 对应 vLLM 专门的混合注意力后端（`v1/attention/backends/gdn_attn.py`、`linear_attn.py`），
   KV 显存模式与纯注意力模型完全不同；
4. **多模态 VL**：视觉编码器 + LLM，vLLM 的 multimodal 输入管线（图像会被编码成
   embed 后拼进序列）；
5. **MTP**：官方训练了投机解码用的小模块（见 11.4）。

**实操（你的 8GB 卡能直接跑的）：**

```bash
# Qwen2.5 / Qwen3 小尺寸 dense —— 全功能体验
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048
vllm serve Qwen/Qwen3-1.7B --max-model-len 4096   # Qwen3 默认带思考模式, 见下

# 关闭思考模式（Qwen3 特性, 通过 chat template kwargs 控制）
vllm serve Qwen/Qwen3-1.7B --max-model-len 4096 \
  --chat-template-content-format string
# 请求时: "chat_template_kwargs": {"enable_thinking": false}
```

**Qwen3 特有注意点**：输出里会出现 `<think>...</think>` 段（思考过程）。
API 层可用 `reasoning parser` 分离（`--enable-reasoning --reasoning-parser deepseek_r1`，
兼容 Qwen3 的格式）——这是"接入新模型 = 模型代码 + 服务层特性"两层配合的典型例子。

## 11.2 GLM 家族：从 Dense 到 MoE 的演进

| 模型 | 架构 | vLLM 要点 |
|---|---|---|
| ChatGLM/GLM-2/3 | `chatglm.py` | 老架构，新项目不建议 |
| GLM-4-9B / GLM-4-32B | `glm4.py` | 标准 GQA dense；9B fp16 需 ~19GB，8GB 卡请用 AWQ 版 |
| GLM-4.5 / GLM-4.6 | `glm4_moe.py` | MoE + **官方 MTP 模块**（`glm4_moe_mtp.py`） |
| GLM-4V / 4.1V | `glm4v.py` / `glm4_1v.py` | 视觉版本 |

**实操：**

```bash
# 8GB 卡: GLM-4-9B 走 AWQ 量化(第08章: 能量化先量化)
vllm serve zai-org/glm-4-9b-chat-awq --max-model-len 4096

# 多卡: GLM-4.5-Air(106B MoE) 在 4×A100 级别机器
vllm serve zai-org/GLM-4.5-Air \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"glm4_mtp","num_speculative_tokens":1}'
```

> GLM-4.5/4.6 系的 MTP 以官方提供的 mtp 模块（`glm4_moe_mtp.py`）实现投机解码——
> 用法是在 `--speculative-config` 里指定对应 method。**方法名以你版本的 vLLM 文档为准**
> （`python -c "from vllm.v1.spec_decode import ..."` 或官方 speculative decoding 文档页可查）。

## 11.3 DeepSeek 家族：MLA 是重头戏

### MLA（Multi-head Latent Attention）：KV Cache 的极限压缩

DeepSeek-V2/V3/R1 的核心创新不是 MoE（业界已有），而是 **MLA 注意力**：

**问题**：DeepSeek-V3 是 671B、61 层的大模型，若用传统 GQA，每 token 的 KV 缓存
即使压缩也很可观，多机部署时 KV 显存成为最大瓶颈。

**方案**：把每层的 KV **压缩成一个低维"潜在向量"（latent，512 维）+ 64 维 rope 位置向量**，
只缓存这 576 维；注意力时再从中解压出 K/V。

```
传统 GQA:  缓存 K、V 全量      → KV大小 ∝ 层数 × kv头数 × head_dim × 2
MLA:       缓存 latent(512) + rope(64) = 576 维/层/token
           → 61 层 × 576 × 2B ≈ 70 KB/token (fp16)
           → 对比: 若该模型用普通 MHA 将达到数 MB/token, MLA 压缩一个数量级
```

**vLLM 的落地（v0.31.0）**：`v1/attention/backends/mla/` 目录下有**一整族 MLA 内核**，
按显卡代际/后端自动或手动选择：

| 内核文件 | 适用 |
|---|---|
| `flashmla.py` | DeepSeek 官方 FlashMLA 内核（Hopper 首选之一） |
| `cutlass_mla.py` | CUTLASS 实现 |
| `flashattn_mla.py` / `flashinfer_mla.py` | FlashAttention / FlashInfer 路线 |
| `triton_mla.py`、`aiter_triton_mla.py` | Triton / AMD ROCm 路线 |
| `amx_mla.py`、`cpu_mla.py` | CPU 路线（AMX 指令集） |

> 这就是第 12 章"硬件插件"与模型实现的交汇处：**同一个模型，不同硬件走不同内核**，
> 由 `Platform` 与 `--attention-backend` 类参数仲裁。

### FP8 块级量化（DeepSeek 官方权重自带）

DeepSeek-V3/R1 官方权重就是 **FP8 block-wise（128×128 分块）** 量化格式，
vLLM 原生支持（`model_executor/layers/quantization/fp8.py` 一族），加载即用，
无需再选 `--quantization`（config 里的 `quantization_config` 自动生效）。

### 实操

```bash
# DeepSeek 是 671B 级 MoE, 本教程 8GB 卡无法本地运行, 给生产参考:
# 单机 8×H200/H20 (FP8 权重 + MTP 投机):
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'

# R1 蒸馏版(Qwen/Llama 底座)在本教程机器可跑蒸馏小尺寸:
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 4096
```

**DeepSeek 系列三大招牌特性与 vLLM 支持对照：**

| 特性 | 原理 | vLLM 支持 |
|---|---|---|
| MLA | KV 压缩为 576 维 latent | 专属 MLA 后端族（见上表） |
| MoE + 无辅助损失负载均衡 | 256 专家选 8 | MoE 层 + EP 并行（第 07 章） |
| MTP | 训练时多 token 预测头, 推理时做投机解码 | `deepseek_mtp.py` + speculative-config |

## 11.4 MTP：三大厂殊途同归的"官方投机解码"

对照第 08 章的投机解码原理：草稿模型的质量决定接受率。
**MTP（Multi-Token Prediction）**是"官方草稿"——主模型训练时就多带了一个
"预测下一个 token"的小模块，推理时把它当草稿模型用：

```
普通投机:  独立小模型猜 → 接受率 60~80%, 且要额外下载/维护草稿模型
MTP:       主模型自带的 1 层小模块猜 → 接受率更高, 权重随主模型一起分发
```

三家的 MTP 实现文件（v0.31.0）：`deepseek_mtp.py`、`glm4_moe_mtp.py`、`qwen3_next_mtp.py`。
**共同使用姿势**：`--speculative-config '{"method":"<对应方法>","num_speculative_tokens":N}'`。

**收益观察点**（做实验时看）：日志中的 acceptance rate；代码/翻译类任务收益大，闲聊收益小。

## 11.5 三家横向对比与选型

| 维度 | Qwen | GLM | DeepSeek |
|---|---|---|---|
| 注意力 | GQA（主流）/GDN（Next 系） | GQA | **MLA** |
| MoE | Qwen3-MoE / Next | GLM-4.5/4.6 系 | 全线（V3 起） |
| 官方投机 | MTP（Next/3.5 系） | MTP | MTP |
| 量化生态 | AWQ/GPTQ/FP8 全套 | AWQ 为主 | 官方 FP8 block |
| 小显存入门型号 | Qwen2.5-0.5B、Qwen3-1.7B | glm-4-9b-chat-awq | R1-Distill-Qwen-1.5B |

**选型速断**：小显存/端侧 → Qwen dense 系；企业级中尺寸 → GLM-4 dense 或 Qwen3；
超大参数池化成本 → DeepSeek / 各家 MoE 大杯。

## 11.6 动手实验

✅ **实验 11-1（必做）**：查模型注册表，亲眼确认三家的"户口"：

```bash
python3 -c "
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
for kw in ['Qwen', 'Glm', 'Deepseek']:
    hits = [a for a in archs if kw.lower() in a.lower()]
    print(kw, '→', len(hits), '个架构')
    print('  ', hits[:8])
"
```

✅ **实验 11-2（8GB 卡真机）**：跑 Qwen3，观察思考模式：

```bash
vllm serve Qwen/Qwen3-1.7B --max-model-len 4096
# 另开终端, 请求体加 "chat_template_kwargs": {"enable_thinking": true}
# 对比开/关思考: 输出中 <think> 段的有无、TTFT 差异(思考=更长的 prefill!)
```

✅ **实验 11-3（纸上推演）**：手算 MLA 显存收益。
DeepSeek-V3：61 层、MLA latent 512+rope 64、fp16。算每 token KV；
再假设它用 8 个 KV 头 GQA（head_dim 128），算同样指标，对比倍数。
（这一题做完，你就真正"拥有"了 11.3 的内容。）

🔬 **实验 11-4（选做，源码走读）**：对照第 10 章四步法精读 `qwen2.py`
（仅数百行），画出它的模块组成（Embedding → 28 层 DecoderLayer(Attention+MLP) → lm_head），
再打开 `glm4.py` diff 一下——你会发现 90% 相同。**模型实现之间确实是互相"抄"的。**

## 11.7 本章小结 + 自测

1. vLLM 靠 config.json 的哪个字段路由到模型实现？三家的 MoE 模型架构名分别是什么？
2. MLA 缓存的是什么？大概把 KV 压缩到传统方案的几分之一（量级）？
3. MTP 与普通投机解码的差异？为什么接受率更高？
4. Qwen3-Next 的混合注意力对 KV Cache 管理提出了什么新挑战？
5. 你的 8GB 卡上，三家各能跑哪个型号？

**上一章**：[第 10 章](10_定制开发_接入自己的模型.md) | **下一章**：[第 12 章 · 异构硬件部署指南](12_异构硬件部署指南.md)

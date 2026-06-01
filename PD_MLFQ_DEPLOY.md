# vLLM PD 分离 + Decode 端 MLFQ 部署说明

本文档用于部署当前修改版 vLLM：P/D 分离推理，并在 Decode 端启用 `mlfq` 调度策略。

## 1. 推荐服务器配置

推荐使用阿里云 GPU ECS，配置优先级如下：

| 用途 | 推荐配置 |
| --- | --- |
| 最小功能验证 | 2 张 NVIDIA T4 / A10 |
| 推荐实验配置 | 2 张 NVIDIA A10 24GB |
| 更稳定实验配置 | 2 张 A100 / L20 / L40S |
| 系统 | Ubuntu 22.04 或 Ubuntu 24.04 |
| 磁盘 | 100GB 以上 |
| 内存 | 32GB 以上，推荐 64GB |

安全组建议开放：

| 端口 | 用途 |
| --- | --- |
| 22 | SSH 登录 |
| 8000 | 对外 proxy 服务 |
| 8100 | Prefill 端 vLLM 服务，建议只开放内网 |
| 8200 | Decode 端 vLLM 服务，建议只开放内网 |
| 30001 | PD service discovery，建议只开放内网 |

如果只是验证代码逻辑，推荐模型：

```bash
Qwen/Qwen2.5-1.5B-Instruct
```

该模型显存占用较低，中文能力较好，适合调度实验。

## 2. 基础环境安装

登录服务器：

```bash
ssh root@你的服务器公网IP
```

安装系统依赖：

```bash
apt update
apt install -y git curl wget vim build-essential python3 python3-venv python3-pip
```

检查 GPU：

```bash
nvidia-smi
```

如果没有 `nvidia-smi`，说明 NVIDIA 驱动或 GPU 实例环境还没有配置好。

## 3. 上传并安装当前修改版 vLLM

在本地机器打包：

```bash
cd /home/lz/桌面
tar czf vllm-mlfq.tar.gz vllm
scp vllm-mlfq.tar.gz root@你的服务器公网IP:/root/
```

在服务器解压：

```bash
cd /root
tar xzf vllm-mlfq.tar.gz
cd /root/vllm
```

创建 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

从源码安装当前修改版 vLLM：

```bash
pip install -e .
```

安装 proxy 和测试依赖：

```bash
pip install quart aiohttp openai tblib
```

确认 `mlfq` 参数已经接入：

```bash
vllm serve --help | grep scheduling-policy
```

## 4. 推荐启动方式：P 端默认调度，D 端 MLFQ

推荐只在 Decode 端开启 MLFQ，这样实验目标更明确：

- Prefill 端：默认 `fcfs`
- Decode 端：`--scheduling-policy mlfq`

进入示例目录：

```bash
cd /root/vllm/examples/disaggregated
source /root/vllm/.venv/bin/activate
```

设置环境变量：

```bash
export PYTHONPATH=/root/vllm
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export HF_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

## 5. 手动启动命令

### 5.1 启动 Prefill 端

开第一个终端：

```bash
cd /root/vllm/examples/disaggregated
source /root/vllm/.venv/bin/activate

export PYTHONPATH=/root/vllm
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8100 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8100","send_type":"PUT_ASYNC"}}'
```

### 5.2 启动 Decode 端，开启 MLFQ

开第二个终端：

```bash
cd /root/vllm/examples/disaggregated
source /root/vllm/.venv/bin/activate

export PYTHONPATH=/root/vllm
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8200 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --scheduling-policy mlfq \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8200","send_type":"PUT_ASYNC"}}'
```

### 5.3 启动 proxy

开第三个终端：

```bash
cd /root/vllm/examples/disaggregated
source /root/vllm/.venv/bin/activate

export PYTHONPATH=/root/vllm
export VLLM_HOST_IP=127.0.0.1

python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py
```

proxy 默认监听：

```text
http://127.0.0.1:8000
```

## 6. 发送测试请求

开第四个终端：

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "请简单介绍一下杭州。",
    "max_tokens": 64,
    "temperature": 0
  }'
```

如果从本地电脑访问阿里云服务器，需要把 `127.0.0.1` 换成服务器公网 IP：

```bash
curl -X POST http://你的服务器公网IP:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "请简单介绍一下杭州。",
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 7. 用脚本启动

也可以直接修改：

```bash
/root/vllm/examples/disaggregated/disaggregated_prefill.sh
```

找到 Decode 端启动命令，在参数中加入：

```bash
--scheduling-policy mlfq \
```

然后运行：

```bash
cd /root/vllm/examples/disaggregated
source /root/vllm/.venv/bin/activate

export PYTHONPATH=/root/vllm
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export HF_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

bash disaggregated_prefill.sh
```

## 8. 验证 MLFQ 是否生效

确认 Decode 端启动命令中包含：

```bash
--scheduling-policy mlfq
```

如果启动时报错：

```text
Unknown scheduling policy: mlfq
```

说明运行的不是当前修改版源码。重新安装：

```bash
cd /root/vllm
source .venv/bin/activate
pip install -e .
```

如果需要观察 MLFQ 的入队、降级、提升行为，可以在以下位置加日志：

```text
vllm/v1/core/sched/scheduler.py
```

重点函数：

```text
_assign_mlfq_level_on_enqueue
_update_mlfq_after_schedule
_promote_starved_mlfq_requests
```

建议打印：

```text
request_id
mlfq_level
mlfq_tokens_in_level
next_iter_tokens
mlfq_starve_time
```

## 9. 常见问题

### 9.1 只有一张 GPU 能不能跑？

当前推荐命令需要两张 GPU：

```text
GPU 0: Prefill
GPU 1: Decode
```

如果只有一张 GPU，不建议跑这个 PD 分离脚本。可以只跑单机 vLLM 并启用：

```bash
--scheduling-policy mlfq
```

但这不是严格的 P/D 分离实验。

### 9.2 模型太大导致 OOM

优先降低：

```bash
--max-model-len 1024
--gpu-memory-utilization 0.7
```

也可以换更小模型：

```bash
facebook/opt-125m
```

### 9.3 pytest 缺依赖

如果运行测试时报：

```text
ModuleNotFoundError: No module named 'tblib'
```

安装：

```bash
pip install tblib
```

然后测试：

```bash
cd /root/vllm
source .venv/bin/activate
python3 -m pytest tests/v1/core/test_scheduler.py -k 'mlfq' -q
```

## 10. 推荐实验启动配置

最终推荐配置如下：

```text
模型：Qwen/Qwen2.5-1.5B-Instruct
Prefill GPU：CUDA_VISIBLE_DEVICES=0
Decode GPU：CUDA_VISIBLE_DEVICES=1
Prefill 调度：默认 fcfs
Decode 调度：mlfq
max_model_len：2048，OOM 时改 1024
对外请求端口：8000
```

关键命令是 Decode 端这一行：

```bash
--scheduling-policy mlfq
```

PD 分离由这一段决定：

```bash
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer",...}'
```


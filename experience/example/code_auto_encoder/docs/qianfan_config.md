# Experience LLM Configuration Guide

## Configuration File

Create `~/.experience.json` to configure LLM APIs for different methods.

### Qianfan (千帆) Configuration

```json
{
  "raw_llm_api": {
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_key": "<KEY>",
    "model": "deepseek-v3.2"
  },
  "coding_agent": {
    "base_url": "https://qianfan.baidubce.com/anthropic",
    "api_key": "<KEY>",
    "model": "deepseek-v3.2"
  },
  "tmux_cc": {
    "base_url": "https://qianfan.baidubce.com/anthropic",
    "api_key": "<KEY>",
    "model": "deepseek-v3.2"
  }
}
```

### Internal Comate Configuration

```json
{
  "raw_llm_api": {
    "base_url": "https://oneapi-comate.baidu-int.com/v1",
    "api_key": "<KEY>",
    "model": "GLM-5",
    "username": "yuhaitao01"
  },
  "coding_agent": {
    "base_url": "https://oneapi-comate.baidu-int.com",
    "api_key": "<KEY>",
    "model": "GLM-5"
  },
  "tmux_cc": {
    "base_url": "https://oneapi-comate.baidu-int.com",
    "model": "GLM-5"
  }
}
```

## tmux_cc Configuration Details

`tmux_cc` uses ducc (Baidu Claude Code) which has special authentication handling:

| API Type | base_url | api_key | Authentication |
|----------|----------|---------|----------------|
| **Comate (internal)** | `https://oneapi-comate.baidu-int.com` | Not needed | Uses ducc's `settings.json` token |
| **Qianfan (external)** | `https://qianfan.baidubce.com/anthropic` | Required (`bce-v3/...`) | Uses `api_key` from config |

**How it works:**
- If `base_url` contains "comate" or "baidu-int" → Internal API, uses ducc's built-in token from `settings.json`
- Otherwise → External API, uses `api_key` from `~/.experience.json`

## Configuration Fields

| Field | Description | Used By |
|-------|-------------|---------|
| `base_url` | API endpoint URL | All methods |
| `api_key` | API authentication key | All methods (optional for tmux_cc with comate) |
| `model` | Model name to use | All methods |

## Qianfan Endpoints

| API Type | Endpoint |
|----------|----------|
| OpenAI Compatible | `https://qianfan.baidubce.com/v2` |
| Anthropic Compatible | `https://qianfan.baidubce.com/anthropic` |

## Available Models

| Model | Platform | Description |
|-------|----------|-------------|
| `deepseek-v3.2` | Qianfan | Recommended for coding tasks |
| `ernie-5.0` | Qianfan/Comate | Baidu ERNIE |
| `MiniMax-M2.5` | Comate | Fast responses |
| `kimi-k2.5` | Qianfan | Fast responses |

## Usage

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --num-iterations 1 \
    --total-batch-size 1 \
    --seed 1 \
    --experiment-dir ./my_workspace
```

## Logging

Console output shows the configuration being used:

```
[Config] tmux_cc: base_url=https://oneapi-comate.baidu-int.com, model=MiniMax-M2.5
```

The same info is logged to `run.log` for later reference.

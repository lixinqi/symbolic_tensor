# 安装：pip3 install openai
import asyncio
import json
import os
import re
from typing import Dict, Optional
from openai import AsyncOpenAI  # 导入异步客户端


DEFAULT_LLM_ENV = {
    "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
    "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", ""),
    "LLM_MODEL": os.environ.get("LLM_MODEL", ""),
}


def _get_custom_headers() -> Dict[str, str]:
    """Get custom headers required for internal API authentication."""
    username = os.environ.get("USER", os.environ.get("LLM_USERNAME", "unknown"))
    custom_header_value = json.dumps({
        "agentId": f"raw_llm_api:user:{username}",
        "username": username,
        "repo": "",
        "source": "raw_llm_api",
    })
    return {"comate_custom_header": custom_header_value}


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks from LLM output.

    Some models (e.g., DeepSeek, MiniMax) output thinking process in <think> tags.
    """
    # Remove <think>...</think> blocks (including multiline)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # Also handle unclosed <think> tag at the start
    content = re.sub(r'^<think>.*', '', content, flags=re.DOTALL)
    return content.strip()


async def raw_llm_query(prompt: str, llm_env: Optional[Dict[str, str]] = None):
    env = llm_env if llm_env is not None else DEFAULT_LLM_ENV

    # Get base_url and ensure it ends with /v1
    base_url = env.get('LLM_BASE_URL') or os.environ.get('LLM_BASE_URL')
    if base_url and not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'

    # 每次调用创建客户端，避免跨 asyncio.run() 复用已关闭的连接池
    client = AsyncOpenAI(
        api_key=env.get('LLM_API_KEY') or os.environ.get('LLM_API_KEY'),
        base_url=base_url,
        default_headers=_get_custom_headers(),
    )
    # 发送异步请求
    try:
        response = await client.chat.completions.create(
            model=env.get('LLM_MODEL') or os.environ.get('LLM_MODEL'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Output raw content only, no thinking process."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        # Handle different response formats
        if isinstance(response, str):
            content = response
        else:
            content = response.choices[0].message.content
        # Strip <think> tags from output
        return _strip_think_tags(content)
    finally:
        await client.close()

if __name__ == "__main__":
    ret = asyncio.run(raw_llm_query("Hello"))
    print(ret)
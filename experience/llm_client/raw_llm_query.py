"""Raw LLM API query interface.

This module provides a clean interface for querying OpenAI-compatible LLM APIs,
decoupled from configuration details through AgentConfig.
"""

import asyncio
import json
import os
import re
import sys
from typing import Optional
from openai import AsyncOpenAI, RateLimitError
from experience.llm_client.agent_config import RawLlmConfig
from experience.llm_client.agent_config_factory import AgentConfigFactory

_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BASE_DELAY = 10.0


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks from LLM output.

    Some models (e.g., DeepSeek, MiniMax) output thinking process in <think> tags.
    """
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'^<think>.*', '', content, flags=re.DOTALL)
    return content.strip()


async def raw_llm_query(
    prompt: str,
    config: Optional[RawLlmConfig] = None,
):
    """Query OpenAI-compatible LLM API with the given prompt.
    
    This function is completely decoupled from configuration details.
    
    Args:
        prompt: The prompt to send to the LLM.
        config: Optional RawLlmConfig. If None, creates default config from environment/~/.experience.json.
    
    Returns:
        The LLM response text.
    """
    # Create config if not provided
    if config is None:
        config = AgentConfigFactory.create_raw_llm_config()
    
    # Get custom headers for authentication
    custom_headers = {}
    if config.username:
        custom_header_value = json.dumps({
            "agentId": f"raw_llm_api:user:{config.username}",
            "username": config.username,
            "repo": "",
            "source": "raw_llm_api",
        })
        custom_headers["comate_custom_header"] = custom_header_value
        print(f"[DEBUG] Using username: {config.username}", file=sys.stderr)
    
    # Create OpenAI client
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=custom_headers,
    )
    
    try:
        for attempt in range(_RATE_LIMIT_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Output raw content only, no thinking process."},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )
                break
            except RateLimitError:
                if attempt == _RATE_LIMIT_RETRIES - 1:
                    raise
                delay = _RATE_LIMIT_BASE_DELAY * (2 ** attempt)
                print(f"[raw_llm_query] 429 rate limit, retry {attempt+1}/{_RATE_LIMIT_RETRIES-1} in {delay:.0f}s", file=sys.stderr)
                await asyncio.sleep(delay)
        
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

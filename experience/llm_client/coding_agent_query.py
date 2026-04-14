import asyncio
import glob
import os
import sys
from typing import Dict, Optional
from claude_agent_sdk import query, ClaudeAgentOptions
from experience.llm_client.config import get_config


def _find_ducc_cli_path() -> str:
    """Find ducc CLI binary path."""
    # 1. Check environment variable
    if env_bin := os.environ.get("TMUX_CC_BIN"):
        return env_bin

    # 2. Glob pattern for comate extension (preferred - supports --setting-sources)
    pattern = os.path.expanduser(
        "~/.comate/extensions/baidu.baidu-cc-*/resources/native-binary/bin/ducc"
    )
    matches = sorted(glob.glob(pattern), reverse=True)
    if matches:
        return matches[0]

    # 3. Check PATH (fallback - may be old version)
    import shutil
    if path_bin := shutil.which("ducc"):
        return path_bin

    return None


def _find_ducc_settings_path() -> str:
    """Find ducc settings.json path."""
    pattern = os.path.expanduser(
        "~/.comate/extensions/baidu.baidu-cc-*/resources/settings.json"
    )
    matches = sorted(glob.glob(pattern), reverse=True)
    if matches:
        return matches[0]
    return None


async def coding_agent_query(
    prompt: str,
    cwd: str | None = None,
    allowed_tools: list[str] = None,
    permission_mode: str = "acceptEdits",
    llm_env: Optional[Dict[str, str]] = None,
    model: str = None,
):
    if allowed_tools is None:
        allowed_tools = ["Read", "Edit"]
    env_backup = {}
    if llm_env is not None:
        for key, val in llm_env.items():
            env_backup[key] = os.environ.get(key)
            os.environ[key] = val

    # Find ducc CLI
    cli_path = _find_ducc_cli_path()
    
    # Use empty settings and disable default setting sources to use env vars
    # This allows ~/.experience.json config to take effect via env vars
    config = get_config("coding_agent")
    if config.get("base_url"):
        settings_path = "{}"  # Empty settings, rely on env vars
        setting_sources = ""  # Disable default settings
    else:
        settings_path = _find_ducc_settings_path()
        setting_sources = None
    
    # Get model from parameter or environment variable
    if model is None:
        model = os.environ.get("ANTHROPIC_MODEL")

    try:
        options = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            cwd=cwd,
            cli_path=cli_path,
            settings=settings_path,
        )
        # Add setting_sources if needed
        if setting_sources is not None:
            options.setting_sources = setting_sources
        if model:
            options.model = model
        async for item in query(
            prompt=prompt,
            options=options,
        ):
            # print(item, file=sys.stderr)
            yield item
    finally:
        if llm_env is not None:
            for key in llm_env:
                if env_backup.get(key) is not None:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

def async_run(handlers : list):
    async def func():
        await asyncio.gather(*handlers)
    asyncio.run(func())

if __name__ == '__main__':

    async def foo():
        async for message in coding_agent_query(
            prompt="hello, save your greeting response to a file named './foo.txt'",
            cwd='/tmp/demo',
        ):
            print(message)  # Claude reads the file, finds the bug, edits it
    import os
    os.makedirs('/tmp/demo', exist_ok=True)
    async_run([foo()])

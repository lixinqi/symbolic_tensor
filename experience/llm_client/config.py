"""LLM Configuration Management.

This module provides a unified configuration interface for different LLM methods.
Configuration is read from ~/.experience.json with the following structure:

{
  "raw_llm_api": {
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_key": "your-api-key",
    "model": "deepseek-v3.2",
    "username": "your-username"  // for comate_custom_header auth
  },
  "coding_agent": {
    "base_url": "https://qianfan.baidubce.com/anthropic",
    "api_key": "your-api-key",
    "model": "deepseek-v3.2",
    "username": "your-username"  // for comate_custom_header auth
  },
  "tmux_cc": {
    "base_url": "https://qianfan.baidubce.com/anthropic",
    "api_key": "your-api-key",
    "model": "deepseek-v3.2",
    "username": "your-username"  // for comate_custom_header auth
  }
}
"""

import json
import os
from typing import Optional, Dict, Any

CONFIG_PATH = os.path.expanduser("~/.experience.json")

_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.llm_config.json"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
    else:
        _config_cache = {}
    
    return _config_cache


def get_config(llm_method: str) -> Dict[str, str]:
    """Get configuration for a specific LLM method.
    
    Args:
        llm_method: One of "raw_llm_api", "coding_agent", "tmux_cc"
    
    Returns:
        Dict with keys: base_url, api_key, model, username (any may be None if not configured)
    """
    config = load_config()
    method_config = config.get(llm_method, {})
    
    return {
        "base_url": method_config.get("base_url"),
        "api_key": method_config.get("api_key"),
        "model": method_config.get("model"),
        "username": method_config.get("username"),
    }


def setup_env_for_method(llm_method: str) -> None:
    """Setup environment variables for a specific LLM method.
    
    This sets the appropriate environment variables based on the llm_method:
    - raw_llm_api: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_USERNAME
    - coding_agent/tmux_cc: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    """
    config = get_config(llm_method)
    
    if llm_method == "raw_llm_api":
        if config["base_url"]:
            os.environ["LLM_BASE_URL"] = config["base_url"]
        if config["api_key"]:
            os.environ["LLM_API_KEY"] = config["api_key"]
        if config["model"]:
            os.environ["LLM_MODEL"] = config["model"]
        if config["username"]:
            os.environ["LLM_USERNAME"] = config["username"]
    elif llm_method in ("coding_agent", "tmux_cc"):
        if config["base_url"]:
            os.environ["ANTHROPIC_BASE_URL"] = config["base_url"]
            print(f"[DEBUG] Set ANTHROPIC_BASE_URL = {config['base_url']}", file=__import__('sys').stderr)
        if config["api_key"]:
            os.environ["ANTHROPIC_API_KEY"] = config["api_key"]
        if config["model"]:
            os.environ["ANTHROPIC_MODEL"] = config["model"]
            print(f"[DEBUG] Set ANTHROPIC_MODEL = {config['model']}", file=__import__('sys').stderr)
        if config["username"]:
            os.environ["LLM_USERNAME"] = config["username"]
            print(f"[DEBUG] Set LLM_USERNAME = {config['username']}", file=__import__('sys').stderr)


def get_config_summary(llm_method: str) -> str:
    """Get a summary string of the actual environment variables for logging."""
    if llm_method == "raw_llm_api":
        base_url = os.environ.get("LLM_BASE_URL", "NOT SET")
        model = os.environ.get("LLM_MODEL", "NOT SET")
    else:
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "NOT SET")
        model = os.environ.get("ANTHROPIC_MODEL", "NOT SET")
    return f"base_url={base_url}, model={model}"

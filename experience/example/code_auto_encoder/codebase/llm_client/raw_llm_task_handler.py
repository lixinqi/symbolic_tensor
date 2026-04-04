import os
import asyncio
from typing import Dict, List, Optional
from experience.llm_client.agent_task import AgentTask
from experience.fs_util.pack_dir import pack_dir
def _flatten_nested(nested) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested(item))
    return result
def _grep_by_file_content_hint(root_dir: str, todo_file_content_hint: str) -> List[str]:
    """Find all files under root_dir whose content contains the hint string."""
    todo_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if todo_file_content_hint in content:
                    todo_files.append(file_path)
            except (UnicodeDecodeError, OSError):
                continue
    return todo_files
class RawLlmTaskHandler:
    def __call__(self, all_tasks, llm_env: Optional[Dict[str, str]] = None) -> None:
        flat_tasks = _flatten_nested(all_tasks)
        jobs = []
        for task in flat_tasks:
            workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir_or_list = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint
            packed_workspace = pack_dir(workspace_dir)
            if isinstance(output_relative_dir_or_list, str):
                output_relative_dirs = [output_relative_dir_or_list]
            else:
                output_relative_dirs = output_relative_dir_or_list
            for output_relative_dir in output_relative_dirs:
                output_root = os.path.join(workspace_dir, output_relative_dir)
                todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)
                for todo_file_path in todo_file_paths:
                    raw_llm_prompt = (
                        f"{prompt}\n\n"
                        f"The Directories are packed as following like repomix:\n\n"
                        f"{packed_workspace}\n\n"
                        f"The whole task are split into several pieces, One LLMs request for one piece, "
                        f"In This request, Your output will replace the {todo_file_content_hint} placeholder in file {todo_file_path}. \n"
                        f"Output raw text only. Do NOT wrap in markdown code fences (``` or ```lang).\n"
                        f"Do not generate unrelated content."
                    )
                    jobs.append((todo_file_path, raw_llm_prompt))
        async def _do_one(todo_file_path: str, raw_llm_prompt: str):
            from experience.llm_client.raw_llm_query import raw_llm_query
            output_content = await raw_llm_query(raw_llm_prompt, llm_env=llm_env)
            with open(todo_file_path, "w", encoding="utf-8") as f:
                f.write(output_content)
        async def _run_all():
            await asyncio.gather(*[_do_one(path, prompt) for path, prompt in jobs])
        asyncio.run(_run_all())

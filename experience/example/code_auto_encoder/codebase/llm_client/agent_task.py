from dataclasses import dataclass
from typing import Union, List
@dataclass
class AgentTask:
    workspace_dir: str
    output_relative_dir: Union[str, List[str]]
    prompt: str
    todo_file_content_hint: str = "TODO"

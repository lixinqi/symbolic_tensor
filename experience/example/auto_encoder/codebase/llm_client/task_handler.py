from experience.llm_client.agent_task import AgentTask
from experience.llm_client.coding_agent_task_handler import CodingAgentTaskHandler
from experience.llm_client.raw_llm_task_handler import RawLlmTaskHandler
import time
import sys
from typing import Dict, Optional
class TaskHandler:
    def __call__(self, all_tasks, llm_method: str, llm_env: Optional[Dict[str, str]] = None) -> None:
        start = time.time()
        if llm_method == "coding_agent":
            CodingAgentTaskHandler()(all_tasks, llm_env=llm_env)
        elif llm_method == "raw_llm_api":
            RawLlmTaskHandler()(all_tasks, llm_env=llm_env)
        else:
            raise ValueError(f"Unknown llm_method: {llm_method}")
        end = time.time()
        # print(f"Finished processing tasks in {end-start:.2f} seconds", file=sys.stderr)

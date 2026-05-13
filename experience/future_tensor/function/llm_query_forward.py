"""
llm_query_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $system_prompt str
    <- $task_prompt str
    <- $llm_method str
    <- $llm_env dict[str, str] | None
    # inline

Raw LLM query forward — no experience retrieval.
Sends input content + system prompt to an LLM and returns the response.
"""

from typing import Dict, List, Optional, Tuple

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


def _read_file_content(tensor, flat_index: int) -> Optional[str]:
    """Read the text content at a flat index, resolving symlinks."""
    import os
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        return None
    with open(real_path, encoding="utf-8") as f:
        return f.read()


def llm_query_forward(
    input_ft: FutureTensor,
    system_prompt: str,
    task_prompt: str,
    llm_method: str,
    llm_env: Optional[Dict[str, str]],
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get calls a raw LLM.

    No experience retrieval, no query generation. Just:
      1. Read input content at coordinates
      2. Build prompt: system_prompt + input_content
      3. Call raw_llm_query
      4. Write-through result to output static tensor

    Args:
        input_ft: FutureTensor with text content as input to the LLM.
        system_prompt: System prompt prepended to the LLM call.
        task_prompt: High-level task description (stored for backward).
        llm_method: LLM method name (e.g. "raw_llm_api").
        llm_env: Optional environment variables for LLM config.

    Returns:
        A lazy FutureTensor whose ft_async_get performs the LLM call.
    """
    input_shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_static_tensor.st_relative_to

    async def llm_query_async_get(
        coordinates: List[int], prompt: str,
    ) -> Tuple[str, Status]:
        import asyncio

        # If output already computed at this coordinate, return from disk
        if output.ft_static_tensor.data[tuple(coordinates)].item() > 0:
            flat_idx = sum(
                c * s
                for c, s in zip(coordinates, output.ft_static_tensor.stride())
            )
            content = _read_file_content(output.ft_static_tensor, flat_idx)
            if content is not None:
                return (content, Status.confidence(1.0))

        # Ensure input is materialized at this coordinate (lazy pull + write-through)
        if isinstance(prompt, dict):
            actual_prompt = prompt.get("prompt", "")
        else:
            actual_prompt = prompt

        if (
            not input_ft.ft_forwarded
            and input_ft.ft_static_tensor.data[tuple(coordinates)].item() == 0
        ):
            input_content, input_status = await input_ft.ft_async_get(
                coordinates, actual_prompt,
            )
            if input_content:
                st_setitem(
                    input_ft.ft_static_tensor,
                    coordinates,
                    input_content,
                    coefficient=Status.convert_status_to_float(input_status),
                )
        else:
            # Read from already-materialized static tensor
            flat_idx = sum(
                c * s
                for c, s in zip(coordinates, input_ft.ft_static_tensor.stride())
            )
            input_content = _read_file_content(input_ft.ft_static_tensor, flat_idx)

        if not input_content:
            return ("", Status.self_confidence_but_failed(0.5))

        # Build the full prompt for the LLM
        if system_prompt:
            full_prompt = system_prompt + "\n\n" + input_content
        else:
            full_prompt = input_content

        # Call raw LLM via thread executor
        def _sync_llm_call() -> str:
            import asyncio as _aio
            from experience.llm_client.raw_llm_query import raw_llm_query
            from experience.llm_client.agent_config_factory import AgentConfigFactory

            config = AgentConfigFactory.create_raw_llm_config()
            if llm_env:
                # Override config fields from env dict
                if "base_url" in llm_env:
                    config = config._replace(base_url=llm_env["base_url"]) if hasattr(config, '_replace') else config
                if "api_key" in llm_env:
                    config = config._replace(api_key=llm_env["api_key"]) if hasattr(config, '_replace') else config
                if "model" in llm_env:
                    config = config._replace(model=llm_env["model"]) if hasattr(config, '_replace') else config

            return _aio.run(raw_llm_query(full_prompt, config))

        loop = asyncio.get_event_loop()
        try:
            result_content = await loop.run_in_executor(None, _sync_llm_call)
        except Exception as e:
            import sys
            print(f"[llm_query_forward] LLM call failed: {e}", file=sys.stderr)
            return ("", Status.self_confidence_but_failed(0.5))

        if not result_content or not result_content.strip():
            return ("", Status.self_confidence_but_failed(0.5))

        # Write-through: persist output to static tensor
        st_setitem(
            output.ft_static_tensor,
            coordinates,
            result_content,
            coefficient=1.0,
        )

        return (result_content, Status.confidence(1.0))

    output = FutureTensor(
        relative_to,
        llm_query_async_get,
        ft_shape_schema=[sympy.Integer(s) for s in input_shape],
    )

    return output

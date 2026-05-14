"""
ft_expert := ft_llm_call(ft_build_expert_context(input, ft_retrieve_experience(input, experience), task_prompt))

Composes ft_retrieve_experience → ft_build_expert_context → ft_llm_call.
Each op is a separate autograd.Function with its own backward/2nd-derivative dispatch.
"""

import torch
from typing import Callable, Dict, Optional

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.ft_retrieve_experience import ft_retrieve_experience
from experience.future_tensor.function.ft_build_expert_context import ft_build_expert_context
from experience.future_tensor.function.ft_llm_call import ft_llm_call


def ft_expert(
    input_ft: FutureTensor,
    experience: torch.Tensor,
    task_prompt_st: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
    topk: int = 16,
    retrieval_method: Optional[Callable[[str, str], float]] = None,
    skip_query_gen: bool = False,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt_for_query: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> FutureTensor:
    """Expert: retrieve experience → build prompt → LLM call.

    Thin composition of three decomposed ops. Each op has its own
    autograd.Function with backward and 2nd-derivative dispatch,
    so TracePolicy sees all three nodes in the graph.

    Args:
        input_ft: FutureTensor input (text content per element).
        experience: ExperienceTensor (last dim=3: query, key, value).
        task_prompt_st: 0D trainable symbolic tensor with task description.
        output_prompt: Custom prompt builder (task_prompt, exp_text, input_text, prompt) -> str.
        topk: Number of top experience entries to retrieve.
        retrieval_method: Custom similarity function for retrieval.
        skip_query_gen: If True, use input directly as retrieval query.
        query_prompt: Custom query prompt builder for retrieval.
        task_prompt_for_query: Task description for query generation LLM call.
        llm_method: LLM method name.
        llm_env: Optional environment variables for LLM config.

    Returns:
        FutureTensor containing LLM responses.
    """
    exp_text = ft_retrieve_experience(
        input_ft, experience,
        topk=topk,
        retrieval_method=retrieval_method,
        skip_query_gen=skip_query_gen,
        query_prompt=query_prompt,
        task_prompt=task_prompt_for_query,
        llm_method=llm_method,
        llm_env=llm_env,
    )
    prompt = ft_build_expert_context(
        input_ft, exp_text,
        task_prompt_st,
        output_prompt=output_prompt,
    )
    return ft_llm_call(prompt, llm_method=llm_method, llm_env=llm_env)

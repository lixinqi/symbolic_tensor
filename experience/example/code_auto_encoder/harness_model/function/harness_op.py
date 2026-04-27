"""HarnessOp base class for deterministic worktree operators.

A HarnessOp is a pure function: no LLM, no randomness.
Forward: execute on worktree, return result string.
Backward: no gradient (deterministic — nothing to learn).

HarnessOps are the "tools" that the agent (LLM inside ft_async_get)
can invoke. They never load full directory contents — they provide
targeted, selective access.
"""


class HarnessOp:
    """Base class for deterministic worktree operators."""

    name: str = ""
    description: str = ""

    def forward(self, worktree_path: str, **kwargs) -> str:
        raise NotImplementedError

    def schema(self) -> str:
        """Tool schema string for LLM system prompt."""
        raise NotImplementedError

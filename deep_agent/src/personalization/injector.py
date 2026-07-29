"""Inject user rules into the agent system prompt.

Appends a **User Custom Instructions** block to the base system prompt
when the user has active rules. The block is omitted when the list is
empty, keeping the prompt clean for users who haven't configured rules.
"""

from __future__ import annotations


def inject_rules(
    system_prompt: str,
    rules: list[str],
) -> str:
    """Return *system_prompt* enriched with user rules.

    Rules are ordered newest-first (as returned by the repository).
    When rules conflict, the LLM is instructed to follow the most
    recent one.

    Args:
        system_prompt: The base system prompt from config.
        rules: Plain-text user rules / custom instructions (newest first).

    Returns:
        The enriched system prompt. Unchanged if the list is empty.
    """
    if not rules:
        return system_prompt

    lines = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
    block = (
        "## User Custom Instructions\n\n"
        "The user has defined the following rules (listed from most recent "
        "to oldest). Follow them for every response unless they conflict "
        "with safety guidelines. **If two rules conflict, the more recent "
        f"rule (lower number) takes precedence.**\n\n{lines}"
    )
    return f"{system_prompt}\n\n---\n\n{block}"

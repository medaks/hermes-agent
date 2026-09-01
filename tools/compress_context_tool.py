#!/usr/bin/env python3
"""Tool schema + runtime helper for agent-initiated context compression."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


COMPRESS_CONTEXT_SCHEMA = {
    "name": "compress_context",
    "description": (
        "Compress conversation context by summarizing older messages. "
        "Use when context is large or when asked to compact the current session. "
        "You may provide focus_topic to preserve details relevant to that topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus_topic": {
                "type": "string",
                "description": (
                    "Optional topic to preserve with higher priority while "
                    "summarizing the rest more aggressively."
                ),
            },
            "force": {
                "type": "boolean",
                "description": (
                    "If true, bypass summary-failure cooldown and retry now. "
                    "Use only after a failed compression attempt."
                ),
                "default": False,
            },
        },
        "required": [],
    },
}


def compress_context_tool(
    *,
    agent: Any,
    messages: Optional[List[Dict[str, Any]]],
    task_id: str,
    focus_topic: Optional[str] = None,
    force: bool = False,
) -> str:
    """Run context compression and update the live in-memory message list."""
    if agent is None:
        return tool_error("No agent context available for compression.")

    if not bool(getattr(agent, "compression_enabled", True)):
        return tool_error("Compression is disabled in config.")

    active_messages = messages if isinstance(messages, list) else getattr(agent, "_session_messages", None)
    if not isinstance(active_messages, list) or not active_messages:
        return tool_error("No conversation history available to compress.")
    if len(active_messages) < 4:
        return tool_error(f"Not enough conversation to compress (need at least 4 messages, got {len(active_messages)}).")

    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return tool_error("No context compressor is configured.")

    try:
        if hasattr(compressor, "has_content_to_compress") and not compressor.has_content_to_compress(active_messages):
            return tool_result(
                success=True,
                noop=True,
                message="Nothing to compress yet; conversation is still within protected head/tail bounds.",
                pre_messages=len(active_messages),
                post_messages=len(active_messages),
                compression_count=getattr(compressor, "compression_count", 0),
                session_id=getattr(agent, "session_id", None),
            )
    except Exception as exc:
        logger.debug("compress_context preflight check failed; continuing: %s", exc)

    # Gate behind should_compress() so the tool respects the same cooldown,
    # anti-thrash, and threshold guards as the auto-compression path. Without
    # this the tool runs unconditionally whenever a middle region exists, which
    # can loop (model calls compress_context → compression runs → model calls
    # it again on the next nudge) with no backstop.
    try:
        from agent.model_metadata import estimate_request_tokens_rough as _est_rough

        _rough_tokens = _est_rough(
            active_messages,
            system_prompt=(getattr(agent, "_cached_system_prompt", "") or ""),
            tools=(getattr(agent, "tools", None) or None),
        )
    except Exception:
        _rough_tokens = 0

    _force = bool(force)

    if not _force:
        # Respect the summary-LLM failure cooldown.
        _cooldown = getattr(compressor, "get_active_compression_failure_cooldown", lambda: None)()
        if _cooldown:
            return tool_result(
                success=True,
                noop=True,
                message=(
                    f"Compression skipped — summary LLM in cooldown for "
                    f"{_cooldown.get('remaining_seconds', 0):.0f}s more."
                ),
                pre_messages=len(active_messages),
                post_messages=len(active_messages),
                compression_count=getattr(compressor, "compression_count", 0),
                session_id=getattr(agent, "session_id", None),
            )

        # Respect anti-thrash and threshold guards.
        if hasattr(compressor, "should_compress") and not compressor.should_compress(_rough_tokens):
            return tool_result(
                success=True,
                noop=True,
                message=(
                    f"Context is not yet large enough to compress "
                    f"(~{_rough_tokens:,} tokens below threshold "
                    f"{getattr(compressor, 'threshold_tokens', 0):,}). "
                    f"Auto-compression will handle it when needed."
                ),
                pre_messages=len(active_messages),
                post_messages=len(active_messages),
                compression_count=getattr(compressor, "compression_count", 0),
                session_id=getattr(agent, "session_id", None),
            )

    approx_tokens_before = _rough_tokens

    pre_count = len(active_messages)
    try:
        compressed, _ = agent._compress_context(
            active_messages,
            None,
            approx_tokens=approx_tokens_before,
            task_id=task_id,
            focus_topic=(focus_topic or None),
            force=bool(force),
        )
    except Exception as exc:
        logger.warning("compress_context failed: %s", exc, exc_info=True)
        return tool_error(f"Compression failed: {exc}")

    if isinstance(messages, list):
        messages[:] = compressed
        agent._session_messages = messages
    else:
        agent._session_messages = compressed

    approx_tokens_after = None
    try:
        from agent.model_metadata import estimate_request_tokens_rough

        approx_tokens_after = estimate_request_tokens_rough(
            compressed,
            system_prompt=(getattr(agent, "_cached_system_prompt", "") or ""),
            tools=(getattr(agent, "tools", None) or None),
        )
    except Exception:
        pass

    post_count = len(compressed)
    removed = pre_count - post_count
    noop = post_count == pre_count

    return tool_result(
        success=True,
        noop=noop,
        message=(
            "Compression finished with no transcript shrink."
            if noop
            else f"Compressed conversation from {pre_count} to {post_count} messages."
        ),
        pre_messages=pre_count,
        post_messages=post_count,
        removed_messages=removed if removed > 0 else 0,
        approx_tokens_before=approx_tokens_before,
        approx_tokens_after=approx_tokens_after,
        compression_count=getattr(compressor, "compression_count", 0),
        session_id=getattr(agent, "session_id", None),
        focus_topic=(focus_topic or None),
    )


registry.register(
    name="compress_context",
    toolset="core",
    schema=COMPRESS_CONTEXT_SCHEMA,
    # compress_context is handled in agent_runtime_helpers.invoke_tool so it
    # can mutate the live turn-local messages list safely.
    handler=lambda args, **kw: tool_error("compress_context must be handled by the agent loop"),
    emoji="🗜️",
)


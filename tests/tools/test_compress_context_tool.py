import json
from unittest.mock import MagicMock, patch


def _agent_with_defaults():
    agent = MagicMock()
    agent.compression_enabled = True
    agent.session_id = "s1"
    agent._cached_system_prompt = "sys"
    agent.tools = []
    agent._memory_manager = None
    agent.context_compressor = MagicMock()
    agent.context_compressor.compression_count = 1
    agent.context_compressor.has_content_to_compress.return_value = True
    return agent


def test_schema_shape_is_openai_function_schema():
    from tools.compress_context_tool import COMPRESS_CONTEXT_SCHEMA

    assert COMPRESS_CONTEXT_SCHEMA["name"] == "compress_context"
    assert "description" in COMPRESS_CONTEXT_SCHEMA
    assert "parameters" in COMPRESS_CONTEXT_SCHEMA
    assert COMPRESS_CONTEXT_SCHEMA["parameters"]["type"] == "object"


def test_no_agent_returns_json_error():
    from tools.compress_context_tool import compress_context_tool

    raw = compress_context_tool(agent=None, messages=[], task_id="t1")
    data = json.loads(raw)
    assert "error" in data


def test_short_history_returns_json_error():
    from tools.compress_context_tool import compress_context_tool

    agent = _agent_with_defaults()
    messages = [{"role": "user", "content": "hi"}]
    raw = compress_context_tool(agent=agent, messages=messages, task_id="t1")
    data = json.loads(raw)
    assert "error" in data
    assert "at least 4" in data["error"]


def test_no_content_to_compress_is_noop_success():
    from tools.compress_context_tool import compress_context_tool

    agent = _agent_with_defaults()
    agent.context_compressor.has_content_to_compress.return_value = False
    messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]

    data = json.loads(compress_context_tool(agent=agent, messages=messages, task_id="t1"))
    assert data["success"] is True
    assert data["noop"] is True
    assert data["pre_messages"] == data["post_messages"] == 5
    agent._compress_context.assert_not_called()


def test_compress_updates_live_messages_and_passes_focus_force():
    from tools.compress_context_tool import compress_context_tool

    agent = _agent_with_defaults()
    messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    compressed = [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "next"}]
    agent._compress_context.return_value = (compressed, "new_sys")

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=1234):
        data = json.loads(
            compress_context_tool(
                agent=agent,
                messages=messages,
                task_id="task-42",
                focus_topic="schema migration",
                force=True,
            )
        )

    assert data["success"] is True
    assert data["noop"] is False
    assert data["pre_messages"] == 8
    assert data["post_messages"] == 2
    assert messages == compressed
    assert agent._session_messages is messages
    agent._compress_context.assert_called_once_with(
        messages,
        None,
        approx_tokens=1234,
        task_id="task-42",
        focus_topic="schema migration",
        force=True,
    )


def test_invoke_tool_routes_compress_context_via_agent_loop():
    from agent.agent_runtime_helpers import invoke_tool

    agent = _agent_with_defaults()
    messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]

    with patch("tools.compress_context_tool.compress_context_tool", return_value='{"ok":true}') as mock_tool:
        result = invoke_tool(
            agent,
            "compress_context",
            {"focus_topic": "db", "force": True},
            effective_task_id="task-a",
            messages=messages,
        )

    assert result == '{"ok":true}'
    mock_tool.assert_called_once_with(
        agent=agent,
        messages=messages,
        task_id="task-a",
        focus_topic="db",
        force=True,
    )


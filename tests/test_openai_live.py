from __future__ import annotations

import json
import unittest
from typing import Any

from voice_demo.openai_live.agent import (
    LiveEventProcessor,
    TranscriptRows,
    session_config,
)
from voice_demo.openai_live.tools import execute_tool
from voice_demo.openai_live.tracing import LiveTracer


class _AudioOut:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.audio: list[bytes] = []

    def start(self) -> None:  # pragma: no cover - protocol fixture
        pass

    def write(self, data: bytes) -> None:
        self.audio.append(data)

    def buffered_bytes(self) -> int:
        return sum(map(len, self.audio))

    def clear(self) -> None:
        self.audio.clear()

    def set_played_callback(self, callback) -> None:  # noqa: ANN001
        pass

    def stop(self) -> None:  # pragma: no cover - protocol fixture
        pass


class _UI:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.states: list[str] = []

    def set_state(self, state: str) -> None:
        self.states.append(state)

    def update_level(self, level: float) -> None:
        pass

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def finish(self) -> None:
        pass


def _tracer() -> LiveTracer:
    return LiveTracer(
        project_name="test",
        thread_id="thread",
        live_model="gpt-live-1",
        backend_model="gpt-5.6-luna",
        enabled=False,
    )


class SessionConfigTests(unittest.TestCase):
    def test_uses_live_model_and_responses_delegation(self) -> None:
        session = session_config()
        self.assertEqual(session["model"], "gpt-live-1")
        self.assertFalse(session["store"])
        responses = session["delegation"]["responses"]
        self.assertEqual(session["delegation"]["type"], "responses")
        self.assertEqual(responses["tools"][0]["name"], "lookup_weather")
        self.assertTrue(responses["tools"][0]["strict"])
        self.assertTrue(responses["parallel_tool_calls"])

    def test_transcript_rows_are_bounded_and_grouped(self) -> None:
        rows = TranscriptRows(gap_ms=100)
        self.assertIsNone(rows.add("user", "hello", 0, 50))
        self.assertEqual(rows.add("user", "again", 200, 250), "hello")
        flushed = rows.flush()
        self.assertEqual(flushed, [("user", "again")])


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_unknown_and_extra_arguments_without_network(self) -> None:
        self.assertEqual(
            await execute_tool("delete_everything", "{}"),
            {"error": "unknown_tool", "tool": "delete_everything"},
        )
        self.assertEqual(
            await execute_tool(
                "lookup_weather", json.dumps({"city": "Paris", "admin": True})
            ),
            {"error": "invalid_arguments"},
        )


class EventProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_all_tool_results_before_continuing(self) -> None:
        sent: list[dict[str, Any]] = []
        calls: list[tuple[str, str]] = []

        async def send(event: dict[str, Any]) -> None:
            sent.append(event)

        async def run_tool(name: str, arguments: str) -> dict[str, Any]:
            calls.append((name, arguments))
            return {"ok": json.loads(arguments)["city"]}

        processor = LiveEventProcessor(
            send_event=send,
            audio_out=_AudioOut(),
            ui=_UI(),
            tracer=_tracer(),
            tool_runner=run_tool,
        )
        await processor.handle(
            {
                "type": "session.delegation.created",
                "delegation_id": "delegation-1",
                "target": "responses",
            }
        )
        await processor.handle(
            {
                "type": "response.event",
                "delegation_id": "delegation-1",
                "event": {
                    "type": "response.created",
                    "response": {"id": "response-1"},
                },
            }
        )
        for index, city in enumerate(("Rome", "Berlin"), start=1):
            await processor.handle(
                {
                    "type": "response.event",
                    "delegation_id": "delegation-1",
                    "event": {
                        "type": "response.output_item.done",
                        "response_id": "response-1",
                        "item": {
                            "type": "function_call",
                            "call_id": f"call-{index}",
                            "name": "lookup_weather",
                            "arguments": json.dumps({"city": city}),
                        },
                    },
                }
            )
        await processor.handle(
            {
                "type": "response.event",
                "delegation_id": "delegation-1",
                "event": {
                    "type": "response.completed",
                    "response": {"id": "response-1"},
                },
            }
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [event["type"] for event in sent],
            ["response.item.create", "response.item.create", "response.create"],
        )
        self.assertEqual(sent[0]["item"]["call_id"], "call-1")
        self.assertEqual(sent[1]["item"]["call_id"], "call-2")

    async def test_unknown_events_are_ignored(self) -> None:
        async def send(event: dict[str, Any]) -> None:
            raise AssertionError(event)

        processor = LiveEventProcessor(
            send_event=send,
            audio_out=_AudioOut(),
            ui=_UI(),
            tracer=_tracer(),
        )
        self.assertTrue(await processor.handle({"type": "future.event"}))


if __name__ == "__main__":
    unittest.main()

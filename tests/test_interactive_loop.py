from __future__ import annotations

import io
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.context import ContextAssemblyConfig
from fullerene.facets import ContextFacet, EchoFacet, MemoryFacet
from fullerene.interactive import InteractiveLoopConfig, run_interactive_loop
from fullerene.interactive.renderer import render_interactive_frame
from fullerene.memory import SQLiteMemoryStore
from fullerene.nexus import (
    DecisionAction,
    Event,
    EventType,
    FacetResult,
    NexusDecision,
    NexusRecord,
    NexusRuntime,
)
from fullerene.state import FileStateStore
from fullerene.workspace_state import workspace_state_root


def make_tempdir_path() -> Path:
    root = workspace_state_root() / f".test-interactive-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _FakeInputProvider:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.closed = False

    def poll_line(self, _timeout_seconds: float) -> str | None:
        if self._lines:
            return self._lines.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.state = type("S", (), {"facet_state": {}, "event_count": 0})()
        self.user_calls = 0
        self.tick_calls = 0

    def process_event(self, event: Event) -> NexusRecord:
        self.events.append(event)
        self.state.event_count += 1
        if event.event_type == EventType.SYSTEM_TICK:
            self.tick_calls += 1
            decision = NexusDecision(action=DecisionAction.WAIT, reason="tick")
            metadata = {
                "system_pressure": 0.2,
                "signal_map": {"system_pressure": 0.2, "policy_status": "allowed"},
                "expression_recommendation": {"mode": "silent", "suppressed": True},
            }
        else:
            self.user_calls += 1
            decision = NexusDecision(action=DecisionAction.ACT, reason="user")
            metadata = {
                "system_pressure": 0.4,
                "signal_map": {"system_pressure": 0.4, "policy_status": "allowed"},
            }
        return NexusRecord(
            event=event,
            facet_results=[
                FacetResult(
                    facet_name="echo",
                    summary="ok",
                    proposed_decision=decision.action,
                    metadata={"output_type": "text", "response_needed": True},
                )
            ],
            decision=decision,
            metadata=metadata,
        )


class InteractiveConfigTests(unittest.TestCase):
    def test_defaults_and_clamping(self) -> None:
        cfg = InteractiveLoopConfig().clamped()
        self.assertEqual(cfg.interval_seconds, 1.0)
        self.assertEqual(cfg.max_ticks, 1000)
        self.assertFalse(cfg.clear_screen)
        self.assertFalse(cfg.allow_model)
        self.assertFalse(cfg.show_ticks)
        self.assertEqual(cfg.status_every, 0)
        self.assertEqual(cfg.input_prompt, "> ")

        cfg2 = InteractiveLoopConfig(interval_seconds=999, max_ticks=0).clamped()
        self.assertEqual(cfg2.interval_seconds, 60.0)
        self.assertEqual(cfg2.max_ticks, 1)


class InteractiveRunnerTests(unittest.TestCase):
    def test_exit_command_stops_cleanly(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider(["quit"])
        result = run_interactive_loop(
            runtime, InteractiveLoopConfig(max_ticks=5, interval_seconds=0), output_writer=writer, input_provider=provider
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "exit_command")

    def test_user_input_becomes_user_message_metadata(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["hello", "quit"])
        result = run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=3, interval_seconds=0),
            input_provider=provider,
            user_output_builder=lambda _record: "ok",
        )
        self.assertEqual(result.inputs_processed, 1)
        user_events = [e for e in runtime.events if e.event_type == EventType.USER_MESSAGE]
        self.assertEqual(len(user_events), 1)
        md = user_events[0].metadata
        self.assertTrue(md["interactive_input"])
        self.assertEqual(md["input_index"], 1)
        self.assertFalse(md["suppress_expression"])

    def test_empty_input_ignored_and_ticks_continue(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["", "  ", "quit"])
        result = run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=3, interval_seconds=0),
            input_provider=provider,
        )
        self.assertEqual(result.inputs_processed, 0)
        self.assertGreaterEqual(result.ticks_processed, 1)

    def test_system_tick_processed_between_inputs(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["hello", "quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=4, interval_seconds=0),
            input_provider=provider,
            user_output_builder=lambda _record: "response",
        )
        event_types = [e.event_type for e in runtime.events]
        self.assertEqual(event_types[0], EventType.SYSTEM_TICK)
        self.assertIn(EventType.USER_MESSAGE, event_types)

    def test_user_output_updates_final_text(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["hello", "quit"])
        result = run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=2, interval_seconds=0),
            input_provider=provider,
            user_output_builder=lambda _record: "user-response",
        )
        self.assertEqual(result.final_text_output, "user-response")

    def test_default_mode_is_transcript_and_tick_silent(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider(["quit"])
        result = run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=8, interval_seconds=0),
            output_writer=writer,
            input_provider=provider,
        )
        output = writer.getvalue()
        self.assertEqual(result.stop_reason, "exit_command")
        self.assertIn("Interactive loop started", output)
        self.assertNotIn("[tick ", output)

    def test_transcript_persists_user_and_response_lines(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider(["hello", "quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=8, interval_seconds=0),
            output_writer=writer,
            input_provider=provider,
            user_output_builder=lambda _record: "response-text",
        )
        output = writer.getvalue()
        self.assertIn("You: hello", output)
        self.assertIn("Fullerene: response-text", output)
        self.assertIn("[status] mode=", output)

    def test_show_ticks_prints_compact_tick_lines(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider(["quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(
                max_ticks=3,
                interval_seconds=0,
                show_ticks=True,
            ),
            output_writer=writer,
            input_provider=provider,
        )
        self.assertIn("[tick 1]", writer.getvalue())

    def test_status_every_prints_only_selected_ticks(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider([])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(
                max_ticks=5,
                interval_seconds=0,
                show_ticks=True,
                status_every=5,
            ),
            output_writer=writer,
            input_provider=provider,
        )
        out = writer.getvalue()
        self.assertIn("[tick 5]", out)
        self.assertNotIn("[tick 1]", out)

    def test_slash_status_and_help_render(self) -> None:
        runtime = _FakeRuntime()
        writer = io.StringIO()
        provider = _FakeInputProvider(["/status", "/help", "quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=4, interval_seconds=0),
            output_writer=writer,
            input_provider=provider,
        )
        out = writer.getvalue()
        self.assertIn("[status] tick=", out)
        self.assertIn("Commands: /status, /help, /quit", out)

    def test_slash_quit_exits_cleanly(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["/quit"])
        result = run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=4, interval_seconds=0),
            input_provider=provider,
        )
        self.assertEqual(result.stop_reason, "exit_command")

    def test_tick_never_calls_user_output_builder(self) -> None:
        runtime = _FakeRuntime()
        provider = _FakeInputProvider(["", "", "quit"])
        calls: list[str] = []
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=4, interval_seconds=0),
            input_provider=provider,
            user_output_builder=lambda _record: calls.append("user") or "ok",
        )
        self.assertEqual(calls, [])


class InteractiveRendererTests(unittest.TestCase):
    def test_clear_and_no_clear_render(self) -> None:
        cfg_clear = InteractiveLoopConfig(clear_screen=True).clamped()
        s1 = render_interactive_frame(
            tick_index=1,
            input_index=0,
            mode="thinking",
            pressure=0.3,
            text_output="(silent)",
            config=cfg_clear,
        )
        self.assertIn("Fullerene Interactive", s1)

        cfg_line = InteractiveLoopConfig(clear_screen=False).clamped()
        s2 = render_interactive_frame(
            tick_index=1,
            input_index=0,
            mode="thinking",
            pressure=0.3,
            text_output="(silent)",
            config=cfg_line,
        )
        self.assertIn("tick=1", s2)


class InteractiveCLITests(unittest.TestCase):
    def test_interactive_rejects_model_without_allow_flag(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--interactive", "--model", "ollama:gemma3:4b"])
        self.assertEqual(ctx.exception.code, 2)

    def test_interactive_rejects_json(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--interactive", "--json"])
        self.assertEqual(ctx.exception.code, 2)

    def test_interactive_status_every_invalid_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--interactive", "--interactive-status-every", "-1"])
        self.assertEqual(ctx.exception.code, 2)


class InteractiveWorkingMemoryTests(unittest.TestCase):
    def test_interactive_stores_user_and_assistant_working_turns(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    store,
                    config=ContextAssemblyConfig(
                        max_working_turns=8,
                        include_policy_summary=False,
                        include_signal_summaries=False,
                    ),
                ),
                MemoryFacet(store),
                EchoFacet(),
            ],
            store=FileStateStore(root),
        )
        provider = _FakeInputProvider(["Do you know your name?", "/quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=3, interval_seconds=0, session_id="session-test"),
            input_provider=provider,
            user_output_builder=lambda _record: "I don't have a name.",
        )
        turns = store.list_working_turns("session-test", limit=8)
        self.assertEqual([t.metadata["dialogue_role"] for t in turns], ["user", "assistant"])
        self.assertEqual(turns[0].content, "Do you know your name?")
        self.assertEqual(turns[1].content, "I don't have a name.")

    def test_continuity_context_contains_previous_turn_pair(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        runtime = NexusRuntime(
            facets=[
                ContextFacet(
                    store,
                    config=ContextAssemblyConfig(
                        max_working_turns=8,
                        include_policy_summary=False,
                        include_signal_summaries=False,
                    ),
                ),
                MemoryFacet(store),
                EchoFacet(),
            ],
            store=FileStateStore(root),
        )
        provider = _FakeInputProvider(["Do you know your name?", "Would you like one?", "/quit"])
        prompts: list[str] = []

        def _capture_output(record: NexusRecord) -> str:
            from fullerene.cli import _build_model_prompt

            prompts.append(
                _build_model_prompt(
                    record,
                    {"query_intent": "factual", "response_template": "grounded_response_available"},
                )
            )
            if len(prompts) == 1:
                return "I don't have a name."
            return "Maybe."

        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=6, interval_seconds=0, session_id="session-test"),
            input_provider=provider,
            user_output_builder=_capture_output,
        )
        second_prompt = prompts[-1]
        self.assertIn("Recent conversation:", second_prompt)
        self.assertIn("User: Do you know your name?", second_prompt)
        self.assertIn("Assistant: I don't have a name.", second_prompt)
        self.assertIn("Conversation continuity:", second_prompt)
        self.assertNotIn("\"reference_anchors\"", second_prompt)

    def test_system_ticks_do_not_store_dialogue_turns(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        store = SQLiteMemoryStore(root / "memory.sqlite3")
        runtime = NexusRuntime(facets=[MemoryFacet(store), EchoFacet()], store=FileStateStore(root))
        provider = _FakeInputProvider(["/quit"])
        run_interactive_loop(
            runtime,
            InteractiveLoopConfig(max_ticks=4, interval_seconds=0, session_id="session-test"),
            input_provider=provider,
        )
        self.assertEqual(store.list_working_turns("session-test", limit=8), [])


if __name__ == "__main__":
    unittest.main()

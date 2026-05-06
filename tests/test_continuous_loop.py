from __future__ import annotations

import io
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.continuous import ContinuousLoopConfig, run_continuous_loop
from fullerene.facets import BehaviorFacet, EchoFacet
from fullerene.nexus import DecisionAction, Event, EventType, NexusDecision, NexusRecord, NexusRuntime
from fullerene.state import InMemoryStateStore
from fullerene.workspace_state import workspace_state_root


def _temp_state_dir() -> Path:
    root = workspace_state_root() / f".test-loop-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class ContinuousConfigTests(unittest.TestCase):
    def test_defaults_and_clamping(self) -> None:
        cfg = ContinuousLoopConfig().clamped()
        self.assertEqual(cfg.interval_seconds, 1.0)
        self.assertEqual(cfg.max_ticks, 100)
        self.assertTrue(cfg.clear_screen)
        self.assertTrue(cfg.mode_pressure_only)
        self.assertFalse(cfg.allow_tick_expression)

        cfg2 = ContinuousLoopConfig(interval_seconds=999, max_ticks=0).clamped()
        self.assertEqual(cfg2.interval_seconds, 60.0)
        self.assertEqual(cfg2.max_ticks, 1)


class ContinuousLoopRunnerTests(unittest.TestCase):
    def test_runs_n_ticks_and_stops_at_max(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet(), EchoFacet()], store=InMemoryStateStore())
        out = io.StringIO()
        result = run_continuous_loop(
            runtime,
            ContinuousLoopConfig(interval_seconds=0, max_ticks=3, clear_screen=False),
            output_writer=out,
        )
        self.assertEqual(result.tick_count, 3)
        self.assertFalse(result.stopped_early)
        self.assertEqual(runtime.state.event_count, 3)

    def test_no_clear_prints_compact_lines(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet(), EchoFacet()], store=InMemoryStateStore())
        out = io.StringIO()
        run_continuous_loop(
            runtime,
            ContinuousLoopConfig(interval_seconds=0, max_ticks=2, clear_screen=False),
            output_writer=out,
        )
        lines = [ln for ln in out.getvalue().splitlines() if ln.startswith("tick=")]
        self.assertEqual(len(lines), 2)
        self.assertIn("mode=", lines[0])
        self.assertIn("pressure=", lines[0])
        self.assertIn("text=", lines[0])

    def test_clear_screen_mode_uses_minimal_three_line_frame(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet(), EchoFacet()], store=InMemoryStateStore())
        out = io.StringIO()
        run_continuous_loop(
            runtime,
            ContinuousLoopConfig(interval_seconds=0, max_ticks=1, clear_screen=True),
            output_writer=out,
        )
        rendered = out.getvalue()
        self.assertIn("Fullerene Loop | tick 1/1", rendered)
        self.assertIn("mode=", rendered)
        self.assertIn("pressure=", rendered)
        self.assertIn("text=", rendered)

    def test_keyboard_interrupt_is_clean_stop(self) -> None:
        class _InterruptRuntime:
            def __init__(self) -> None:
                self.state = type("S", (), {"facet_state": {}})()

            def process_event(self, _event):  # noqa: ANN001
                raise KeyboardInterrupt()

        result = run_continuous_loop(
            _InterruptRuntime(),  # type: ignore[arg-type]
            ContinuousLoopConfig(interval_seconds=0, max_ticks=3),
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_reason, "keyboard_interrupt")
        self.assertEqual(result.tick_count, 0)

    def test_stop_on_ask_user_when_configured(self) -> None:
        class _AskRuntime:
            def __init__(self) -> None:
                self._count = 0
                self.state = type("S", (), {"facet_state": {}, "event_count": 0})()

            def process_event(self, event: Event) -> NexusRecord:
                self._count += 1
                self.state.event_count = self._count
                return NexusRecord(
                    event=event,
                    facet_results=[],
                    decision=NexusDecision(action=DecisionAction.ASK, reason="ask"),
                    metadata={
                        "system_pressure": 0.2,
                        "signal_map": {"system_pressure": 0.2, "policy_status": "allowed"},
                        "expression_recommendation": {
                            "mode": "ask_user",
                            "suppressed": False,
                            "suggested_intent": "ask_approval",
                        },
                    },
                )

        res = run_continuous_loop(
            _AskRuntime(),  # type: ignore[arg-type]
            ContinuousLoopConfig(
                interval_seconds=0,
                max_ticks=5,
                clear_screen=False,
                stop_on_ask_user=True,
                allow_tick_expression=True,
            ),
        )
        self.assertTrue(res.stopped_early)
        self.assertEqual(res.stop_reason, "expression_ask_user")
        self.assertEqual(res.tick_count, 1)

    def test_latent_only_high_pressure_does_not_stop_after_five_ticks(self) -> None:
        runtime = NexusRuntime(facets=[BehaviorFacet(), EchoFacet()], store=InMemoryStateStore())
        out = io.StringIO()
        result = run_continuous_loop(
            runtime,
            ContinuousLoopConfig(interval_seconds=0, max_ticks=8, clear_screen=False),
            output_writer=out,
            extra_metadata={"pressure": 0.99},
        )
        self.assertFalse(result.stopped_early)
        self.assertEqual(result.tick_count, 8)


class ContinuousLoopCLITests(unittest.TestCase):
    def test_loop_json_output_shape(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        from contextlib import redirect_stdout

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--loop",
                    "--loop-max-ticks",
                    "3",
                    "--loop-interval",
                    "0.1",
                    "--loop-json",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("continuous_loop", payload)
        self.assertEqual(payload["continuous_loop"]["tick_count"], 3)

    def test_loop_rejects_model(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--loop", "--model", "ollama:gemma3:4b"])
        self.assertEqual(ctx.exception.code, 2)

    def test_loop_interval_and_max_ticks_clamp(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        from contextlib import redirect_stdout

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--full",
                    "--loop",
                    "--loop-json",
                    "--loop-max-ticks",
                    "0",
                    "--loop-interval",
                    "999",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())["continuous_loop"]
        self.assertEqual(payload["tick_count"], 1)
        self.assertEqual(payload["metadata"]["interval_seconds"], 60.0)


if __name__ == "__main__":
    unittest.main()

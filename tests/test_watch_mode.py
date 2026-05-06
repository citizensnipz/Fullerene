from __future__ import annotations

import io
import json
import shutil
import unittest
from contextlib import redirect_stdout
from uuid import uuid4
from pathlib import Path

from fullerene.cli import main as cli_main
from fullerene.watch import WatchConfig, render_watch_snapshot, WatchSnapshot
from fullerene.watch.renderer import render_watch_trace
from fullerene.workspace_state import workspace_state_root


def _temp_state_dir() -> Path:
    root = workspace_state_root() / f".test-watch-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class WatchConfigUnitTests(unittest.TestCase):
    def test_defaults_and_clamping(self) -> None:
        cfg = WatchConfig()
        clamped = cfg.clamped()
        self.assertEqual(clamped.ticks, 10)
        self.assertEqual(clamped.interval_seconds, 1.0)
        self.assertFalse(clamped.clear_screen)
        self.assertTrue(clamped.show_presentation)
        self.assertTrue(clamped.show_pressure)
        self.assertTrue(clamped.show_interrupts)
        self.assertTrue(clamped.show_expression)
        self.assertTrue(clamped.stop_on_tick_runner_stop)

        clamped2 = WatchConfig(ticks=999, max_ticks=100, interval_seconds=999).clamped()
        self.assertEqual(clamped2.ticks, 100)
        self.assertEqual(clamped2.interval_seconds, 60.0)

        clamped3 = WatchConfig(ticks=0, max_ticks=5, interval_seconds=-1).clamped()
        self.assertEqual(clamped3.ticks, 1)
        self.assertEqual(clamped3.interval_seconds, 0.0)


class WatchSnapshotUnitTests(unittest.TestCase):
    def test_snapshot_to_dict_json_roundtrip(self) -> None:
        snap = WatchSnapshot(
            tick_index=1,
            tick_count=3,
            timestamp="2026-01-01T00:00:00+00:00",
            decision="wait",
            system_pressure=0.1,
            latent_pressure=0.2,
            presentation_mode="thinking",
            presentation_motion="ellipsis",
            presentation_intensity=0.3,
            presentation_channel="internal",
            expression_mode="silent",
            expression_suppressed=True,
            interrupt_candidates_count=1,
            suppressed_interrupt_count=1,
            allowed_interrupt_type=None,
            internal_event_processed=False,
            stop_reason=None,
            stopped_early=False,
            summary_line="",
            metadata={"x": 1, "y": None},
        )
        raw = snap.to_dict()
        json.dumps(raw)
        rebuilt = WatchSnapshot(**raw)
        self.assertEqual(rebuilt.tick_index, 1)
        self.assertEqual(rebuilt.metadata["x"], 1)


class WatchRendererUnitTests(unittest.TestCase):
    def test_compact_renderer_contains_key_fields(self) -> None:
        cfg = WatchConfig()
        snap = WatchSnapshot(
            tick_index=3,
            tick_count=10,
            timestamp="",
            decision="wait",
            system_pressure=0.38,
            latent_pressure=0.22,
            presentation_mode="thinking",
            presentation_motion="ellipsis",
            presentation_intensity=0.42,
            presentation_channel="internal",
            expression_mode="silent",
            expression_suppressed=True,
            interrupt_candidates_count=1,
            suppressed_interrupt_count=1,
            allowed_interrupt_type=None,
            internal_event_processed=False,
            stop_reason=None,
            stopped_early=False,
            summary_line="",
            metadata={
                "top_latent_entry": {"entry_type": "x", "id": "y"},
                "expression_reasons": ["a", "b"],
                "expression_suppression_reason": "s",
                "allowed_interrupt_reason": "r",
            },
        )

        line = render_watch_snapshot(snap, config=cfg)
        self.assertIn("tick 3/10", line)
        self.assertIn("mode=thinking", line)
        self.assertIn("pressure=0.38/0.22", line)
        self.assertIn("expression=silent", line)
        self.assertIn("suppressed=true", line)
        self.assertIn("interrupts=1/1", line)
        self.assertIn("allowed=none", line)
        self.assertIn("decision=wait", line)
        self.assertNotIn("\n", line)

        trace = render_watch_trace(snap)
        self.assertIsNotNone(trace)
        self.assertIn("trace:", trace or "")


class WatchCLITests(unittest.TestCase):
    def test_watch_runs_multiple_ticks(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "3",
                    "--watch-interval",
                    "0",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        lines = [ln for ln in stdout.getvalue().splitlines() if ln.startswith("tick ")]
        self.assertEqual(len(lines), 3)

    def test_watch_ticks_controls_count(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "5",
                    "--watch-interval",
                    "0",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        lines = [ln for ln in stdout.getvalue().splitlines() if ln.startswith("tick ")]
        self.assertEqual(len(lines), 5)

    def test_watch_json_emits_valid_payload(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "3",
                    "--watch-interval",
                    "0",
                    "--watch-json",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("watch_run", payload)
        wr = payload["watch_run"]
        self.assertEqual(wr["tick_count"], 3)
        self.assertFalse(wr["stopped_early"])
        self.assertIsNone(wr["stop_reason"])
        self.assertEqual(len(wr["snapshots"]), 3)

        snap0 = wr["snapshots"][0]
        for k in (
            "tick_index",
            "tick_count",
            "timestamp",
            "decision",
            "system_pressure",
            "latent_pressure",
            "presentation_mode",
            "presentation_motion",
            "presentation_intensity",
            "presentation_channel",
            "expression_mode",
            "expression_suppressed",
            "interrupt_candidates_count",
            "suppressed_interrupt_count",
            "allowed_interrupt_type",
            "internal_event_processed",
            "stop_reason",
            "stopped_early",
            "summary_line",
            "metadata",
        ):
            self.assertIn(k, snap0)

    def test_watch_clear_includes_ansi_and_preserves_ticks(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "2",
                    "--watch-interval",
                    "0",
                    "--watch-clear",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        out = stdout.getvalue()
        self.assertIn("\033[2J", out)
        self.assertGreaterEqual(out.count("tick "), 2)

    def test_watch_trace_includes_trace_fragments(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "2",
                    "--watch-interval",
                    "0",
                    "--watch-trace",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        out = stdout.getvalue()
        self.assertIn("trace:", out)

    def test_watch_honors_early_stop(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "12",
                    "--watch-interval",
                    "0",
                    "--watch-json",
                    "--pressure",
                    "0.99",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        wr = payload["watch_run"]
        self.assertTrue(wr["stopped_early"])
        self.assertEqual(wr["stop_reason"], "consecutive_high_system_pressure")
        # Same stop rule as manual tick runner (5 ticks at pressure 0.99).
        self.assertEqual(wr["tick_count"], 5)
        self.assertEqual(len(wr["snapshots"]), 5)

    def test_watch_includes_presentation_vector_data(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "2",
                    "--watch-interval",
                    "0",
                    "--watch-json",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        snap0 = payload["watch_run"]["snapshots"][0]
        self.assertIsInstance(snap0["presentation_mode"], str)
        self.assertIsInstance(snap0["presentation_motion"], str)
        self.assertIsInstance(snap0["presentation_intensity"], (int, float))
        self.assertIsInstance(snap0["presentation_channel"], str)

    def test_watch_rejects_model(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        with self.assertRaises(SystemExit) as ctx:
            cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--model",
                    "ollama:gemma3:4b",
                    "--watch-ticks",
                    "1",
                    "--watch-interval",
                    "0",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_watch_does_not_print_expression_prose(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--watch",
                    "--watch-ticks",
                    "2",
                    "--watch-interval",
                    "0",
                    "--pressure",
                    "0.1",
                    "--state-dir",
                    str(root),
                ]
            )

        self.assertEqual(code, 0)
        out = stdout.getvalue()
        # Watch mode is state-only: no response/tool prose headers.
        self.assertNotIn("response:", out)
        self.assertNotIn("tool:", out)
        self.assertNotIn("I need a bit more context", out)


if __name__ == "__main__":
    unittest.main()


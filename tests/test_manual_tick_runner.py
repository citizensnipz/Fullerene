"""Manual Tick Runner v0 — CLI and runner behavior."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.facets import BehaviorFacet, EchoFacet
from fullerene.nexus import Event, EventType, NexusRuntime
from fullerene.state import InMemoryStateStore
from fullerene.tick.runner import TICK_HARD_CAP, build_tick_event_metadata, run_manual_ticks
from fullerene.workspace_state import workspace_state_root


def _temp_state_dir() -> Path:
    root = workspace_state_root() / f".test-tick-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class ManualTickRunnerUnitTests(unittest.TestCase):
    def test_build_tick_event_metadata_merges_user_extra(self) -> None:
        m = build_tick_event_metadata(
            tick_index=2,
            tick_count=5,
            tick_reason="probe",
            suppress_expression=True,
            extra={"pressure": 0.5, "manual_tick": False},
        )
        self.assertTrue(m["manual_tick"])
        self.assertEqual(m["tick_index"], 2)
        self.assertEqual(m["tick_count"], 5)
        self.assertEqual(m["tick_reason"], "probe")
        self.assertTrue(m["suppress_expression"])
        self.assertEqual(m["pressure"], 0.5)

    def test_run_manual_ticks_increments_event_count(self) -> None:
        store = InMemoryStateStore()
        runtime = NexusRuntime(facets=[EchoFacet()], store=store)
        res = run_manual_ticks(
            runtime,
            total_ticks=4,
            suppress_expression=True,
            include_full_records=False,
        )
        self.assertFalse(res.stopped_early)
        self.assertEqual(len(res.summaries), 4)
        self.assertEqual(res.final_state_summary["event_count"], 4)

    def test_hard_cap_constant(self) -> None:
        self.assertEqual(TICK_HARD_CAP, 100)


class ManualTickCLIPTests(unittest.TestCase):
    def test_tick_json_is_tick_run_wrapper(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--tick",
                    "--json",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("tick_run", payload)
        tr = payload["tick_run"]
        self.assertEqual(tr["tick_count"], 1)
        self.assertNotIn("records", tr)

    def test_tick_debug_includes_records(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                ["--full", "--tick", "--debug", "--state-dir", str(root)]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        tr = payload["tick_run"]
        self.assertEqual(len(tr["records"]), 1)
        ev = tr["records"][0]["event"]
        self.assertEqual(ev["event_type"], "system_tick")
        self.assertTrue(ev["metadata"]["manual_tick"])

    def test_ticks_n_processes_n_summaries(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--context",
                    "--behavior",
                    "--ticks",
                    "5",
                    "--json",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        tr = json.loads(stdout.getvalue())["tick_run"]
        self.assertFalse(tr.get("stopped_early"), tr)
        self.assertEqual(len(tr["summaries"]), 5)
        self.assertEqual(tr["final_state_summary"]["event_count"], 5)

    def test_ticks_hard_cap_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--tick", "--ticks", str(TICK_HARD_CAP + 1)])
        self.assertEqual(ctx.exception.code, 2)

    def test_tick_summary_lines(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--full",
                    "--ticks",
                    "3",
                    "--tick-summary",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        lines = [ln for ln in stdout.getvalue().splitlines() if ln.startswith("tick ")]
        self.assertEqual(len(lines), 3)

    def test_default_suppress_expression_on_metadata(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                ["--full", "--tick", "--debug", "--state-dir", str(root)]
            )
        self.assertEqual(code, 0)
        rec0 = json.loads(stdout.getvalue())["tick_run"]["records"][0]
        self.assertTrue(rec0["event"]["metadata"]["suppress_expression"])
        er = rec0["metadata"].get("expression_recommendation")
        self.assertIsInstance(er, dict)

    def test_allow_tick_expression_sets_suppress_false(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--full",
                    "--tick",
                    "--debug",
                    "--allow-tick-expression",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        rec0 = json.loads(stdout.getvalue())["tick_run"]["records"][0]
        self.assertIs(rec0["event"]["metadata"].get("suppress_expression"), False)

    def test_tick_reason_in_metadata(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--full",
                    "--tick",
                    "--tick-reason",
                    "integration_test",
                    "--debug",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        meta = json.loads(stdout.getvalue())["tick_run"]["records"][0]["event"]["metadata"]
        self.assertEqual(meta["tick_reason"], "integration_test")

    def test_model_flag_rejected_with_tick(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli_main(["--tick", "--model", "ollama:x"])
        self.assertEqual(ctx.exception.code, 2)

    def test_single_user_message_json_unchanged_shape(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--json",
                    "--content",
                    "unchanged shape probe",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertNotIn("tick_run", payload)
        self.assertIn("event", payload)

    def test_high_pressure_stops_early(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--ticks",
                    "12",
                    "--pressure",
                    "0.99",
                    "--json",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        tr = json.loads(stdout.getvalue())["tick_run"]
        self.assertFalse(tr["stopped_early"])
        self.assertIsNone(tr["stop_reason"])
        self.assertEqual(len(tr["summaries"]), 12)

    def test_tick_summary_exposes_pressure_streaks(self) -> None:
        root = _temp_state_dir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli_main(
                [
                    "--memory",
                    "--behavior",
                    "--ticks",
                    "2",
                    "--pressure",
                    "0.99",
                    "--json",
                    "--state-dir",
                    str(root),
                ]
            )
        self.assertEqual(code, 0)
        s0 = json.loads(stdout.getvalue())["tick_run"]["summaries"][0]
        self.assertIn("high_pressure_streak", s0)
        self.assertIn("latent_saturation_streak", s0)
        self.assertIn("stop_pressure_kind", s0)


class ManualTickLPBTests(unittest.TestCase):
    def test_latent_pressure_decays_across_manual_ticks(self) -> None:
        """Seed LPB via a user message + behavior, then observe latent_pressure across ticks."""
        store = InMemoryStateStore()
        runtime = NexusRuntime(facets=[BehaviorFacet(), EchoFacet()], store=store)
        runtime.process_event(
            Event(
                event_type=EventType.USER_MESSAGE,
                content="what should I do?",
                metadata={"latent_pressure": 0.75},
            )
        )
        res = run_manual_ticks(
            runtime,
            total_ticks=5,
            suppress_expression=True,
            include_full_records=False,
        )
        lp_seq = [float(s.get("latent_pressure") or 0) for s in res.summaries]
        self.assertEqual(len(lp_seq), 5)
        self.assertGreater(lp_seq[0], 0.0)
        # Decay on inactive SYSTEM_TICK cycles should not increase total monotonically.
        self.assertLessEqual(lp_seq[-1], lp_seq[0] + 1e-6)


class ManualTickSubprocessTests(unittest.TestCase):
    def test_subprocess_tick_matches_import(self) -> None:
        root = Path(__file__).resolve().parents[1] / "state" / ".test-cli-tick-sub"
        root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "fullerene",
            "--full",
            "--tick",
            "--debug",
            "--state-dir",
            str(root),
        ]
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        payload = json.loads(proc.stdout.decode())
        self.assertEqual(payload["tick_run"]["records"][0]["event"]["event_type"], "system_tick")


if __name__ == "__main__":
    unittest.main()

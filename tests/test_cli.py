from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fullerene.cli import main as cli_main
from fullerene.workspace_state import DEFAULT_STATE_DIR


def make_tempdir_path() -> Path:
    return Path(tempfile.mkdtemp(prefix="fullerene-cli-test-"))


class CLIUsabilityTests(unittest.TestCase):
    def test_full_preset_enables_all_implemented_facets(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--full",
                    "--json",
                    "--content",
                    "What are you doing?",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())
        facet_names = {result["facet_name"] for result in payload["facet_results"]}

        self.assertEqual(exit_code, 0)
        self.assertTrue(
            {
                "memory",
                "context",
                "goals",
                "world_model",
                "behavior",
                "policy",
                "planner",
                "executor",
                "learning",
                "attention",
                "affect",
                "verifier",
                "echo",
            }.issubset(facet_names)
        )

    def test_default_output_is_concise_not_full_json(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--memory",
                    "--context",
                    "--content",
                    "test scoped facets",
                    "--state-dir",
                    str(root),
                ]
            )

        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("decision: RECORD", output)
        self.assertIn("response: null", output)
        self.assertIn("recorded: true", output)
        self.assertIn("reason: ", output)
        self.assertNotIn("facet_results", output)
        self.assertFalse(output.lstrip().startswith("{"))

    def test_json_and_debug_output_full_record(self) -> None:
        for output_flag in ("--json", "--debug"):
            with self.subTest(output_flag=output_flag):
                root = make_tempdir_path()
                self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
                stdout = io.StringIO()

                with redirect_stdout(stdout):
                    exit_code = cli_main(
                        [
                            "--full",
                            output_flag,
                            "--content",
                            "What are you doing?",
                            "--state-dir",
                            str(root),
                        ]
                    )

                payload = json.loads(stdout.getvalue())

                self.assertEqual(exit_code, 0)
                self.assertIn("event", payload)
                self.assertIn("facet_results", payload)
                self.assertIn("decision", payload)

    def test_positional_prompt_is_used_when_content_is_omitted(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--json",
                    "--memory",
                    "positional prompt input",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["event"]["content"], "positional prompt input")

    def test_state_directory_contains_bounded_known_files(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        for index in range(7):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--full",
                        "--content",
                        f"state file check {index}",
                        "--state-dir",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)

        state_entries = {path.name for path in root.iterdir()}
        snapshot_names = {
            path.name for path in (root / "snapshots").iterdir() if path.is_file()
        }

        self.assertEqual(DEFAULT_STATE_DIR, "state/.fullerene-state")
        self.assertLessEqual(
            state_entries,
            {
                "state.json",
                "runtime-log.jsonl",
                "memory.sqlite3",
                "goals.sqlite3",
                "world.sqlite3",
                "policy.sqlite3",
                "snapshots",
            },
        )
        self.assertEqual(
            snapshot_names,
            {
                "state-1.json",
                "state-2.json",
                "state-3.json",
                "state-4.json",
                "state-5.json",
            },
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import shutil
import urllib.error
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fullerene.cli import main as cli_main
from fullerene.workspace_state import DEFAULT_STATE_DIR, workspace_state_root


def make_tempdir_path() -> Path:
    root = workspace_state_root() / f".test-cli-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


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

    def test_full_default_output_renders_direct_question_response(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--full",
                    "--content",
                    "What are you doing right now?",
                    "--state-dir",
                    str(root),
                ]
            )

        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("decision: ACT", output)
        self.assertIn("tool: text", output)
        self.assertIn("response: ", output)
        self.assertNotIn("response: null", output)
        self.assertIn("local runtime cycle", output)

    def test_model_output_uses_ollama_text_after_behavior_decision(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with patch(
            "fullerene.models.ollama.OllamaAdapter.generate",
            return_value="Model phrasing only.",
        ) as generate:
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--full",
                        "--model",
                        "ollama:gemma3:4b",
                        "--content",
                        "What are you doing?",
                        "--state-dir",
                        str(root),
                    ]
                )

        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("decision: ACT", output)
        self.assertIn('response: "Model phrasing only."', output)
        generate.assert_called_once()
        prompt = generate.call_args.args[0]
        self.assertIn("System decision: ACT", prompt)
        self.assertIn("Only generate text", prompt)
        self.assertIn("Current working context:", prompt)
        self.assertIn("- current event: What are you doing?", prompt)
        self.assertIn("- active goals: none", prompt)
        self.assertIn("- recent memories: none", prompt)
        self.assertIn("- active beliefs: none", prompt)
        self.assertIn("- query intent: status_request", prompt)
        self.assertIn("- missing context: none", prompt)

    def test_model_offline_falls_back_to_template(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with patch(
            "fullerene.models.ollama.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--full",
                        "--model",
                        "ollama:gemma3:4b",
                        "--content",
                        "What are you doing?",
                        "--state-dir",
                        str(root),
                    ]
                )

        output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("decision: ACT", output)
        self.assertIn("response: ", output)
        self.assertIn("local runtime cycle", output)

    def test_model_is_not_used_for_record_decision_logic(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with patch(
            "fullerene.models.ollama.OllamaAdapter.generate",
            return_value="Should not be used.",
        ) as generate:
            with redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--memory",
                        "--model",
                        "ollama:gemma3:4b",
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
        generate.assert_not_called()

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

    def test_debug_direct_question_still_outputs_full_record(self) -> None:
        root = make_tempdir_path()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "--full",
                    "--debug",
                    "--content",
                    "What are you doing right now?",
                    "--state-dir",
                    str(root),
                ]
            )

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["decision"]["action"], "act")
        self.assertIn("facet_results", payload)

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
                "memory.sqlite3-journal",
                "goals.sqlite3",
                "goals.sqlite3-journal",
                "world.sqlite3",
                "world.sqlite3-journal",
                "policy.sqlite3",
                "policy.sqlite3-journal",
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

# Commands - install, run, test

Agents should prefer commands documented here.

## Setup

```bash
# No bootstrap step is required yet beyond a supported local Python.
```

## Run (development)

```bash
python -m fullerene --help
python -m fullerene --event-type user_message --content "hello nexus" --state-dir state/.fullerene-state
python -m fullerene --memory --content "hello memory"
python -m fullerene --memory --behavior --content "don't ever skip my boss emails"
python -m fullerene --behavior --content "what should I do next?"
python -m fullerene --full --latent-pressure --json --content "what should I do next?"
python -m fullerene --expression-gate --verify --json --content "inspect expression recommendation"
```

`--json` / `--debug` emit the full `NexusRecord`, including Nexus v2 `interrupt_candidates`, `suppression_decisions`, Expression Gate recommendation fields (`expression_recommendation`, `expression_score`, `expression_mode`, etc.), and related `cycle_trace` fields when present.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Lint and format

```bash
# No formatter/linter command is standardized yet.
```

## Model backends

```bash
python -m fullerene --full --model ollama:gemma3:4b --content "What are you doing?"
```

## For AI agents

- Prefer commands documented in this file.
- If tooling is added without updating this file, add the commands in the same change.

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

Manual Tick Runner v0 (explicit `SYSTEM_TICK` cycles — not a daemon or watch UI):

```bash
python -m fullerene --full --tick --debug --state-dir state/.smoke-tick
python -m fullerene --full --tick --presentation --state-dir state/.smoke-tick
python -m fullerene --full --ticks 5 --tick-summary --presentation --state-dir state/.smoke-tick
python -m fullerene --full --ticks 3 --json --presentation --state-dir state/.smoke-tick
```

- `--tick` runs one `SYSTEM_TICK` (combine with `--ticks N` for a sequence; default cap **100**).
- Default tick metadata sets `suppress_expression`; use `--allow-tick-expression` to opt out.
- `--json` / `--debug` emit `{ "tick_run": … }` for manual ticks; full per-tick `records` appear with **`--debug`** (or use **`--json` together with `--debug`** for full nested records).
- **`--presentation`** emits compact Presentation Vector lines and embeds **`presentation_vector`** in each tick **`summaries`** row when **`--presentation`** or **`--tick-summary`** is active.

`--json` / `--debug` emit the full `NexusRecord`, including Nexus v2 `interrupt_candidates`, `suppression_decisions`, Expression Gate recommendation fields (`expression_recommendation`, `expression_score`, `expression_mode`, etc.), and related `cycle_trace` fields when present.

Watch Mode v0 (bounded, terminal snapshots):

```bash
python -m fullerene --full --watch
python -m fullerene --full --watch --watch-ticks 20 --watch-interval 0.5
python -m fullerene --full --watch --watch-json
python -m fullerene --full --watch --watch-clear
```

- `--watch` runs bounded manual `SYSTEM_TICK` cycles and renders compact snapshots (no `--tick` required).
- `--watch-ticks N` controls the number of ticks (clamped).
- `--watch-interval SECONDS` sleeps between rendered ticks; `0` disables sleeping.
- `--watch-clear` clears the screen between renders.
- `--watch-trace` adds compact `trace:` fragments when available.
- `--watch-json` emits `{ "watch_run": ... }` JSON output only.
- Watch Mode v0 stops when the Manual Tick Runner stop conditions trigger, and reports the `stop_reason`.

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

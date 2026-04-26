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
```

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
# Future: model backend commands will be added after model integration lands.
```

## For AI agents

- Prefer commands documented in this file.
- If tooling is added without updating this file, add the commands in the same change.

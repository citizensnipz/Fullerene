# Commands - install, run, test

Agents should prefer commands documented here.

## Setup

```bash
# No bootstrap step is required yet beyond a supported local Python.
```

## Run (development)

```bash
python -m fullerene --help
python -m fullerene --event-type user_message --content "hello nexus" --state-dir .fullerene-state
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Lint and format

```bash
# TBD
```

## Model backends

```bash
# Not wired yet.
```

## For AI agents

- Prefer commands documented in this file.
- If tooling is added without updating this file, add the commands in the same change.

# Convenience targets. Each one is a thin wrapper over a command CI also runs,
# so "it passed make" and "it passed CI" mean the same thing.

SERVER_SPEC ?= ../lucenia/rust/api/openapi.yaml

.PHONY: models vendor check test conformance

# Regenerate the models from the vendored description (pinned generator).
models:
	scripts/regen-models.sh

# Copy the server's description in byte for byte, then regenerate.
vendor:
	cp $(SERVER_SPEC) src/gnarl/openapi.yaml
	scripts/regen-models.sh

check:
	ruff check .
	mypy

test:
	pytest

conformance:
	pytest tests/conformance -v

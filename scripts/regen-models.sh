#!/usr/bin/env bash
# Regenerate src/gnarl/_models.py from the vendored description.
#
# THE one regeneration command. CI's spec-drift job runs this script and diffs
# the result, so a model generated any other way is a model CI will reject.
#
# The generator and its formatters are pinned in requirements-codegen.txt and
# installed into their own virtualenv (.venv-codegen), so regeneration does not
# depend on whatever happens to be installed alongside the client.
#
#   scripts/regen-models.sh            # regenerate
#   PYTHON=python3.12 scripts/regen-models.sh
#   CODEGEN=datamodel-codegen scripts/regen-models.sh   # use one already on PATH
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [ -z "${CODEGEN:-}" ]; then
  venv="$root/.venv-codegen"
  if [ ! -x "$venv/bin/datamodel-codegen" ] \
     || ! cmp -s requirements-codegen.txt "$venv/.requirements"; then
    "${PYTHON:-python3}" -m venv "$venv"
    "$venv/bin/pip" install --quiet --disable-pip-version-check -r requirements-codegen.txt
    cp requirements-codegen.txt "$venv/.requirements"
  fi
  CODEGEN="$venv/bin/datamodel-codegen"
fi

# Every flag is load-bearing; see "How this package is built" in README.md.
#   --openapi-scopes schemas paths   inline request/response shapes (entitlement,
#                                    memory, namespaces, schedules) get models
#                                    too, not just the named components
#   --disable-timestamp              the output is a pure function of the input,
#                                    so CI can compare it byte for byte
#   --formatters black isort         explicit, because the generator's default
#                                    is announced to change
"$CODEGEN" \
  --input src/gnarl/openapi.yaml --input-file-type openapi \
  --openapi-scopes schemas paths \
  --output src/gnarl/_models.py --output-model-type pydantic_v2.BaseModel \
  --target-python-version 3.10 --use-standard-collections --use-union-operator \
  --field-constraints --use-schema-description --collapse-root-models \
  --deserialize-default-values enum --use-default-kwarg \
  --disable-timestamp --formatters black isort

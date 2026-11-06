#!/usr/bin/env bash
#
# CI entrypoint: run every check/test that must pass before merging/publishing.
#
# Usage:
#   scripts/ci.sh            # run all checks
#   scripts/ci.sh <name>...  # run only the named checks, e.g. scripts/ci.sh pytest reuse
#
# Exit code is non-zero if any check fails. Each check is run in a subshell so
# a failure in one does not abort the others.
#
# SPDX-FileCopyrightText: 2026 DLR e.V. (German Aerospace Center)
# SPDX-License-Identifier: Apache-2.0

set -u

# Resolve the repo root (parent of this script's directory).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the project virtualenv if present, else whatever is on PATH.
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
    BIN="$ROOT/.venv/bin"
else
    PY="python"
    BIN=""
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0
FAILED_NAMES=()

# run <name> <command...>
run() {
    local name="$1"
    shift
    echo
    echo "=================================================================="
    echo "  $name"
    echo "=================================================================="
    if "$@"; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
        FAILED_NAMES+=("$name")
    fi
}

# run_module <name> <module> <args...>  -- run a tool as a Python module.
#
# Tools are invoked as "$PY -m <module>" rather than by name so that they are
# always resolved from the project virtualenv. Invoking them by bare name can
# pick up unrelated shims earlier on PATH (e.g. stale ~/.local/bin wrappers
# pointing at a system interpreter that no longer has the package installed).
run_module() {
    local name="$1"
    local module="$2"
    shift 2
    if ! "$PY" -c "import $module" >/dev/null 2>&1; then
        echo
        echo "  [SKIP] $name (module '$module' not installed)"
        return
    fi
    run "$name" "$PY" -m "$module" "$@"
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
check_pytest() {
    run "pytest (unit tests)" "$PY" -m pytest
}

check_reuse() {
    run_module "reuse lint (SPDX headers)" reuse lint
}

check_flake8() {
    run_module "flake8 (style + unused imports)" flake8 eduplane tests
}

check_yapf() {
    run_module "yapf (formatting check)" yapf --diff --recursive eduplane tests
}

check_isort() {
    run_module "isort (import order check)" isort --check-only eduplane tests
}

check_docs() {
    run "sphinx (docs build, warnings as errors)" \
        "$PY" -m sphinx -b html -W --keep-going docs/source /tmp/eduplane-docs
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
ALL_CHECKS=(reuse flake8 yapf isort docs pytest)

if [ "$#" -eq 0 ]; then
    SELECTED=("${ALL_CHECKS[@]}")
else
    SELECTED=("$@")
fi

for name in "${SELECTED[@]}"; do
    case "$name" in
        reuse)    check_reuse ;;
        flake8)   check_flake8 ;;
        yapf)     check_yapf ;;
        isort)    check_isort ;;
        docs)     check_docs ;;
        pytest)   check_pytest ;;
        *)
            echo "Unknown check: $name" >&2
            echo "Available: ${ALL_CHECKS[*]}" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "=================================================================="
echo "  Summary: $PASS passed, $FAIL failed"
echo "=================================================================="
if [ "$FAIL" -gt 0 ]; then
    printf '  Failed: %s\n' "${FAILED_NAMES[@]}"
    exit 1
fi
exit 0
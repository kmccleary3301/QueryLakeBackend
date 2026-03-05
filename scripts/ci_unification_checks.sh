#!/usr/bin/env bash
set -euo pipefail

python scripts/verify_compat_pins.py
bash scripts/ci_guard_legacy_querylakebackend_refs.sh

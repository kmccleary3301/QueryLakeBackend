#!/usr/bin/env bash
set -euo pipefail

# QueryLake driver upgrade preflight (read-only).
# This script is intentionally non-destructive: it does NOT install packages.
#
# Usage:
#   bash scripts/driver_upgrade_preflight.sh

echo "== Host kernel =="
uname -a || true
echo

echo "== NVIDIA state (nvidia-smi) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi
echo

echo "== Current NVIDIA driver version (procfs) =="
cat /proc/driver/nvidia/version 2>/dev/null || echo "No /proc/driver/nvidia/version"
echo

echo "== Secure Boot state (may affect DKMS module load) =="
if command -v mokutil >/dev/null 2>&1; then
  mokutil --sb-state || true
else
  echo "mokutil not installed"
fi
echo

echo "== Active GPU compute processes (best effort) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi pmon -c 1 2>/dev/null || true
fi
echo

echo "== Ubuntu driver recommendation (read-only) =="
if command -v ubuntu-drivers >/dev/null 2>&1; then
  ubuntu-drivers devices || true
else
  echo "ubuntu-drivers not installed"
fi
echo

echo "== Candidate driver packages (apt-cache policy) =="
if command -v apt-cache >/dev/null 2>&1; then
  apt-cache policy nvidia-driver-550 nvidia-driver-570 nvidia-driver-575 2>/dev/null || true
else
  echo "apt-cache not found"
fi
echo

echo "== DKMS + headers presence (best effort) =="
dpkg -l | awk '/^ii/ && ($2 ~ /dkms/ || $2 ~ /^linux-headers/){print $2, $3}' | sed -n '1,80p' || true
echo

echo "DONE: preflight complete (no changes were made)."


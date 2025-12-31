#!/usr/bin/env bash
set -euo pipefail

# QueryLake NVIDIA driver upgrade helper (Ubuntu 22.04+)
#
# Goal: streamline a safe upgrade to nvidia-driver-580 with an operator-controlled reboot.
#
# Usage:
#   bash scripts/driver_upgrade_580.sh --install
#   bash scripts/driver_upgrade_580.sh --post-reboot
#
# Notes:
# - This script intentionally does NOT stop any processes for you. You must stop your own
#   GPU workloads first (and avoid global Ray shutdowns like `ray stop`).
# - A reboot is required after installing the new driver.

TARGET_SERIES="580"
TARGET_PACKAGE="nvidia-driver-${TARGET_SERIES}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/driver_upgrade_${TARGET_SERIES}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PREF_FILE="/etc/apt/preferences.d/querylake-prefer-ubuntu-nvidia.pref"

say() { printf "\n== %s ==\n" "$*"; }
warn() { printf "\nWARN: %s\n" "$*" >&2; }
die() { printf "\nERROR: %s\n" "$*" >&2; exit 1; }

as_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found (run as root or install sudo)"
    sudo -H "$@"
  else
    "$@"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

prompt_yes_no() {
  local prompt="${1}"
  local default="${2:-N}" # Y/N
  local answer=""

  while true; do
    if [[ "${default}" == "Y" ]]; then
      read -r -p "${prompt} [Y/n] " answer || true
      answer="${answer:-y}"
    else
      read -r -p "${prompt} [y/N] " answer || true
      answer="${answer:-n}"
    fi
    case "${answer}" in
      y|Y) return 0 ;;
      n|N) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

show_usage() {
  cat <<EOF
QueryLake driver upgrade helper (${TARGET_PACKAGE})

Logs to: ${LOG_FILE}

Usage:
  bash scripts/driver_upgrade_580.sh --install
  bash scripts/driver_upgrade_580.sh --post-reboot
  bash scripts/driver_upgrade_580.sh --preflight

What it does:
  --preflight    Read-only checks (GPU state, Secure Boot, apt candidates).
  --install      Installs prerequisites + ${TARGET_PACKAGE}; prompts before changes; offers to write an apt pin.
  --post-reboot  Verifies the driver is active after reboot; optional torch CUDA probe if conda env exists.
EOF
}

preflight() {
  say "Log file"
  echo "${LOG_FILE}"

  say "Host OS"
  if [[ -f /etc/os-release ]]; then
    cat /etc/os-release || true
  else
    echo "No /etc/os-release found"
  fi

  say "Kernel"
  uname -a || true

  say "NVIDIA state (nvidia-smi)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi

  say "Current NVIDIA driver version (procfs)"
  cat /proc/driver/nvidia/version 2>/dev/null || echo "No /proc/driver/nvidia/version"

  say "Secure Boot state (may affect DKMS module load)"
  if command -v mokutil >/dev/null 2>&1; then
    mokutil --sb-state || true
  else
    echo "mokutil not installed"
  fi

  say "Active GPU processes (best effort)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi pmon -c 1 2>/dev/null || true
    echo
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
  fi

  say "Ubuntu driver recommendation (ubuntu-drivers)"
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    ubuntu-drivers devices || true
  else
    echo "ubuntu-drivers not installed"
  fi

  say "APT candidates (apt-cache policy)"
  if command -v apt-cache >/dev/null 2>&1; then
    apt-cache policy "nvidia-driver-${TARGET_SERIES}" nvidia-driver-550 nvidia-driver-570 nvidia-driver-575 2>/dev/null || true
  else
    echo "apt-cache not found"
  fi

  say "DONE: preflight complete (no changes were made)."
}

write_pin_prefer_ubuntu() {
  say "Writing apt pin (prefer Ubuntu for NVIDIA driver packages)"
  echo "Target: ${PREF_FILE}"
  echo
  echo "This is intended to avoid third-party repos (e.g. LambdaLabs) winning package selection for NVIDIA drivers."
  echo "It only pins NVIDIA driver-related packages."
  echo

  local tmpfile
  tmpfile="$(mktemp)"
  cat >"${tmpfile}" <<EOF
# QueryLake: prefer Ubuntu-origin NVIDIA driver packages.
# Created by: scripts/driver_upgrade_580.sh
#
# Remove this file if you later want third-party repos to take precedence again.

Package: nvidia-driver-${TARGET_SERIES} nvidia-dkms-${TARGET_SERIES} nvidia-kernel-common-${TARGET_SERIES} nvidia-kernel-source-${TARGET_SERIES} xserver-xorg-video-nvidia-${TARGET_SERIES} nvidia-utils-${TARGET_SERIES} nvidia-compute-utils-${TARGET_SERIES} libnvidia-*
Pin: release o=Ubuntu
Pin-Priority: 1002
EOF

  as_root install -m 0644 "${tmpfile}" "${PREF_FILE}"
  rm -f "${tmpfile}"

  say "APT pin installed"
  as_root ls -la "${PREF_FILE}" || true
}

install_driver() {
  need_cmd apt-get
  need_cmd apt-cache

  say "Preflight (read-only)"
  preflight

  say "Safety check"
  echo "You should stop any GPU workloads you care about before proceeding."
  echo "Do NOT run a global Ray shutdown (e.g. 'ray stop') on this machine."
  if ! prompt_yes_no "Confirm you have stopped your GPU workloads and are ready to install ${TARGET_PACKAGE}?" "N"; then
    die "Aborted by operator."
  fi

  if command -v mokutil >/dev/null 2>&1; then
    if mokutil --sb-state 2>/dev/null | grep -qi "enabled"; then
      warn "Secure Boot appears ENABLED. DKMS driver module signing may be required."
      if ! prompt_yes_no "Proceed anyway (you may need MOK enrollment)?" "N"; then
        die "Aborted due to Secure Boot being enabled."
      fi
    fi
  fi

  say "APT candidate for ${TARGET_PACKAGE}"
  apt-cache policy "${TARGET_PACKAGE}" 2>/dev/null || true

  if prompt_yes_no "Install (or refresh) an apt pin to prefer Ubuntu-origin NVIDIA packages?" "Y"; then
    write_pin_prefer_ubuntu
    say "APT candidates after pin (for ${TARGET_PACKAGE} and dependencies)"
    apt-cache policy \
      "${TARGET_PACKAGE}" \
      "nvidia-utils-${TARGET_SERIES}" \
      "nvidia-compute-utils-${TARGET_SERIES}" \
      2>/dev/null || true
  else
    warn "No apt pin created. If a third-party repo has higher priority, it may win selection."
  fi

  say "Dry-run install (simulation)"
  echo "Showing apt's planned actions for: ${TARGET_PACKAGE}"
  echo
  as_root env DEBIAN_FRONTEND=noninteractive apt-get -s install -y "${TARGET_PACKAGE}" || true
  echo
  if ! prompt_yes_no "Proceed with REAL installation of ${TARGET_PACKAGE}?" "N"; then
    die "Aborted by operator."
  fi

  say "Install prerequisites"
  as_root env DEBIAN_FRONTEND=noninteractive apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    dkms \
    "linux-headers-$(uname -r)" \
    ubuntu-drivers-common

  say "Install ${TARGET_PACKAGE}"
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y "${TARGET_PACKAGE}"

  say "Install complete"
  echo "A reboot is required for the new driver module to load."
  echo "After reboot, run:"
  echo "  bash scripts/driver_upgrade_580.sh --post-reboot"
  echo

  if prompt_yes_no "Reboot now?" "N"; then
    say "Rebooting"
    as_root reboot
  else
    warn "Operator chose not to reboot now. Driver will not take effect until reboot."
  fi
}

post_reboot_verify() {
  say "Post-reboot verification"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found; driver may not be installed/loaded."
  fi

  nvidia-smi || true

  local driver_version
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  if [[ -z "${driver_version}" ]]; then
    warn "Could not read driver version via nvidia-smi."
  elif [[ "${driver_version}" == ${TARGET_SERIES}* ]]; then
    echo "OK: driver_version=${driver_version} (matches ${TARGET_SERIES}.x)"
  else
    warn "Driver version does not look like ${TARGET_SERIES}.x: ${driver_version}"
  fi

  say "procfs driver version"
  cat /proc/driver/nvidia/version 2>/dev/null || true

  say "Optional torch CUDA probe (if conda env QL_1 exists)"
  if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -Eq "^QL_1[[:space:]]"; then
      conda run -n QL_1 python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY
    else
      echo "Conda env QL_1 not found; skipping torch probe."
    fi
  else
    echo "conda not available; skipping torch probe."
  fi

  say "DONE: post-reboot verification finished"
}

main() {
  if [[ "${#}" -eq 0 ]]; then
    show_usage
    exit 2
  fi

  case "${1}" in
    --preflight)
      preflight
      ;;
    --install)
      install_driver
      ;;
    --post-reboot)
      post_reboot_verify
      ;;
    -h|--help)
      show_usage
      ;;
    *)
      show_usage
      die "Unknown argument: ${1}"
      ;;
  esac
}

main "$@"

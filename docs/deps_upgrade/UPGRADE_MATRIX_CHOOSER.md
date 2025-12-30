# QueryLake Dependencies Upgrade — Matrix Chooser (Torch 2.9 + vLLM 0.13)
**Audience:** operator/maintainer  
**Goal:** choose a GPU stack + runtime matrix that unlocks PyTorch 2.9 and modern vLLM, with a clean rollback path.  
**Hardware:** 2× RTX 6000 Ada (48GB)  

---

## Current decision (as of 2025-12-30)

- **Preferred matrix:** Matrix A (Aggressive / driver 575.x+)
- **Blocked by:** a required reboot for the driver upgrade
- **Operator constraint:** do **not** reboot this system until a maintenance window is scheduled.

This means we can proceed now with all reboot-free work:

- implement the QueryLake ↔ vLLM decoupling (HTTP boundary)
- build reproducible env definitions/lockfiles
- prepare driver upgrade preflight checks + rollback steps

---

## 0) Non-negotiables (why this is not “just pip install”)

1. **NVIDIA driver step-up is required** on this host.
   - Current host: NVIDIA driver `550.107.02`, CUDA `12.4` (from `nvidia-smi`).
2. **Do not upgrade in-place** inside `QL_1`.
   - Build fresh envs and cut over via routing/config.
3. **Do not keep subclassing vLLM internals** for the long term.
   - Treat vLLM as an OpenAI-compatible HTTP service boundary whenever possible.

---

## 1) Quick decision (recommended default)

If you want the most “bleeding-edge but still operationally sane” path on Ubuntu 22.04:

**Choose Matrix A (Driver 575.x+)** and run torch 2.9 + vLLM 0.13 in a **separate runtime env/service**.

Why:
- Driver 575.x meets CUDA 12.9-era minimums and keeps headroom for newer CUDA 12.x/13.x stacks.
- This reduces future rework when vLLM changes its default CUDA build.

If you want the lowest-risk driver step (still modern):

**Choose Matrix B (Driver 570.x+)** and target the CUDA 12.8 ecosystem.

---

## 2) Matrices (pick one)

### Matrix A — Aggressive (recommended default for “bleeding edge”)

- **Driver target:** `>= 575.57.08` (CUDA 12.9 Update 1 minimum; confirm before executing)
- **Torch:** `2.9.0` (CUDA 12.8/12.9/13.0 variant as required by your vLLM wheel selection)
- **vLLM:** `0.13.0`
- **Python:** 3.11 or 3.12 (separate env from QueryLake API)

Use when:
- You want the highest likelihood that “latest vLLM wheels just work”.
- You’re OK with a driver upgrade that is more likely to be “new”.

### Matrix B — Safer-but-modern

- **Driver target:** `>= 570.26` (CUDA 12.8 minimum; confirm before executing)
- **Torch:** `2.9.0` (CUDA 12.8)
- **vLLM:** `0.13.0` (CUDA 12.8 build/variant if applicable)
- **Python:** 3.11 or 3.12

Use when:
- You want the smallest driver jump that still unlocks torch 2.9-class stacks.

---

## 3) Driver upgrade runbook (Ubuntu 22.04) — safe and rollbackable

### 3.1 Preconditions

1. Ensure you have console access or a proven remote-access plan (driver upgrades can affect display/SSH in edge cases).
2. Identify and stop only **your** GPU-serving workloads (do not run `ray stop`).
3. Record current state:

```bash
nvidia-smi
uname -r
cat /proc/driver/nvidia/version || true
```

4. Secure Boot check (if enabled, plan MOK enrollment or disable Secure Boot):

```bash
mokutil --sb-state || true
```

### 3.2 Install prerequisites

```bash
sudo apt-get update
sudo apt-get install -y build-essential dkms "linux-headers-$(uname -r)" ubuntu-drivers-common
```

### 3.3 Choose driver package

List what Ubuntu recommends:

```bash
ubuntu-drivers devices
```

Then install one of:

- Matrix A: `nvidia-driver-575` (or whatever 575-series metapackage Ubuntu recommends)
- Matrix B: `nvidia-driver-570` (or whatever 570-series metapackage Ubuntu recommends)

Example:

```bash
sudo apt-get install -y nvidia-driver-575
sudo reboot
```

### 3.4 Post-reboot verification

```bash
nvidia-smi
cat /proc/driver/nvidia/version || true
```

### 3.5 Rollback (if needed)

If the new driver fails to load or causes regressions, roll back to the known-good driver series:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550
sudo reboot
```

---

## 4) After driver upgrade: what to verify before touching QueryLake

1. Torch can see GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

2. vLLM runtime env can start a standalone server (do this in a fresh env; do not reuse `QL_1`).

---

## 5) Notes on CUDA “tracks” (what to validate explicitly)

Because vLLM and torch are distributed as binary wheels, it is essential to verify that:

1. The **vLLM wheel variant** you install is compatible with the **torch CUDA wheel variant** you install.
2. Your **driver** satisfies the CUDA runtime minimum required by those wheels.

**Verification step (recommended):** after installing torch+vLLM in the new env, run:

```bash
python -c "import vllm, torch; print('vllm', getattr(vllm,'__version__','?')); print('torch', torch.__version__)"
python -c "from vllm.engine.async_llm_engine import AsyncLLMEngine; print('AsyncLLMEngine import OK')"
vllm --help >/dev/null
```

If anything fails here, do not proceed to QueryLake integration until the runtime env is clean.

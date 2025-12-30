# QueryLake Dependencies Upgrade — Gates / Issue List
**Goal:** upgrade toward a modern stack (Torch 2.9 + vLLM 0.13) without destabilizing the QueryLake API or shared Ray workloads.  
**Principle:** no in-place upgrades; ship via fresh envs + explicit routing/cutover.

---

## Gate 0 — Choose matrix + schedule downtime

- [ ] Choose Matrix A or B in `docs/deps_upgrade/UPGRADE_MATRIX_CHOOSER.md`.
- [ ] Decide the operational constraint: dedicated maintenance window (driver upgrade requires reboot).
- [ ] Identify an “owner” for the driver step (human operator).
- [ ] Record current state for rollback:
  - [ ] `nvidia-smi` output
  - [ ] `cat /proc/driver/nvidia/version`
  - [ ] `uname -r`

**Done when:** matrix is selected and a maintenance window exists.

---

## Gate 1 — Driver upgrade completed + validated

- [ ] Install new driver series per matrix (Ubuntu packages preferred).
- [ ] Reboot.
- [ ] Validate:
  - [ ] `nvidia-smi` shows the expected driver series
  - [ ] CUDA visible in `nvidia-smi` output
  - [ ] A trivial torch CUDA check succeeds (in any clean env)

**Done when:** GPU is usable and driver rollback steps are documented for this host.

---

## Gate 2 — Create a clean vLLM runtime environment (separate from QueryLake API)

**Objective:** a minimal env that can run vLLM 0.13 standalone.

- [ ] Create a new env (recommended: uv + venv; conda acceptable if needed).
- [ ] Install:
  - [ ] `torch==2.9.0` (matching the chosen CUDA track)
  - [ ] `vllm==0.13.0`
- [ ] Sanity checks:
  - [ ] `python -c "import torch; print(torch.cuda.is_available())"`
  - [ ] `python -c "import vllm; print(vllm.__version__)"`

**Done when:** env is reproducible (lockfile or exact install script) and imports cleanly.

---

## Gate 3 — Prove vLLM standalone serving works (before touching QueryLake integration)

- [ ] Start `vllm serve ...` for one small model (prefer a small text model first).
- [ ] Validate endpoints:
  - [ ] `GET /v1/models`
  - [ ] `POST /v1/chat/completions` non-streaming
  - [ ] `POST /v1/chat/completions` streaming
- [ ] Capture baseline perf numbers:
  - [ ] time-to-first-token (TTFT) rough estimate
  - [ ] tokens/sec rough estimate
  - [ ] GPU mem footprint (`nvidia-smi`)

**Done when:** vLLM is a known-good upstream service.

---

## Gate 4 — Refactor QueryLake to stop importing vLLM internals (use HTTP adapter)

**Objective:** make vLLM upgrades survivable by turning it into a provider boundary.

- [x] Remove `vllm.entrypoints.openai.protocol` imports from the FastAPI routing layer:
  - [x] `/v1/chat/completions` parses JSON directly and passes dicts to the Serve deployment
  - [x] `/v1/embeddings` parses JSON directly and returns an OpenAI-shaped JSON response
- [x] Move non-vLLM helpers out of `ray_vllm_class` so core modules don’t import vLLM by accident:
  - [x] `format_chat_history` now lives in `QueryLake/misc_functions/chat_history.py`
- [ ] Add an internal abstraction for “LLM provider” with a concrete HTTP implementation:
  - [ ] chat completions
  - [ ] embeddings (if used via vLLM)
- [ ] Add config for local vLLM upstream base URL and model mapping.
- [ ] Keep the old in-process path only as a temporary fallback (optional).

**Done when:** QueryLake can serve `/v1/chat/completions` by proxying to a running vLLM server.

---

## Gate 5 — Build a clean QueryLake API environment (separate from vLLM runtime)

**Objective:** QueryLake API env should not include heavyweight inference kernels.

- [ ] Create a new env for QueryLake API.
- [ ] Install only what the API needs (FastAPI/Ray/DB/http clients).
- [ ] Ensure the API env does *not* pull in the full inference stack unless intentionally required.
- [ ] Align Ray version with the cluster version you actually run.

**Done when:** QueryLake server boots and can talk to vLLM upstream reliably.

---

## Gate 6 — Testing and rollout

- [ ] Smoke tests:
  - [ ] `/v1/chat/completions` works for the target model
  - [ ] embeddings endpoint works (if enabled)
  - [ ] rerank endpoint works (if enabled)
- [ ] Canary:
  - [ ] route a small % of requests (or a small set of API keys) to the new stack
  - [ ] define rollback triggers (error rate, latency regressions, OOMs)
- [ ] Promote and keep old stack available for rollback.

**Done when:** new stack is default and rollback is proven.

---

## Gate 7 — Post-upgrade cleanup (dependency hygiene)

- [ ] Replace `requirements.txt`-only installs with a locked dependency set (uv lockfile or equivalent).
- [ ] Split optional stacks into extras/groups (e.g., API vs OCR vs training experimentation).
- [ ] Prune unused packages from the production env(s) to keep `pip check` clean.

**Done when:** “stable” channel is reproducible and “edge” channel exists for experimentation.

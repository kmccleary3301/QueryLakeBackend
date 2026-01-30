#!/usr/bin/env python3
"""
QueryLake GPU Placement System & Application Starter

This system implements the "one-node-per-GPU" architecture for Ray,
enabling fine-grained, GPU-level placement control for QueryLake's AI models.

Key Architectural Principles:
1.  **GPU Isolation**: Each physical GPU on a machine is assigned its own dedicated
    Ray worker process. This is achieved by setting the `CUDA_VISIBLE_DEVICES`
    environment variable for each worker, making it believe it's on a
    single-GPU machine.

2.  **Honest Resource Representation**: Instead of VRAM-fraction hacks, each worker
    node registers itself with `num_gpus=1` and reports its true VRAM capacity
    (detected via `pynvml`) as a custom Ray resource (`"VRAM_MB"`).

3.  **Unlocking Native Ray Scheduling**: By treating each GPU as a separate "node",
    Ray's built-in, node-level placement strategies (`PACK`, `SPREAD`, etc.)
    can be used to control replica placement *at the GPU level*.

4.  **Configuration-Driven**: The entire system is driven by `config.json`.
    The `QueryLake.typing.config.Config` Pydantic model is used for validation
    and to ensure all settings (models, resources, cluster ports) are loaded
    from a single source of truth.

5.  **Robust Lifecycle Management**: The CLI handles port cleanup, sequential
    startup of the head and worker nodes, cluster health verification, and

    graceful shutdown of all managed processes.
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Apply Ray logging defaults before importing ray so every Ray process inherits them.
_DEFAULT_RAY_ENV: Dict[str, str] = {
    "RAY_ROTATION_MAX_BYTES": str(100 * 1024 * 1024),  # 100 MiB
    "RAY_ROTATION_BACKUP_COUNT": "10",
    "RAY_BACKEND_LOG_LEVEL": "warning",
    "RAY_DEDUP_LOGS": "1",
}

_DEFAULT_TMPDIR = (
    os.environ.get("RAY_TMPDIR")
    or os.environ.get("QUERYLAKE_RAY_TMPDIR")
    or "/tmp/querylake_ray"
)
_DEFAULT_RAY_ENV["RAY_TMPDIR"] = _DEFAULT_TMPDIR

_ENV_SETUP_WARNINGS: List[str] = []
for _key, _value in _DEFAULT_RAY_ENV.items():
    os.environ.setdefault(_key, _value)

try:
    tmp_dir_path = Path(os.environ["RAY_TMPDIR"])
    tmp_dir_path.mkdir(parents=True, exist_ok=True)
except OSError as _exc:  # pragma: no cover - best effort guardrail
    _ENV_SETUP_WARNINGS.append(
        f"Failed to ensure Ray temp directory {os.environ['RAY_TMPDIR']}: {_exc}"
    )
    os.environ.pop("RAY_TMPDIR", None)

import ray  # noqa: E402  pylint: disable=wrong-import-position
from ray import serve  # noqa: E402  pylint: disable=wrong-import-position
try:
    from ray.logging_config import LoggingConfig  # noqa: E402
except ImportError:  # pragma: no cover
    LoggingConfig = None  # type: ignore
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetMemoryInfo,
    NVMLError
)


# Import QueryLake's own Config validator and application builder
from pydantic import ValidationError

from QueryLake.typing.config import Config, ToolChain
from QueryLake.typing.toolchains import ToolChainV2
from QueryLake.toolchains.legacy_converter import convert_toolchain_dict
# This function is the new entrypoint for the application, created in server.py
from server import build_and_run_application

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('querylake_cluster.log')
    ]
)
logger = logging.getLogger(__name__)

if _ENV_SETUP_WARNINGS:
    for _warning in _ENV_SETUP_WARNINGS:
        logger.warning(_warning)

logger.info(
    "Ray logging defaults in effect | rotation_max_bytes=%s | rotation_backups=%s | backend_level=%s | dedup=%s | temp_dir=%s",
    os.environ.get("RAY_ROTATION_MAX_BYTES"),
    os.environ.get("RAY_ROTATION_BACKUP_COUNT"),
    os.environ.get("RAY_BACKEND_LOG_LEVEL"),
    os.environ.get("RAY_DEDUP_LOGS"),
    os.environ.get("RAY_TMPDIR") or "<ray default>",
)

def load_config_and_toolchains(
    config_path: str = "config.json",
) -> Tuple[Optional[Config], Dict[str, ToolChain], Dict[str, ToolChainV2]]:
    """Load and parse configuration using QueryLake's Pydantic models."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_json_str = f.read()
        
        config = Config.model_validate_json(config_json_str)
        logger.info(f"✅ Loaded and validated configuration from {config_path}")

        toolchains_v1: Dict[str, ToolChain] = {}
        toolchains_v2: Dict[str, ToolChainV2] = {}
        toolchain_files_list = os.listdir("toolchains")
        for toolchain_file in toolchain_files_list:
            if not toolchain_file.endswith(".json"):
                continue
            with open(os.path.join("toolchains", toolchain_file), 'r', encoding='utf-8') as f:
                toolchain_retrieved = json.load(f)
            # Always attempt to load a legacy v1 object so DB storage remains compatible
            try:
                toolchains_v1[toolchain_retrieved["id"]] = ToolChain(**toolchain_retrieved)
            except ValidationError:
                # If strict validation fails, fall back to permissive legacy model (may allow extras)
                toolchains_v1[toolchain_retrieved["id"]] = ToolChain(**toolchain_retrieved)

            # Prefer native v2 definitions when possible; otherwise convert from legacy v1
            try:
                v2_obj = ToolChainV2.model_validate(toolchain_retrieved)
                toolchains_v2[v2_obj.id] = v2_obj
            except ValidationError:
                toolchains_v2[toolchain_retrieved["id"]] = convert_toolchain_dict(toolchain_retrieved)
        logger.info(f"✅ Loaded {len(toolchains_v1)} toolchains")

        return config, toolchains_v1, toolchains_v2
        
    except FileNotFoundError:
        logger.error(f"❌ Configuration file {config_path} not found")
        return None, {}, {}
    except Exception as e:
        logger.error(f"❌ Invalid configuration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, {}

class RayGPUCluster:
    """
    Manages a Ray cluster using the "one-node-per-GPU" architecture.
    """
    
    def __init__(self, config: Config, toolchains: Dict[str, ToolChain], toolchains_v2: Dict[str, ToolChainV2]):
        self.config = config
        self.toolchains = toolchains
        # Preserve the loaded v2 toolchains; do not overwrite with an empty dict
        self.toolchains_v2 = toolchains_v2
        self.head_process = None
        self.worker_processes = []
        self.available_gpus = []
        self.cluster_ready = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Detect available NVIDIA GPUs and their VRAM using pynvml."""
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                gpus.append({
                    "index": i,
                    "total_vram_mb": memory_info.total // (1024 ** 2)
                })
            if not gpus:
                logger.warning("⚠️ No NVIDIA GPUs detected by pynvml.")
            return gpus
        except NVMLError as error:
            logger.error(f"❌ pynvml error: {error}. Could not detect GPUs.")
            return []

    def _get_vram_per_gpu(self) -> Optional[int]:
        """Get the VRAM amount per GPU for placement group allocation."""
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not available - cannot detect VRAM per GPU")
            return None
            
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            if device_count == 0:
                return None
                
            # Get VRAM from first GPU (assuming all GPUs have same VRAM)
            handle = nvmlDeviceGetHandleByIndex(0)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            vram_mb = memory_info.total // (1024 * 1024)
            
            # Verify all GPUs have the same VRAM
            for i in range(1, device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                current_vram_mb = memory_info.total // (1024 * 1024)
                if current_vram_mb != vram_mb:
                    logger.warning(f"GPU {i} has different VRAM ({current_vram_mb}MB) than GPU 0 ({vram_mb}MB)")
                    # Use the minimum VRAM across all GPUs for safety
                    vram_mb = min(vram_mb, current_vram_mb)
            
            return vram_mb
        except Exception as e:
            logger.error(f"Failed to detect VRAM per GPU: {e}")
            return None
    
    def cleanup_ports(self, ports: List[int]):
        """Clean up any processes using target ports before startup."""
        logger.info("🧹 Cleaning up existing processes on target ports...")
        for port in ports:
            try:
                # Use `lsof` to find PIDs listening on the port
                check_cmd = f"lsof -ti :{port}"
                result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    logger.info(f"  🔧 Found {len(pids)} process(es) on port {port}. Terminating...")
                    
                    # Kill the found processes
                    kill_cmd = f"kill -9 {' '.join(pids)}"
                    kill_result = subprocess.run(kill_cmd, shell=True, capture_output=True)
                    if kill_result.returncode == 0:
                        logger.info(f"  ✅ Cleaned up port {port}")
                    else:
                        logger.warning(f"  ⚠️ Failed to clean port {port}. Stderr: {kill_result.stderr}")
                else:
                    logger.info(f"  ✅ Port {port} is already clean")
            except Exception as e:
                logger.warning(f"  ⚠️ Error during cleanup for port {port}: {e}")
        time.sleep(2) # Give OS time to release ports
    
    def force_stop_existing_cluster(self) -> None:
        """Best-effort shutdown of any previously running local Ray cluster."""
        try:
            result = subprocess.run(
                ["ray", "stop", "--force"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                logger.info("🧹 Cleared lingering Ray cluster state with `ray stop --force`.")
            else:
                stderr = result.stderr.strip()
                if stderr:
                    logger.debug("`ray stop --force` stderr: %s", stderr)
        except FileNotFoundError:
            logger.debug("`ray` executable not found when attempting to stop existing cluster.")
        except subprocess.TimeoutExpired:
            logger.warning("`ray stop --force` timed out; continuing with manual cleanup.")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Unexpected issue while running `ray stop --force`: %s", exc)
    
    def start_head_node(self) -> bool:
        """Start the Ray head node (for coordination, with no GPUs allocated to it)."""
        cluster_config = self.config.ray_cluster
        if not self.available_gpus:
            self.available_gpus = self.get_gpu_info()
        gpu_count = max(len(self.available_gpus) if self.available_gpus else 0, 1)
        port_base = getattr(cluster_config, "worker_port_base", None)
        port_step = getattr(cluster_config, "worker_port_step", None)
        head_min_port = None
        head_max_port = None
        if port_base is not None and port_step is not None:
            # Reserve one additional slot worth of ports for local drivers and job submissions.
            head_min_port = port_base
            head_max_port = port_base + port_step * (gpu_count + 1) - 1
        try:
            logger.info(f"🚀 Starting Ray head node on port {cluster_config.head_port}...")
            
            head_cmd = [
                "ray", "start", "--head", "--num-gpus=0",
                "--node-ip-address=127.0.0.1",
                f"--dashboard-port={cluster_config.dashboard_port}",
                f"--port={cluster_config.head_port}",
                "--block",
                "--disable-usage-stats"
            ]
            if head_min_port is not None and head_max_port is not None:
                head_cmd.extend([
                    f"--min-worker-port={head_min_port}",
                    f"--max-worker-port={head_max_port}",
                ])
            ray_tmpdir = os.environ.get("RAY_TMPDIR")
            if ray_tmpdir:
                head_cmd.insert(-1, f"--temp-dir={ray_tmpdir}")
            
            with open("head_node.log", "w") as head_log:
                self.head_process = subprocess.Popen(
                    head_cmd,
                    stdout=head_log,
                    stderr=head_log,
                    env=os.environ.copy(),
                )
            
            logger.info("⏳ Waiting for head node to initialize (8s)...")
            time.sleep(8)
            
            # Verify the head node is accessible on its port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                if s.connect_ex(('127.0.0.1', cluster_config.head_port)) == 0:
                    logger.info("✅ Head node started successfully and is accessible.")
                    return True
            
            logger.error("❌ Head node port is not accessible after startup.")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start head node: {e}")
            return False
    
    def start_workers(self, head_address: str) -> bool:
        """Start one worker node per detected GPU."""
        cluster_config = self.config.ray_cluster
        try:
            if not self.available_gpus:
                self.available_gpus = self.get_gpu_info()
            if not self.available_gpus:
                logger.error("❌ No GPUs found. Cannot start GPU workers.")
                return False
            
            logger.info(f"🖥️ Found {len(self.available_gpus)} GPUs. Starting one worker per GPU.")
            for gpu in self.available_gpus:
                logger.info(f"   - GPU {gpu['index']}: {gpu['total_vram_mb']}MB VRAM")
            
            logger.info(f"🔧 Starting {len(self.available_gpus)} worker nodes, connecting to {head_address}...")
            
            for i, gpu in enumerate(self.available_gpus):
                logger.info(f"  📦 Starting worker {i+1}/{len(self.available_gpus)} for GPU {gpu['index']}...")
                
                # Isolate the worker to a single GPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
                
                # Register this worker's true resources with Ray
                vram_mb = gpu["total_vram_mb"]
                resources = json.dumps({"VRAM_MB": vram_mb})
                
                # Assign a unique port range to prevent gRPC conflicts
                min_port = cluster_config.worker_port_base + (i * cluster_config.worker_port_step)
                max_port = min_port + cluster_config.worker_port_step - 1
                
                worker_cmd = [
                    "ray", "start", f"--address={head_address}",
                    "--node-ip-address=127.0.0.1",
                    "--num-gpus=1",  # This worker provides exactly 1 GPU to the cluster
                    f"--resources={resources}",
                    f"--min-worker-port={min_port}",
                    f"--max-worker-port={max_port}",
                    "--block",
                    "--disable-usage-stats"
                ]
                ray_tmpdir = os.environ.get("RAY_TMPDIR")
                if ray_tmpdir:
                    worker_cmd.insert(-1, f"--temp-dir={ray_tmpdir}")
                
                log_file = f"worker_node_{gpu['index']}.log"
                with open(log_file, "w") as worker_log:
                    proc = subprocess.Popen(worker_cmd, env=env, stdout=worker_log, stderr=worker_log)
                self.worker_processes.append(proc)
                logger.info(f"  ✅ Launched worker for GPU {gpu['index']} (PID: {proc.pid}, Ports: {min_port}-{max_port})")
                
                # Stagger worker startups slightly to avoid GCS registration storms
                if i < len(self.available_gpus) - 1:
                    logger.info(f"     -> Waiting 3s before starting next worker...")
                    time.sleep(3)
            
            logger.info("⏳ Waiting 10s for all workers to register with the head node...")
            time.sleep(10)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start workers: {e}")
            return False
    
    def connect_to_cluster(self, address: str) -> bool:
        """Connect the current process to the Ray cluster."""
        try:
            logger.info(f"🔗 Connecting client to Ray cluster at {address}...")
            init_kwargs: Dict[str, Any] = {
                "address": address,
                "namespace": "querylake",
                "log_to_driver": False,
                "logging_level": logging.WARNING,
            }
            ray_tmpdir = os.environ.get("RAY_TMPDIR")
            if ray_tmpdir:
                init_kwargs["_temp_dir"] = ray_tmpdir
            if LoggingConfig is not None:
                init_kwargs["logging_config"] = LoggingConfig(
                    log_level="WARNING",
                    encoding="TEXT",
                )
            # We connect to an existing cluster, so we don't initialize a new one
            ray.init(**init_kwargs)
            logger.info("✅ Client connected successfully to the 'querylake' namespace.")
            self.cluster_ready = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect client to cluster: {e}")
            return False
    
    def verify_cluster_status(self) -> bool:
        """Verify that the cluster is healthy and all expected GPU nodes are present."""
        if not self.cluster_ready:
            logger.error("Cannot verify status, client not connected.")
            return False
        try:
            logger.info("📊 Verifying cluster status...")
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n['alive']]
            gpu_nodes = [n for n in alive_nodes if n.get('Resources', {}).get('GPU', 0) > 0]
            
            total_gpus = sum(n.get('Resources', {}).get('GPU', 0) for n in alive_nodes)
            total_vram = sum(n.get('Resources', {}).get('VRAM_MB', 0) for n in alive_nodes)
            
            logger.info(f"  - Alive nodes: {len(alive_nodes)}")
            logger.info(f"  - GPU worker nodes: {len(gpu_nodes)}")
            logger.info(f"  - Total GPUs in cluster: {total_gpus}")
            logger.info(f"  - Total VRAM in cluster: {total_vram}MB")
            
            expected_gpu_nodes = len(self.available_gpus)
            api_only = os.environ.get("QUERYLAKE_API_ONLY", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if api_only and expected_gpu_nodes and len(gpu_nodes) == 0:
                logger.info(
                    "QUERYLAKE_API_ONLY=1: proceeding without GPU worker nodes (expected %s).",
                    expected_gpu_nodes,
                )
                return True
            if len(gpu_nodes) == expected_gpu_nodes and total_gpus == expected_gpu_nodes:
                logger.info("  🎉 SUCCESS: All GPU workers registered correctly!")
                return True
            else:
                logger.error(f"  ❌ MISMATCH: Expected {expected_gpu_nodes} GPU nodes, but found {len(gpu_nodes)}.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cluster verification failed: {e}")
            return False
    
    def deploy_querylake_application(self, strategy: str) -> bool:
        """Deploy the main QueryLake application using Ray Serve."""
        if not self.cluster_ready:
            logger.error("Cannot deploy application, client not connected to Ray.")
            return False

        try:
            logger.info(f"🚀 Deploying QueryLake application with '{strategy}' strategy...")
            # Ray Serve will handle placement groups internally now
            build_and_run_application(
                strategy=strategy,
                global_config=self.config,
                toolchains_v1=self.toolchains,
                toolchains_v2=self.toolchains_v2,
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to deploy QueryLake application: {e}")
            logger.error(traceback.format_exc())
            return False

    def shutdown(self):
        """Gracefully shut down all managed Ray processes."""
        logger.info("🛑 Shutting down the cluster...")
        
        try:
            # Disconnect client and shut down Serve
            if self.cluster_ready and ray.is_initialized():
                serve.shutdown()
                try:
                    ray.disconnect()
                    logger.info("  - Ray client disconnected.")
                except AttributeError:
                    logger.warning("  - `ray.disconnect()` not available in this version of Ray. Skipping.")

            # Terminate worker processes
            for i, proc in enumerate(self.worker_processes):
                if proc.poll() is None:
                    logger.info(f"  - Terminating worker {i} (PID: {proc.pid})")
                    proc.terminate()
            
            # Terminate head process
            if self.head_process and self.head_process.poll() is None:
                logger.info(f"  - Terminating head node (PID: {self.head_process.pid})")
                self.head_process.terminate()

            # Wait for processes to exit
            for proc in self.worker_processes:
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: proc.kill()
            if self.head_process:
                try: self.head_process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.head_process.kill()

            logger.info("  - All managed processes terminated.")
            
        except Exception as e:
            logger.error(f"⚠️ Error during shutdown: {e}")


def main():
    """Main CLI entry point for starting and managing QueryLake."""
    parser = argparse.ArgumentParser(
        description="QueryLake GPU Placement System & Application Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a full local cluster (head + workers) with the default PACK strategy
  python3 start_querylake.py

  # Start a full local cluster with the SPREAD strategy
  python3 start_querylake.py --strategy SPREAD

  # Start workers only and connect to an existing head node on another machine
  python3 start_querylake.py --workers --head-node 192.168.1.100:6379

  # Start workers for only the first 2 GPUs and connect to a head node
  python3 start_querylake.py --workers 2 --head-node 192.168.1.100:6379
        """
    )
    
    parser.add_argument("--strategy", choices=["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"],
                        help="Override the default GPU placement strategy from config.json.")
    parser.add_argument("--config", default="config.json",
                        help="Path to the QueryLake configuration file.")
    parser.add_argument("--workers", nargs='?', const=-1, type=int,
                        help="Run in worker-only mode. Optionally specify the number of GPUs to use.")
    parser.add_argument("--head-node", type=str,
                        help="The address of the Ray head node to connect to (e.g., 127.0.0.1:6379). Required for --workers mode.")
    parser.add_argument(
        "--allow-ray-stop",
        action="store_true",
        help=(
            "Allow `ray stop --force` to run before startup. "
            "WARNING: this can stop unrelated Ray workloads on the machine."
        ),
    )
    parser.add_argument(
        "--with-gpu-workers",
        action="store_true",
        help="Start GPU worker processes even when QUERYLAKE_API_ONLY=1.",
    )
    parser.add_argument("--no-monitor", action="store_true",
                        help="Exit immediately after startup, for scripting.")

    args = parser.parse_args()

    # Validate worker mode arguments
    if args.workers is not None and not args.head_node:
        parser.error("--head-node is required when using --workers")

    config, toolchains, toolchains_v2 = load_config_and_toolchains(args.config)
    if not config:
        sys.exit(1)

    # Override strategy if provided via CLI
    strategy = args.strategy or config.ray_cluster.default_gpu_strategy
    logger.info(f"📋 Using GPU placement strategy: {strategy}")

    cluster = RayGPUCluster(config, toolchains, toolchains_v2)

    try:
        if args.workers is None:
            # --- Head + Worker Mode (Default) ---
            logger.info("🚀 Starting in default mode: Head + Workers on this machine.")
            cluster_config = config.ray_cluster
            ports_to_clean = [cluster_config.head_port, cluster_config.dashboard_port]
            # Add worker ports to cleanup list later if needed
            
            if args.allow_ray_stop:
                cluster.force_stop_existing_cluster()
            else:
                logger.info(
                    "Skipping `ray stop --force` (use --allow-ray-stop to enable)."
                )
            cluster.cleanup_ports(ports_to_clean)
            
            if not cluster.start_head_node():
                raise RuntimeError("Failed to start the head node.")
            
            head_address = f"127.0.0.1:{cluster_config.head_port}"
            api_only = os.environ.get("QUERYLAKE_API_ONLY", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if api_only and not args.with_gpu_workers:
                logger.info(
                    "QUERYLAKE_API_ONLY=1: skipping GPU worker nodes (pass --with-gpu-workers to override)."
                )
            else:
                if not cluster.start_workers(head_address):
                    raise RuntimeError("Failed to start worker nodes.")

            if not cluster.connect_to_cluster(head_address):
                raise RuntimeError("Failed to connect client to the cluster.")
            
        else:
            # --- Worker-Only Mode ---
            logger.info("🚀 Starting in Worker-Only mode.")
            head_address = args.head_node
            # Logic to limit GPUs will be added to start_workers
            if not cluster.start_workers(head_address):
                raise RuntimeError("Failed to start and connect worker nodes.")
            
            if not cluster.connect_to_cluster(head_address):
                raise RuntimeError("Failed to connect client to the cluster.")

        # --- Common Cluster Verification and Deployment ---
        if not cluster.verify_cluster_status():
            raise RuntimeError("Cluster verification failed.")

        if not cluster.deploy_querylake_application(strategy):
            raise RuntimeError("Failed to deploy the QueryLake application.")
        
        logger.info("🎉 System startup complete!")
        if not args.workers:
            logger.info(f"   - Ray Dashboard: http://127.0.0.1:{config.ray_cluster.dashboard_port}")
        logger.info(f"   - Connected to head at: {ray.get_runtime_context().gcs_address}")
        
        if not args.no_monitor:
            logger.info("✅ QueryLake is running. Press Ctrl+C to shutdown.")
            while True:
                time.sleep(60)

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("\nShutdown requested by user.")
        else:
            logger.error(f"❌ An error occurred during startup: {e}")
            import traceback
            logger.error(traceback.format_exc())
    finally:
        logger.info("Initiating shutdown sequence...")
        cluster.shutdown()
        logger.info("Shutdown complete. Exiting.")


if __name__ == "__main__":
    # Add a guard for socket which is used in the script
    import socket
    sys.exit(main()) 

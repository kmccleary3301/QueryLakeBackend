from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from server import _normalize_chandra_deployment_resources


def test_chandra_vllm_server_is_cpu_only_and_does_not_reserve_vram():
    deployment_config = {
        "vram_required": 18000,
        "ray_actor_options": {
            "num_gpus": 1.0,
            "num_cpus": 4,
            "resources": {"VRAM_MB": 18000},
        },
    }
    normalized, replica_resources, required_vram = _normalize_chandra_deployment_resources(
        runtime_backend="vllm_server",
        deployment_config=deployment_config,
    )
    assert required_vram == 0
    assert normalized["ray_actor_options"]["num_gpus"] == 0
    assert "VRAM_MB" not in normalized["ray_actor_options"]["resources"]
    assert "GPU" not in replica_resources
    assert "VRAM_MB" not in replica_resources
    assert replica_resources["CPU"] == 4


def test_chandra_hf_reserves_gpu_and_vram_mb():
    deployment_config = {
        "vram_required": 18000,
        "ray_actor_options": {
            "num_gpus": 0.01,
            "num_cpus": 3,
            "resources": {},
        },
    }
    normalized, replica_resources, required_vram = _normalize_chandra_deployment_resources(
        runtime_backend="hf",
        deployment_config=deployment_config,
    )
    assert required_vram == 18000
    assert normalized["ray_actor_options"]["num_gpus"] >= 0.1
    assert normalized["ray_actor_options"]["resources"]["VRAM_MB"] == 18000
    assert replica_resources["GPU"] == normalized["ray_actor_options"]["num_gpus"]
    assert replica_resources["VRAM_MB"] == 18000
    assert replica_resources["CPU"] == 3


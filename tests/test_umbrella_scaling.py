from QueryLake.runtime.umbrella_scaling import get_umbrella_deployment_options


def test_default_umbrella_options() -> None:
    options = get_umbrella_deployment_options(env={})
    assert options["max_ongoing_requests"] == 100
    assert "autoscaling_config" not in options


def test_autoscaling_env_options() -> None:
    env = {
        "QL_UMBRELLA_MIN_REPLICAS": "2",
        "QL_UMBRELLA_MAX_REPLICAS": "4",
        "QL_UMBRELLA_TARGET_ONGOING_REQUESTS": "16",
        "QL_UMBRELLA_UPSCALE_DELAY_S": "3",
        "QL_UMBRELLA_DOWNSCALE_DELAY_S": "7",
        "QL_UMBRELLA_MAX_ONGOING_REQUESTS": "200",
        "QL_UMBRELLA_RESOURCES_JSON": "{\"CPU_NODE\": 1}",
    }
    options = get_umbrella_deployment_options(env=env)
    assert options["max_ongoing_requests"] == 200
    assert options["autoscaling_config"]["min_replicas"] == 2
    assert options["autoscaling_config"]["max_replicas"] == 4
    assert options["autoscaling_config"]["target_ongoing_requests"] == 16
    assert options["autoscaling_config"]["upscale_delay_s"] == 3.0
    assert options["autoscaling_config"]["downscale_delay_s"] == 7.0
    assert options["ray_actor_options"]["resources"]["CPU_NODE"] == 1

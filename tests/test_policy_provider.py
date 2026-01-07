from QueryLake.runtime.policy import EffectivePolicy, EnvPolicyProvider, set_policy_provider, resolve_policy


class _TestProvider:
    def __init__(self, policy: EffectivePolicy) -> None:
        self.policy = policy

    def resolve(self, route: str, api_key_id: str | None = None) -> EffectivePolicy:
        return self.policy


def test_policy_provider_override() -> None:
    override = EffectivePolicy(enabled=False, requests_per_minute=1)
    set_policy_provider(_TestProvider(override))
    resolved = resolve_policy("/api/test")
    assert resolved == override
    # restore default for other tests
    set_policy_provider(EnvPolicyProvider())

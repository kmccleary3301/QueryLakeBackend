from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import os


@dataclass(frozen=True)
class EffectivePolicy:
    enabled: bool = True
    requests_per_minute: Optional[int] = None
    requests_per_second: Optional[int] = None
    concurrency_limit: Optional[int] = None
    lease_ttl_seconds: int = 60


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


class PolicyProvider(Protocol):
    def resolve(self, route: str, api_key_id: Optional[str] = None) -> EffectivePolicy:
        raise NotImplementedError


class EnvPolicyProvider:
    def resolve(self, route: str, api_key_id: Optional[str] = None) -> EffectivePolicy:
        # Minimal, conservative defaults. Prefer per-route env overrides.
        prefix = route.upper().replace("/", "_")
        return EffectivePolicy(
            enabled=os.getenv(f"QL_POLICY_{prefix}_ENABLED", "1") != "0",
            requests_per_minute=_env_int(f"QL_POLICY_{prefix}_RPM"),
            requests_per_second=_env_int(f"QL_POLICY_{prefix}_RPS"),
            concurrency_limit=_env_int(f"QL_POLICY_{prefix}_CONCURRENCY"),
            lease_ttl_seconds=_env_int("QL_POLICY_LEASE_TTL", 60) or 60,
        )


_policy_provider: PolicyProvider = EnvPolicyProvider()


def set_policy_provider(provider: PolicyProvider) -> None:
    global _policy_provider
    _policy_provider = provider


def resolve_policy(route: str, api_key_id: Optional[str] = None) -> EffectivePolicy:
    return _policy_provider.resolve(route, api_key_id=api_key_id)

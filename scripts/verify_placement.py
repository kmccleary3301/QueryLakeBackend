#!/usr/bin/env python3
"""Inspect GPU placement for QueryLake Serve replicas."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List

import ray
from ray.util import state


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _describe_serve_replicas(
    *,
    serve_namespace: str,
    include_node_details: bool,
) -> List[Dict[str, Any]]:
    nodes_by_id = {}
    if include_node_details:
        for node in ray.nodes():
            if node.get("alive"):
                nodes_by_id[node["NodeID"]] = {
                    "ip": node.get("NodeManagerAddress"),
                    "hostname": node.get("NodeManagerHostname"),
                    "resources": node.get("Resources", {}),
                }

    placement_groups = state.list_placement_groups(limit=200, detail=True)
    results: List[Dict[str, Any]] = []

    for pg_entry in placement_groups:
        pg = asdict(pg_entry)
        name = pg.get("name", "")
        if not name.startswith("SERVE_REPLICA::"):
            continue

        bundles = pg.get("bundles", [])
        bundle = bundles[0] if bundles else {}
        node_id = bundle.get("node_id")
        deployment_info = {
            "placement_group": pg.get("placement_group_id"),
            "replica_name": name,
            "application": name.split("::", maxsplit=1)[-1].split("#", maxsplit=1)[0],
            "deployment": "",
            "replica_tag": "",
            "bundle_resources": bundle.get("unit_resources", {}),
            "node_id": node_id,
            "node": nodes_by_id.get(node_id, {}) if include_node_details else {},
            "runtime_metadata": None,
            "warnings": [],
        }

        try:
            _, deployment, replica_tag = name.split("#", maxsplit=2)
            deployment_info["deployment"] = deployment
            deployment_info["replica_tag"] = replica_tag
        except ValueError:
            deployment_info["warnings"].append("Unable to parse deployment name.")

        try:
            actor = ray.get_actor(name, namespace=serve_namespace)
            describe_runtime = getattr(actor, "describe_runtime", None)
            if describe_runtime is None:
                deployment_info["warnings"].append("describe_runtime not exposed on replica.")
            else:
                deployment_info["runtime_metadata"] = ray.get(describe_runtime.remote())
        except ValueError:
            deployment_info["warnings"].append("Replica actor not found.")
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            deployment_info["warnings"].append(f"Runtime probe failed: {exc}")

        results.append(deployment_info)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GPU placement for QueryLake Serve replicas.")
    parser.add_argument(
        "--address",
        default="auto",
        help="Ray cluster address (default: auto).",
    )
    parser.add_argument(
        "--namespace",
        default="querylake",
        help="Driver namespace to use when connecting (default: querylake).",
    )
    parser.add_argument(
        "--serve-namespace",
        default="serve",
        help="Ray namespace where Serve actors live (default: serve).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of logs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for the harness.",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)
    logging.info("Connecting to Ray cluster at %s (namespace=%s)", args.address, args.namespace)

    ray.init(
        address=args.address,
        namespace=args.namespace,
        ignore_reinit_error=True,
        log_to_driver=False,
    )

    try:
        replica_snapshots = _describe_serve_replicas(
            serve_namespace=args.serve_namespace,
            include_node_details=True,
        )
    finally:
        ray.shutdown()

    if args.json:
        print(json.dumps(replica_snapshots, indent=2, sort_keys=True))
        return 0

    if not replica_snapshots:
        logging.warning("No Serve replicas found. Is the application running?")
        return 0

    for snapshot in replica_snapshots:
        runtime = snapshot.get("runtime_metadata") or {}
        gpu_ids = runtime.get("gpu_ids")
        cuda_visible = runtime.get("cuda_visible_devices")
        node = snapshot.get("node", {})
        logging.info(
            "%s | deployment=%s | node=%s | gpu_ids=%s | CUDA_VISIBLE_DEVICES=%s",
            snapshot["replica_name"],
            snapshot.get("deployment"),
            f"{node.get('hostname')} ({snapshot.get('node_id')})",
            gpu_ids,
            cuda_visible,
        )
        if snapshot.get("warnings"):
            for warning in snapshot["warnings"]:
                logging.warning("  ↳ %s", warning)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

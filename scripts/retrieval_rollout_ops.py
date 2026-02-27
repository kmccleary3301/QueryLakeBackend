#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.create_db_session import initialize_database_engine
from QueryLake.runtime.retrieval_rollout import (
    resolve_active_pipeline,
    rollback_active_pipeline,
    set_active_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Operate retrieval pipeline bindings for canary/rollback.")
    parser.add_argument("action", choices=["set", "rollback", "show"])
    parser.add_argument("--route", required=True)
    parser.add_argument("--tenant-scope", default=None)
    parser.add_argument("--pipeline-id", default=None)
    parser.add_argument("--pipeline-version", default=None)
    parser.add_argument("--updated-by", default="automation")
    parser.add_argument("--reason", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    db, _ = initialize_database_engine()
    if args.action == "set":
        if not args.pipeline_id or not args.pipeline_version:
            raise ValueError("--pipeline-id and --pipeline-version are required for action=set")
        row = set_active_pipeline(
            db,
            route=args.route,
            tenant_scope=args.tenant_scope,
            pipeline_id=args.pipeline_id,
            pipeline_version=args.pipeline_version,
            updated_by=args.updated_by,
            reason=args.reason,
        )
    elif args.action == "rollback":
        row = rollback_active_pipeline(
            db,
            route=args.route,
            tenant_scope=args.tenant_scope,
            updated_by=args.updated_by,
            reason=args.reason or "rollback",
        )
    else:
        row = resolve_active_pipeline(db, route=args.route, tenant_scope=args.tenant_scope)
        if row is None:
            print(json.dumps({"route": args.route, "tenant_scope": args.tenant_scope, "binding": None}, indent=2))
            return 0

    payload = row.model_dump()
    output_text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


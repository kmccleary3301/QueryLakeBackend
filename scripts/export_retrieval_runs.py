#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sqlmodel import select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.create_db_session import initialize_database_engine
from QueryLake.database.sql_db_tables import retrieval_run as RetrievalRun


def main() -> int:
    parser = argparse.ArgumentParser(description="Export retrieval_run rows to JSON for gate checks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--route", type=str, default=None)
    parser.add_argument("--status", type=str, default=None)
    args = parser.parse_args()

    db, _ = initialize_database_engine()
    stmt = select(RetrievalRun).order_by(RetrievalRun.created_at.desc()).limit(max(1, int(args.limit)))
    if args.route:
        stmt = stmt.where(RetrievalRun.route == args.route)
    if args.status:
        stmt = stmt.where(RetrievalRun.status == args.status)

    rows = list(db.exec(stmt).all())
    payload = [row.model_dump() for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(payload), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


## Summary

Describe the change in clear, concrete terms.

## Scope

- [ ] SDK (`sdk/python`)
- [ ] Retrieval/runtime (`QueryLake/runtime`, `scripts/retrieval_*`, `scripts/bcas_*`)
- [ ] API/backend (`QueryLake/api`, `server.py`, DB models/migrations)
- [ ] Docs/setup
- [ ] CI/workflows

## Validation

List commands actually run and key results.

```bash
# Example
make sdk-precommit-run
make sdk-ci
make ci-unification
make ci-retrieval-smoke
```

## Metrics impact (required for retrieval changes)

- Accuracy/quality deltas (recall, MRR, overlap):
- Latency/throughput deltas:
- Gate outcomes:

## Risk and rollback

- Risk level: low / medium / high
- Rollback plan:

## Checklist

- [ ] I ran the relevant local gates for this change.
- [ ] I updated docs for behavior/setup/workflow changes.
- [ ] I did not commit local secrets or machine-specific files.
- [ ] CI workflow effects were validated locally where possible.


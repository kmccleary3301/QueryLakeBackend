## Title
Strategic Review: Restoring GPU-Aware Placement for Ray Serve Deployments

## Summary
We previously concluded that Ray Serve (2.47.1) could not cleanly enforce per-GPU placement for deployment replicas when multiple GPUs live on a single physical node. Since then, Ray has continued to evolve, and it may now offer primitives that unlock the placement guarantees we need. We would like a deep technical reassessment of the problem—including Ray Core, Ray Serve, Ray Cluster launcher/autoscaler, and the associated resource semantics—to determine whether we can regain reliable, policy-driven GPU distribution (PACK/SPREAD) without fragile workarounds.

The planner should assume no additional context beyond the attachments listed below.

## Attachments
@CLI_SPRINT.md  
@ray_gpu_strategy_working_Node_Strategy.md  
@ray_gpu_strategy_working_RAY_GPU_PLACEMENT_COMPREHENSIVE_GUIDE.md  
@ray_gpu_strategy_working_IMPLEMENTATION_SUCCESS_SUMMARY.md  
@ray_gpu_strategy_working_final_01_cli.py  
@ray_gpu_strategy_working_final_01_deployments.py  
@start_querylake.py  
@server.py  
@config.json  
@docs_HANDOFF_V4.md

## Background (for the planner)
We operate QueryLake, a Ray Serve + FastAPI system that deploys several GPU-heavy models (LLMs via vLLM, embeddings, rerankers, etc.). We rely on fractional GPU reservations (`num_gpus`) plus a custom `VRAM_MB` resource to keep aggregate VRAM usage within bounds.

Our earlier attempts to control replica placement explored:

1. **Placement groups for Serve** — by preallocating bundles sized to each replica. Result: Ray Serve ignored the placement group or ended up deadlocked because the actors could not schedule into the externally reserved bundles.
2. **Custom resources and fractional GPU trickery** — Ray’s default scheduler still treated the node as a single pool and packed everything onto the first GPU until fully utilized.
3. **“One node per GPU” workaround** — Run separate Ray workers, each pinned to a single GPU via `CUDA_VISIBLE_DEVICES`, then rely on node-level PACK/SPREAD. This works but complicates cluster management and incurs extra Ray worker overhead.

We would prefer to let Ray Serve manage replicas on a single multi-GPU node, while still honoring a placement policy (PACK vs SPREAD) and respecting VRAM requirements, ideally without bespoke node orchestration.

## Key Questions for the Planner
Please deliver a thorough, well-researched response to each item. We are willing to digest a 20+ page report if necessary. Assume you have ample time to reason, experiment (mentally), and review the referenced attachments. Focus on Ray versions ≥ 2.7 and any roadmap items that may affect us within the next 6–12 months.

### A. Current Capabilities & APIs
1. **Ray Serve Placement Groups**  
   - Have there been changes in Ray Serve (post-2.47.1) that allow direct integration with placement groups or scheduling strategies for replicas?  
   - Are there internal APIs (documented or emerging) that expose per-replica placement controls?

2. **Serve Scheduling Strategies**  
   - Can we now set `scheduling_strategy` (e.g., `PlacementGroupSchedulingStrategy`, `NodeAffinitySchedulingStrategy`) on Serve deployments or underlying actors in a stable way?  
   - Is there an officially supported hook to influence placement that we previously missed?

3. **Ray Core Enhancements**  
   - Investigate recent Ray Core features (task/actor scheduling, custom resources, GPU scheduling hints) that could help enforce per-GPU isolation without splitting nodes.  
   - Evaluate whether raylets now provide GPU-memory-aware scheduling or if any proposals are close to landing.

4. **Autoscaler & Multi-node Strategy**  
   - If the recommended solution remains “one worker per GPU,” document best practices for scaling, failover, and maintenance (especially for heterogeneous clusters).  
   - Suggest how to align this with Ray’s autoscaler (both on-prem and cloud) so that node-level PACK/SPREAD can be toggled dynamically.

### B. Resource Modeling & Monitoring
5. **VRAM Accounting**  
   - Determine whether Ray now tracks GPU memory usage directly (or via plugins) in a way that could replace the `VRAM_MB` custom resource.  
   - Outline strategies to prevent vLLM replicas from launching when KV cache requirements exceed per-GPU limits, ideally using built-in Ray alarms or backpressure rather than manual heuristics.

6. **Fractional GPU Scheduling**  
   - Examine whether Ray’s fractional GPU scheduler has gained new knobs (e.g., fairness, spillover patterns) that could help approximate SPREAD within a node.  
   - Assess trade-offs between fractional GPU reservations vs. reserving full GPUs per replica when balancing throughput vs. placement control.

7. **Observability**  
   - Recommend tooling—Ray dashboard, metrics, logs, event hooks—to verify GPU placement and detect drift in real time.  
   - Suggest any patching or instrumentation we should build into `server.py` to emit placement metadata (perhaps ties into Ray observability upgrades).

### C. Architectural Alternatives
8. **Serve vs. Core Actors**  
   - Should we consider bypassing Serve for specific deployments and managing actors manually to gain finer placement control (while preserving HTTP ingress)?  
   - Compare the complexity vs. benefits, and highlight any hybrid architecture patterns.

9. **Multi-node Aggregation**  
   - Explore the feasibility of running multiple Serve applications or Ray namespaces to segregate different model classes across GPUs.  
   - Discuss whether namespaced scheduling or resource quotas could approximate per-GPU policies.

10. **Containerization & Kubernetes**  
    - Evaluate whether running Ray on Kubernetes with GPU scheduling frameworks (e.g., NVIDIA Device Plugin, GPU Operator) introduces additional placement primitives we should leverage.  
    - If yes, outline the configuration required to map physical GPUs to logical Ray nodes without manual `CUDA_VISIBLE_DEVICES` hacks.

### D. Roadmap & Recommendations
11. **Ray Roadmap**  
    - Summarize upcoming Ray features (Serve, Core, Runtime) that may change the GPU placement story. Prioritize items with concrete timelines or early access.  
    - Identify any Ray issues/PRs that we should monitor or contribute to.

12. **Proposed Implementation Plan**  
    - Provide ranked scenarios (e.g., “Stay the course with PACK,” “Adopt one-node-per-GPU with automation,” “Adopt new Serve API X”) along with pros/cons, prerequisites, and migration steps.  
    - For each viable path, specify expected engineering effort, risks, and validation steps (unit tests, load tests, fallback plans).

13. **Guidance for vLLM & Toolchains**  
    - Offer model-specific advice: for vLLM replicas with large context windows, how should we shape Ray resources, autoscaling settings, and cancellation signals to balance latency vs. GPU saturation?  
    - Address how toolchain workflows (which may spawn long-running jobs with cancellation) interplay with GPU scheduling decisions.

### E. Documentation Requests
14. **Documentation & Playbooks**  
    - Provide a template for an internal “GPU Placement Playbook” that our team can maintain. It should cover cluster startup, placement verification, troubleshooting steps, and escalation paths.  
    - Suggest updates to our `docs/HANDOFF_V4.md` so new engineers inherit clear guidance on GPU scheduling constraints and current best practices.

### F. Open Questions & Risks
15. **Edge Cases**  
    - Are there known Ray bugs or limitations (e.g., PG capture semantics, backpressure loops) that we must architect around if we revisit placement controls?  
    - What failure patterns should we monitor if we adopt more sophisticated scheduling (e.g., PG starvation, autoscaler oscillation, queue buildup)?

16. **Research Gaps**  
    - Identify any experiments the planning model cannot conclusively answer (e.g., because they require hands-on benchmarking). Provide a prioritized list of tests we should run locally once we pick an approach.

## Deliverables
Please deliver:
1. A cohesive report covering all the above questions, with explicit references to Ray documentation, PRs, or community discussions where relevant.
2. Decision matrices comparing potential architectures (single-node multi-GPU vs. one-node-per-GPU vs. Serve bypass, etc.).
3. Concrete recommendations for next steps, including a suggested proof-of-concept plan if we decide to revisit placement control.

## Constraints & Preferences
- Do not assume we can redesign the entire stack; aim for evolutions of the current QueryLake architecture unless a compelling alternative is available.
- Prioritize solutions that keep operational complexity manageable (we run both on-prem and cloud clusters).
- Emphasize reproducibility: any recommended approach should include guidance for automated testing and verification inside CI or staging clusters.

## Thank You
We appreciate the exhaustive diligence. We’ll use your findings to decide whether to re-open the GPU placement project or continue with the existing heuristics.

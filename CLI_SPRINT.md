# QueryLake GPU Placement System - Sprint Summary

This document outlines the work completed to diagnose and fix the GPU resource allocation and placement strategy for the QueryLake platform.

## 1. Goal

The primary objective was to implement a `SPREAD` placement strategy for embedding model replicas to ensure they are distributed evenly across the available GPUs (a 2-2 split on a 2-node, 2-GPU cluster). This would optimize resource utilization and prevent resource contention.

## 2. The Problem: Scheduling Failures

Despite correctly configuring a placement group to reserve resources, Ray Serve deployments were failing to schedule.

- **Symptom:** Embedding model replicas were stuck in a pending state, with Ray reporting that all cluster resources were claimed by actors or reserved in placement groups.
- **Key Evidence:** `ray status` showed that resources (CPU, GPU, VRAM) were successfully *reserved* by the placement group, but the *available* resources for scheduling deployments were near zero. The system could not access the resources it had just reserved.

## 3. The Root Cause: External vs. Internal Placement Groups

The core issue was a fundamental disconnect between how resources were being reserved and how Ray Serve was attempting to use them.

- **The Old Method (External):** A placement group was created externally in `start_querylake.py` before the Ray Serve application was deployed. The placement group was then passed to the Ray Serve run function.
- **The Conflict:** In the version of Ray being used (2.47.1), Ray Serve deployments were unable to "see" or schedule into these externally created placement groups. The resources were locked away, inaccessible to the actors that needed them.

## 4. The Solution: Ray Serve Internal Placement Groups

The solution was to move the placement group creation logic from the orchestration script (`start_querylake.py`) directly into the Ray Serve application definition (`server.py`). This allows Ray Serve to manage the placement groups internally as part of the deployment process.

### Key Implementation Steps:

1.  **Removed External Placement Group Logic:** All code related to creating, managing, and passing `placement_group` objects in `start_querylake.py` was removed.

2.  **Created an Internal Placement Group Helper:** A new helper function, `_get_placement_group_config`, was added to `server.py`. This function is responsible for dynamically creating the necessary placement group configuration based on the chosen strategy.

3.  **The Critical Fix: Replica-Sized Bundles:** A major discovery during implementation was that Ray Serve allocates one entire placement group "bundle" per replica. Our initial internal implementation created node-sized bundles (e.g., `{"GPU": 1, "CPU": 43, "VRAM_MB": 49140}`), which caused failures because a single small replica would try to reserve an entire GPU.

    The final implementation dynamically creates bundles that are sized to the *actual resource requirements of each replica*.

    ```python
    # server.py - The corrected logic
    def _get_placement_group_config(strategy: str, num_replicas: int, replica_resources: Dict) -> Dict:
        # ... logic to get number of GPU nodes ...
        
        # Create bundles sized for each replica, not the whole node
        bundles = [replica_resources for _ in range(num_replicas)]
        
        return {
            "placement_group_bundles": bundles,
            "placement_group_strategy": strategy,
        }
    ```

4.  **Updated All Deployments:** Each model deployment (`embedding`, `llm`, `rerank`, `surya`) was updated to call `_get_placement_group_config` and pass its specific `replica_resources` to generate the correct placement configuration.

## 5. Validation and Outcome

After implementing the changes, we ran the QueryLake application with the `SPREAD` strategy.

- **Result:** **SUCCESS!**
- **All deployments** became `HEALTHY`.
- The embedding models were **successfully distributed** across the available GPUs.
- `ray status` and `serve status` confirmed that all replicas were running and resources were allocated as expected, with no pending resource demands.

The system is now robustly configured to handle GPU placement strategies directly within Ray Serve, resolving the resource conflict and achieving the desired scheduling behavior. 
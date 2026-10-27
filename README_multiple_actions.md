# Agent1: multiple independent actions

Built against agent1.rar + agent1_v2_integration.zip + agent1_conversation_patch.zip supplied in this conversation. This is an overlay patch, not a full project or model download.

## Install in VS Code

1. Save/commit your current Agent1 work. Restart the kernel of notebook 15 to release GPU memory.
2. Extract this archive to a temporary folder. Copy its `src`, `tests`, and `notebooks` folders **into your existing project root**, merging folders and replacing matching files. Do not replace/delete whole directories. If you modified these files after the conversation patch, review `changes.patch` and merge those edits.
3. Open `notebooks/16_test_multiple_actions.ipynb` with the existing Agent1 `.venv` kernel.
4. Run Part A (offline tests). Then run Part B and the live checks in order. Set CLASSIFIER_PATH to the same saved classifier used in notebook 15 if needed.

No installation, classifier retraining, adapter training, or database setup is required in your existing environment.

## Changes

| File | Change |
| --- | --- |
| analysis/action_plan.py | ActionStep with its own arguments; MULTI_ACTION requires >=2 steps. Candidate alternatives no longer represent execution steps. |
| analysis/request_type_normalizer.py | Recognizes the new steps structure. |
| analysis/request_analyzer.py | Multi-step extraction prompt, missing arguments, repeated services, dependency flags, whole pending plan in reply classification. |
| conversation/state.py | Allows a whole multi-action plan to remain pending. |
| execution/multi_action_runner.py | Collect all required arguments, then execute supported independent steps sequentially; preserve individual results. |
| execution/action_executor.py | Dispatch mock stock lookup via availability/product_service. |
| tools/stock_service.py | Fixed demo inventory: blue backpack 5, red backpack 0; unlisted products unknown. |
| agent.py | Dispatch/resume multi-action plans; preserve abandon/switch/clarify flow. |
| generation/response_service.py | One generation call with all labeled results, complete deterministic fallback on inference failure. |
| tests/test_multi_action.py | 21 new offline tests. |
| tests/test_v2_integration.py | Removes obsolete expectation that every multi-action request is rejected. |

Existing SINGLE_ACTION structures remain valid. Historical MULTI_ACTION JSON using only candidate_actions must be reanalyzed; it cannot safely be converted without per-step intents/arguments. No stored plans are migrated automatically.

## Scope and inspection

Supported pairs: track_order/order_service (order_id) and availability/product_service (product, color). Inspect `agent.last_action_plan.steps` and `agent.multi_action_runner.last_results`; `action_runner.last_tool_result` remains the single-action trace. Step indices in results/metadata are zero-based; questions use human-readable task numbers.

All arguments are collected before execution. Arguments are stored under per-step keys so two orders cannot overwrite each other. Supply one requested value at a time during follow-ups. Invalid/multiple order IDs keep the task pending. Product/color follow-ups currently accept the plain value requested, rather than performing general semantic argument extraction.

Unsupported steps produce failed results, while independent supported steps still run. Any explicit depends_on/condition prevents execution of the whole plan. Dependency recognition relies on analyzer output and is not a general dependency planner. Only the two read-only tool pairs are allowlisted. No writes/cancellations, concurrency, retry engine, or databases are introduced.

Stock is mock data, explicitly labeled in responses; fixture order dates are unchanged. Standalone stock questions keep the earlier informational response path. This stage connects the stock tool specifically to multi-action execution.

## Verification

44 offline tests passed in the authoring environment: 21 multi-action tests and 23 existing integration/conversation tests (one obsolete multi-action rejection assertion updated). Tests include missing arguments, repeated service, partial failure, tool exceptions, invalid plans, pending transitions and generator failure.

Notebook code cells were syntax checked and its offline section was run. The actual classifier/base model/V2 adapter were unavailable in the authoring environment, so live notebook checks must be run locally. Offline tests do not establish LLM decomposition or final-answer accuracy. The existing V2 adapter receives a new multi-result shape and may omit or conflate results; inspect its answer alongside the structured records before calling this stage complete. No claim of live model validation is made.

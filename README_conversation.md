# Pending conversation patch

Apply **after** agent1_v2_integration.zip. Merge `src/`, `tests/`, and `notebooks/` into the project root, replacing only the two supplied source modules. Save existing changes in Git first. No data, adapter, or experiment folders move.

## Behaviour
- A bare order ID continues tracking without another analyzer call.
- Other replies are classified by the existing base-model analyzer as continue, abandon, new_request, or clarify.
- Abandoning a pending tracking request clears it; it does not cancel a customer order.
- A new request is analyzed using the new message, avoiding stale tracking context. Full history remains stored. The old task is cleared only after a valid new action plan is obtained.
- Ambiguous replies or invalid decision JSON keep the pending task and ask a focused clarification.
- An ID inside a different operation, such as "Cancel ORD-123", is not automatically treated as tracking input.

## Run
Restart the kernel after replacing modules; open notebook 15 with the Agent1 kernel. Its setup is independent of notebook 14; it uses the same classifier, adapter and knowledge paths. Do not keep both notebooks' GPU models loaded.

From the project root in PowerShell:
```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -p test_pending_conversation.py -v
```

The original V2 integration test double needs `classify_pending_reply` added for non-bare ID follow-ups; the included replacement test_v2_integration.py preserves its tests and adds that method.

This is a bounded conversation change. Real LLM classification can still be wrong. No task stack, broad reference resolution, multi-action executor or persistence is added here. Next milestone: multiple actions.

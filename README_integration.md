# RetailFlow V2 integration

> Historical implementation notes for RetailFlow. For current installation and startup, use the [main README](readme.md). Original archive names, paths, commands, and notebook references are retained.

Copy the `src`, `notebooks`, and `tests` contents into the matching folders of your existing RetailFlow project. This is a patch, not a replacement for the entire project. Keep the original files in Git before applying it. No models or data are bundled.

## Changed modules
- `src/agent1/agent.py`: inject an optional response service; connect informational routes to knowledge; preserve the existing tracking, clarification and unsupported-action paths.
- `src/agent1/execution/action_runner.py`: send real tool results to the response service, preserve the original question across ID collection, clear pending execution before response generation. Missing identifiers still trigger deterministic questions.
- `src/agent1/generation/response_service.py`: new shared grounded response layer; explicit no-live-stock response; no-source fallback; trace fields and deterministic fallback if generation fails.
- `src/agent1/generation/response_generator.py`: use the base tokenizer, as in the successful benchmark, and a default 128-token response budget.
- `src/agent1/analysis/request_analyzer.py`: retain the specific intent for informational requests so knowledge selection can work.

Existing constructor calls remain supported; passing no response service preserves the earlier workflow. The generator's prompt builder and the actual tool implementations are unchanged.

## Run
1. Restart kernels holding old modules/models. Open notebook 14 with the RetailFlow `.venv` kernel.
2. Select your saved ensemble path if the first cell finds several classifier files.
3. Run the setup and demonstration cells. The notebook loads one Llama model and temporarily disables its adapter for request analysis. Sentence embeddings run on CPU.
4. Keep adapter files at `models/adapter_v2/` and knowledge at `data/knowledge/demo_retail_knowledge_v1.json`.

## Validation
From the project root in the existing virtual environment, PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -p test_v2_integration.py -v
```

The included tests use real orchestration, router, state, argument collector and mock order service, with controlled classifier/analyzer/generator doubles. They test missing and ambiguous IDs, successful and failed tools, knowledge delivery, unknown stock, unsupported cancellation, inference failure, reset and backward compatibility. They do not establish real LLM routing or answer accuracy.

## Portfolio scope
V2 met core requirements in 22/24 manually reviewed response-only diagnostic cases versus 21/24 for the base model. It regressed on asking for a missing order ID and on distinguishing unknown stock from out-of-stock. This integration handles those conditions in control flow; that does not change the raw benchmark score or guarantee correct routing.

Training/validation share templates and the diagnostic influenced V2 design. These are development results, not an untouched final test. The order service is mocked, knowledge retrieval is intent-based, cancellation is unsupported, multi-action execution is unsupported, and pending-action topic changes are not yet handled. Live local classifier/base-analyzer/V2 integration must be exercised in notebook 14.

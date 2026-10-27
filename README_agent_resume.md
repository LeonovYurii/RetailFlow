# Agent1 SQLite history — step 2

Prerequisite: install step 1 (src/agent1/persistence/sqlite_store.py). Merge this ZIP's src, notebooks, and tests folders into your Agent1 root. All files are new. Do not replace agent.py, request_analyzer.py, or either runner. No git apply command is needed.

## First check: no model required

Open notebooks/18_test_agent_resume.ipynb and run parts A–C with your normal Agent1 Python environment. It creates a pending two-action fixture, saves an Agent checkpoint, restores and completes the real mock lookups in a separate Python process, saves completion, and reopens the completed history. It does not call an LLM or load model weights.

## Live check: reuse notebook 16's model

If notebook 16's working agent is still in memory, paste notebooks/18_live_resume_cell.py into a cell at the bottom of that notebook and run it. Do not restart or rerun model setup. The code uses the existing model, classifier, router, analyzer, context builder, and response service; it creates only new lightweight PersistentAgent instances.

Do not execute archived experimental wrapper cells. If notebook 16's kernel has already stopped, run its regular final setup once before the live check. Different notebooks do not automatically share kernels/variables. Part D of notebook 18 also displays this live code as Markdown.

Expected live flow: ask for order number -> save -> restore into a new agent -> reply ORD-123 -> two successful tool results -> grounded combined answer -> save completed state -> restore without pending tasks. Review the answer for shipped ORD-123 and five blue backpacks labelled mock inventory. Assertions alone do not evaluate answer quality.

## API

    from agent1.persistent_agent import PersistentAgent
    from agent1.persistence.checkpoint_store import SQLiteCheckpointStore

Construct PersistentAgent just like Agent, plus:

    checkpoint_store=SQLiteCheckpointStore(database_path)

Then explicitly call:

    agent.save_conversation('conversation-123')
    another_agent.load_conversation('conversation-123')

load_conversation returns the agent and raises KeyError if no checkpoint exists. Loading does not call the model or execute tools. It replaces the receiving instance's in-memory conversation; prefer a fresh instance. The existing Agent class remains available unchanged.

## Stored data

The existing ConversationState snapshot plus single-action request text, multi-action request text, and _fresh_request. Restore binds both runners to the same restored state. Per-call diagnostics are cleared; historical tool results remain in the message history. The pending plan is available as last_action_plan for inspection.

Checkpoint data uses a separate agent_checkpoints table in data/local/conversations.sqlite3. Step-1 conversation_snapshots rows remain unchanged. A state-only snapshot is not enough to restore the runner context, so there is no implicit migration. Keep /data/local/ in .gitignore.

## Boundaries

Save explicitly after a completed handle_message call, including after successful execution or abandonment. Same-ID saves replace the prior checkpoint; last writer wins. Saving an old in-memory instance can overwrite newer progress. There is no automatic save, concurrency coordination, or exactly-once protection for a crash between tool execution and save. Restoring a stale pending checkpoint can repeat its lookups; scope remains independent read-only mock tools. No write actions are introduced.

Model weights and dependencies are not stored. After a real restart the runtime/model still needs initialization. Reusing a loaded model for this test avoids that cost but does not make the model itself persistent.

## Validation performed

Eight offline tests cover multi-action resume and saved completion, single-action resume, fresh-request state after abandonment, replacement by a new question, non-destructive failed restore, preservation of step-1 rows, clarification routing, and invalid runtime data. Notebook parts A–C passed, including restore and execution in a separate Python process. Live model output requires the local live check.

Optional offline tests (Agent1 must be importable in the environment):

    python -m unittest discover -s tests -p test_agent_checkpoints.py -v

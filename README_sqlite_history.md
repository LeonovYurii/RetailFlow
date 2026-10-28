# SQLite conversation history — step 1

> Historical implementation notes for RetailFlow. For current installation and startup, use the [main README](readme.md). Original archive names, paths, commands, and notebook references are retained.

Merge the ZIP's src, notebooks and tests folders into the matching folders in your RetailFlow project. These are new files only; no existing source files need replacement. There is no temporary installation folder or git apply step.

Open notebooks/17_test_sqlite_history.ipynb with your usual RetailFlow Python environment and run its cells in order. No LLM, CUDA setup, model load, or training is required. SQLite is included with Python; existing project dependencies (including Pydantic) are reused. The notebook creates data/local/conversations.sqlite3 automatically.

Add this line to your project's .gitignore:

    /data/local/

The database holds local conversation data and should not be committed. Commit the new source, notebook, tests, and this README.

The first stage uses a JSON snapshot stored under conversation_id. It preserves the complete ConversationState, including message order, role/type enums, tool-result text, JSON metadata, clarification routing, pending ActionPlan, collected arguments, and required argument names. Save replaces the latest snapshot for that ID in a transaction. Load returns a separate state object or None for an unknown ID. Metadata must be JSON-compatible; arbitrary Python objects are rejected.

The notebook demonstrates a pending two-action fixture, saves it, reads it from a separate Python process, compares the restored state and context, then updates the collected order ID and saves again. This verifies persistence, not live semantic planning.

Seven offline unit tests cover pending-state round trips, message metadata and Unicode, clarification routing, update/isolation, failed serialization preserving the previous row, unsupported snapshot versions, and invalid IDs. They have already been run; running them yourself is optional:

    python -m unittest discover -s tests -p test_sqlite_conversation_store.py -v

Run that command in an environment where RetailFlow is installed or src is on PYTHONPATH.

Compatibility: the four newly uploaded files were inspected. The existing ChatMessage and routing dataclass definitions from the earlier project copy were used for serialization; these files are neither modified nor bundled. If you have changed those definitions, report any error rather than replacing them.

Next step: explicit Agent checkpoint integration. ConversationState alone is not a complete Agent checkpoint: runners retain state references/request text, and Agent retains _fresh_request. Do not assign a restored state directly to a running Agent. No tools are executed or automatically replayed by this storage module. Concurrent writer coordination, automatic save, and crash recovery during tool execution are outside this first step.

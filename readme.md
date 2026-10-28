# RetailFlow — Retail support agent

RetailFlow combines intent classification, LLM planning, business tools, grounded response generation, and persistent conversations. A local Gradio interface lets you ask retail questions, track orders, execute multiple independent lookups, and resume unfinished requests after reopening a conversation.

Order and inventory tools currently use **fixtures, not operational business data**. Store information comes from a small retail knowledge file. Conversation storage supports SQLite and MongoDB; it is separate from business-data storage.

## Project name

**RetailFlow** is the public project name. The Python package remains named
`agent1` to preserve imports and compatibility with the saved classifier.
Commands such as `python -m agent1.chat`, the `agent1-chat` entry point,
database names, and existing artifact filenames are unchanged. Older notebooks,
transcripts, and the current interface may still display the original name.

## Capabilities

- TF-IDF and sentence-embedding classifiers combined into an intent ensemble.
- Structured LLM plans validated with Pydantic, including single actions, multiple actions, clarification, and informational responses.
- Order tracking and stock lookups within multi-action requests, with separate results for each task.
- Missing-argument collection, abandonment, and topic changes while a request is pending.
- Grounded responses generated using a QLoRA-tuned response adapter.
- Automatic conversation checkpoints, including pending arguments and the state required to resume execution.
- Gradio browser chat and an optional console interface, using the same session layer.

## Architecture

```mermaid
flowchart TD
    A[Browser or console] --> B[Chat session]
    B --> C[Intent ensemble and LLM planner]
    C --> D[Validated plan and argument collection]
    D --> E[Tools or intent-selected knowledge]
    E --> F[Grounded response]
    F --> G[Conversation checkpoint]
    G --> H[SQLite or MongoDB]
    G --> A
```

The planner uses the base language model with its response adapter temporarily disabled. Response generation uses the adapter. Models are loaded once per process; inference and session mutations are serialized. Each browser tab has separate in-memory conversation state. Mutable analyzer and response-service diagnostics are created per agent, while model weights are shared.

## Requirements and model assets

Use Python 3.12 and a CUDA-enabled PyTorch installation on a compatible NVIDIA GPU. The existing response generator loads the base model in 4-bit NF4 using bitsandbytes. **This launcher does not provide CPU-only or Ollama inference.** Loading can take several minutes, and first use may download base-model and embedding weights. Close any notebook kernel holding the same model before starting the app on a memory-constrained GPU.

The repository does not include the trained classifier or adapter weights. A source checkout alone is not sufficient for live inference. Supply these assets from your existing RetailFlow setup or obtain a compatible artifact bundle from the maintainer:

| Asset | Default location | Notes |
| --- | --- | --- |
| Fitted intent ensemble | `models/intent_classifier/ensemble.joblib` | Load only trusted joblib/pickle files. Use compatible scikit-learn/joblib versions from the artifact-producing environment. |
| Response adapter | `models/adapter_v2/` | Must include `adapter_config.json` and adapter weights, normally `adapter_model.safetensors`. |
| Base language model | Resolved from the adapter config | Requires access to the referenced model and acceptance of any applicable model terms. |
| Sentence embedding model | Resolved from the saved classifier | Downloaded/cached by sentence-transformers; runs on CPU. |
| Retail knowledge | `data/knowledge/demo_retail_knowledge_v1.json` | Included in this repository. |

The classifier and fine-tuning notebooks document artifact creation. Historical dependency versions are recorded under `experiments/`; they are provenance records, not a universal environment lock. There is currently no published, versioned model-bundle download linked from this repository.

## Install and launch

From the repository root, activate your Python environment. In an existing working RetailFlow environment, install the interface without upgrading the inference stack:

```bash
python -m pip install -e ".[chat]"
python -m agent1.chat --check-config
python -m agent1.chat
```

For a new environment, install a CUDA-enabled PyTorch build appropriate for your system using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the inference dependencies as well:

```bash
python -m pip install -e ".[chat,inference]"
```

If the base model requires authentication, authenticate with Hugging Face before launching. Model downloads, access permissions, GPU compatibility, and classifier serialization compatibility are separate from the local path check.

The app opens at **http://127.0.0.1:7860**. It binds only to loopback and does not create a public sharing link. The terminal remains running; stop it with Ctrl+C after the current response has completed and saved.

Alternative locations and the console interface:

```bash
python -m agent1.chat --classifier path/to/ensemble.joblib --adapter path/to/adapter
python -m agent1.chat --port 7861
python -m agent1.chat --console
```

Paths are relative to the current working directory unless absolute. `agent1-chat` is also installed as a command. `--help` and `--check-config` do not load models or connect to the database.

## Quick interface guide

| Action | What to do |
| --- | --- |
| Start | Run `python -m agent1.chat` and wait for model loading. |
| Open the interface | Visit http://127.0.0.1:7860 if the browser does not open automatically. |
| Send a request | Type a message and click **Send**. |
| Resume | Click **Refresh list**, select a saved conversation, then **Open**. |
| Close the page | Close the browser tab; the Python server keeps running. |
| Stop completely | Wait for **Saved**, then press **Ctrl+C** in the terminal that launched the app. |

Reopening the page while the server is running does not reload the model.
Restarting the Python process does. Saved conversations survive a restart.

## Conversations and recovery

Send a message to start, or select a saved conversation and click **Open**. **New conversation** creates a separate checkpoint. **Refresh list** updates the saved-conversation selector (latest 200). The current conversation ID is shown above the transcript.

Successful turns save automatically. Reopening a conversation restores its messages and pending request. Browser refresh resets the UI's in-memory session; use Open to restore saved work. The list belongs to this local installation, not to individual authenticated users. Two tabs cannot actively edit the same conversation in one app process: use **Release conversation** in the first tab before opening it in another. Do not run multiple app processes against the same conversations.

If processing fails, the interface blocks further turns until you explicitly check **Discard unsaved changes** and click **Discard and reload**. This restores the previous checkpoint; it does not fix the underlying planning error. If processing succeeds but saving fails, the answer stays visible and **Retry save** retries persistence without executing the turn again. Closing a browser before a failed save is recovered can lose unsaved progress.

Console commands are `/new`, `/list`, `/open ID`, `/history`, `/retry-save`, `/reload`, and `/quit`.

## SQLite and MongoDB

SQLite is the default, at `data/local/conversations.sqlite3`. No database server is needed. Use `--sqlite-path` to choose another file.

For MongoDB:

```bash
python -m pip install -e ".[mongo]"
python -m agent1.chat --backend mongodb --mongo-database agent1_demo
```

Set `MONGODB_URI` in the process environment **before** launching. The launcher does not read `.env` automatically. For Atlas, configure a database user with read/write privileges on the selected database and allow your client IP. URL-encode password characters in the URI. Do not commit or print credentials.

In PowerShell, this prompts privately for the complete URI and removes it from the environment after the app stops:

```powershell
$mongoSecret = Read-Host "MongoDB URI" -AsSecureString
$mongoCredential = [System.Net.NetworkCredential]::new("", $mongoSecret)
$env:MONGODB_URI = $mongoCredential.Password
try {
    python -m agent1.chat --backend mongodb --mongo-database agent1_demo
} finally {
    Remove-Item Env:MONGODB_URI
    Remove-Variable mongoSecret, mongoCredential
}
```

Selecting a different backend does not migrate conversations. MongoDB stores one checkpoint document per conversation, including structured state and runtime fields; SQLite stores the equivalent checkpoint as JSON.

## Example

```text
You: Track my order and check whether the blue backpack is available.
Agent1: For task 1 (track_order), please provide one order number, such as ORD-123.

[Reopen the saved conversation, then continue.]

You: ORD-123
Agent1: Demo stock results use mock inventory.
Your order, ORD-123, is shipped. The blue backpack is available at 5 units.
```

Exact phrasing may vary. Fixture dates are fixed, not recalculated relative to today.

Other useful requests: “Track order ORD-123 and track order ORD-456”, “What time does Central close on Saturday?”, and “Never mind” while an order number is pending. ORD-999 is unknown. A red backpack has zero stock; an unlisted item has unknown stock, which is not the same as zero.

## Connecting operational data

Replace the fixture lookups in `src/agent1/tools/order_service.py` and `stock_service.py` with database queries or service API calls. Preserve their input contracts and `ToolResult` outputs, including explicit failures and unknown inventory. This lets planning, argument collection, response grounding, and conversation persistence continue using the same interfaces.

Real integration still requires credentials, authorization, schema mapping, and error handling. Changing the conversation backend to MongoDB does **not** connect live order or supply data. The knowledge source can likewise be replaced or extended through the existing knowledge-store interface.

## Validation and artifacts

Run the offline regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests use controlled analyzer responses and fixture tools. They check planning schemas, control flow, tool results, checkpoint restoration, UI session isolation, and failure recovery; they do not measure live LLM accuracy. The Gradio construction test runs when the chat extra is installed.

The `notebooks/` directory records classifier evaluation, response fine-tuning, grounding comparisons, multi-action tests, and persistence checks. `experiments/` contains training metrics and before/after responses. These component-level results should not be read as an end-to-end production accuracy claim. `data/evaluation/agent1_grounding_benchmark_v1.json` and the grounding comparisons provide concrete response-evaluation cases.

## Scope

- Read-only operations; no real order cancellation, refunds, or purchase transactions.
- Independent multi-action steps executed sequentially; dependency/condition scheduling is unsupported.
- Single stock questions retain the informational response path; the fixture stock tool is connected to multi-action execution.
- Informational retrieval selects documents by intent; it is not a general web search or vector database.
- The LLM can still return an invalid plan. Strict validation and explicit failure recovery remain necessary.
- Local, single-operator interface; no authentication, tenant isolation, or distributed concurrency control.
- Checkpoints support conversational resume, not exactly-once recovery for external write operations.

The focused implementation and its tests demonstrate the complete request-to-response flow, including persistence and recovery, without requiring a notebook to run the agent.

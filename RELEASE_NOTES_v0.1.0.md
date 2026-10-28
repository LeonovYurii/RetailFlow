# RetailFlow v0.1.0

Built with Llama.

Retail support agent with structured planning, read-only tool execution,
grounded responses, persistent conversations, and a local Gradio interface.

## Included
- Single and multiple independent requests, with missing-argument collection.
- Order tracking and fixture-based inventory lookups.
- Topic changes, abandonment, and resumable pending conversations.
- SQLite or MongoDB checkpoint storage.
- Gradio and console launchers.
- Trained intent ensemble and Llama-RetailFlow response adapter v2,
  provided as a separate model-bundle asset.

## Setup
Clone/download the source, download retailflow-models-v0.1.0.zip, and extract it
into the repository root. Follow readme.md for environment setup and model access.
The .sha256 asset contains the archive checksum. GitHub's automatic source archive
contains code only.

The application uses fixtures rather than live supply/order data. Connecting a
business database/API requires implementing the existing business-tool contracts.
Conversation storage is a separate integration.

## Validation
77 offline tests passed during launcher development, including session recovery,
Gradio callbacks and component encoding. Local HTTP startup was checked. The
maintainer reports successful local Gradio testing with the supplied environment.
A clean third-party environment and live model inference were not retested by
the packaging process. The archived model payloads are unchanged.

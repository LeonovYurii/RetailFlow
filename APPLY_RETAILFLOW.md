# Apply RetailFlow documentation

Based on GitHub commit 8a48d59eee7a2a7c1912594953b2891b7180063f.

Copy the eight README files from this archive into the root of your existing
repository, replacing the corresponding files. No Python files, notebooks,
model assets, database files, or package identifiers are changed.

The public name is RetailFlow; agent1 remains the internal package name.
Historical transcripts and archive names are preserved. The main README now
includes a short guide to starting, reopening, and stopping the Gradio app.

Review with git diff and commit/push yourself. No GitHub changes were made.
Renaming the GitHub repository is a separate action for the owner; this archive
does not perform it. No live-model rerun or kernel restart is needed.

Next: prepare ensemble.joblib and the complete adapter_v2 folder as a versioned
model bundle, along with compatible environment versions and artifact checksums.
Do not include database files, credentials, or the entire Hugging Face cache.

"""Launch Agent1 locally: python -m agent1.chat --help."""
import argparse
from pathlib import Path
from uuid import uuid4

from agent1.runtime import RuntimeConfig, load_runtime


def build_app(runtime, backend="sqlite"):
    import gradio as gr
    from agent1.web_controller import ChatController

    controller = ChatController(runtime)
    with gr.Blocks(title="Agent1 · Retail assistant", analytics_enabled=False) as app:
        token = gr.State(value=lambda: uuid4().hex, delete_callback=controller.release)
        gr.Markdown("# Agent1\nOrder tracking, retail questions, and conversations you can resume.")
        gr.Markdown(f"**Storage: {backend}** · Order and inventory results use demo fixtures.")
        with gr.Row():
            saved = gr.Dropdown(label="Saved conversations", choices=[], scale=5)
            refresh = gr.Button("Refresh list", scale=1)
            open_button = gr.Button("Open", scale=1)
            new_button = gr.Button("New conversation", scale=1)
        conversation = gr.Textbox(label="Current conversation", interactive=False)
        chat = gr.Chatbot(label="Conversation", height=440)
        with gr.Row():
            message = gr.Textbox(label="Message", placeholder="Where is my order?", scale=6)
            send = gr.Button("Send", variant="primary", scale=1)
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Accordion("Recovery and session controls", open=False):
            gr.Markdown("Retry save does not repeat model or tool execution. "
                        "Reload discards the unsaved turn. Release allows another tab "
                        "to open this conversation.")
            retry = gr.Button("Retry save")
            confirm = gr.Checkbox(label="Discard unsaved changes", value=False)
            reload_button = gr.Button("Discard and reload")
            release = gr.Button("Release conversation")
        gr.Examples(
            examples=[
                ["Track order ORD-123."],
                ["Track my order and check whether the blue backpack is available."],
                ["What time does Central close on Saturday?"],
            ],
            inputs=message,
        )

        def callback(action):
            def run(key, text, selected, discard):
                result = controller.handle(
                    key, action, selected if action == "open" else text, discard
                )
                ids = [cid for _, cid in result.choices]
                selection = result.conversation_id if result.conversation_id in ids else None
                return (
                    result.history, result.status, result.conversation_id,
                    gr.Dropdown(choices=result.choices, value=selection),
                    "" if result.clear_message else text, False,
                )
            return run

        inputs = [token, message, saved, confirm]
        outputs = [chat, status, conversation, saved, message, confirm]
        options = dict(
            inputs=inputs, outputs=outputs,
            concurrency_id="agent1-runtime", concurrency_limit=1,
            api_visibility="private",
        )
        for button, action in (
            (send, "send"), (new_button, "new"), (open_button, "open"),
            (refresh, "refresh"), (retry, "retry"), (reload_button, "reload"),
            (release, "release"),
        ):
            button.click(callback(action), **options)
        message.submit(callback("send"), **options)
        app.load(callback("refresh"), **options)
    app.queue(default_concurrency_limit=1, max_size=16)
    return app


def parser():
    result = argparse.ArgumentParser(description="Agent1 local retail chat")
    result.add_argument("--classifier", type=Path,
                        default=Path("models/intent_classifier/ensemble.joblib"))
    result.add_argument("--adapter", type=Path, default=Path("models/adapter_v2"))
    result.add_argument("--knowledge", type=Path,
                        default=Path("data/knowledge/demo_retail_knowledge_v1.json"))
    result.add_argument("--backend", choices=["sqlite", "mongodb"], default="sqlite")
    result.add_argument("--sqlite-path", type=Path,
                        default=Path("data/local/conversations.sqlite3"))
    result.add_argument("--mongo-database", default="agent1_demo")
    result.add_argument("--port", type=int, default=7860)
    result.add_argument("--console", action="store_true", help="Use terminal chat instead of Gradio")
    result.add_argument("--check-config", action="store_true",
                        help="Check local asset paths; do not load models or connect to storage")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not 1 <= args.port <= 65535:
        cli.error("--port must be between 1 and 65535")
    config = RuntimeConfig(
        args.classifier, args.adapter, args.knowledge, args.sqlite_path,
        args.backend, args.mongo_database,
    )
    try:
        config.validate()
    except (ValueError, OSError) as exc:
        cli.exit(2, f"Configuration: {exc}\n")
    if args.check_config:
        print("Local configuration checks passed. Models and database were not loaded.")
        return
    print("Loading Agent1 models once. First startup may download base/embedding weights.")
    try:
        runtime = load_runtime(config)
    except Exception as exc:
        cli.exit(1, f"Startup failed ({type(exc).__name__}). Check model assets, CUDA, "
                 "dependencies and storage configuration; see readme.md.\n")
    try:
        if args.console:
            from agent1.chat_console import run_chat
            run_chat(runtime.new_session())
        else:
            app = build_app(runtime, backend=args.backend)
            try:
                app.launch(server_name="127.0.0.1", server_port=args.port,
                           share=False, inbrowser=True, show_error=False)
            finally:
                app.close()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

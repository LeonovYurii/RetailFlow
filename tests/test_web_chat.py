"""Browser controller regressions use real sessions/tools and fixture planning."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent1.runtime import Runtime, RuntimeConfig
from agent1.web_controller import ChatController
from agent1.persistence.sqlite_session_store import SQLiteSessionStore
from test_sessions import factory


class WebChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SQLiteSessionStore(Path(self.temp.name) / "history.sqlite3")
        self.runtime = Runtime(self.store, factory)
        self.controller = ChatController(self.runtime)

    def test_pending_resume_in_different_tab(self):
        pending = self.controller.handle("a", "send", "Track my order and check blue backpack")
        self.assertIn("order number", pending.history[-1]["content"])
        cid = pending.conversation_id
        denied = self.controller.handle("b", "open", cid)
        self.assertIn("another tab", denied.status)
        self.assertEqual(denied.conversation_id, "")
        self.controller.handle("a", "release")
        restored = self.controller.handle("b", "open", cid)
        self.assertEqual(restored.history, pending.history)
        completed = self.controller.handle("b", "send", "ORD-123")
        self.assertEqual(completed.status, "Saved.")
        self.assertIn("shipped", completed.history[-1]["content"])
        self.assertIn("5", completed.history[-1]["content"])
        self.assertTrue(all(m["role"] in {"user", "assistant"} for m in completed.history))
        self.assertFalse(self.store.load_checkpoint(cid)[0].has_pending_action())

    def test_browser_histories_are_separate(self):
        a = self.controller.handle("a", "send", "First request")
        b = self.controller.handle("b", "new")
        self.assertNotEqual(a.conversation_id, b.conversation_id)
        self.assertEqual(b.history, [])
        self.assertEqual(self.controller.handle("a", "refresh").history, a.history)

    def test_save_retry_does_not_repeat_turn(self):
        self.controller.handle("a", "send", "Track my order and check blue backpack")
        session = self.controller.sessions["a"]
        with patch.object(self.store, "save_checkpoint", side_effect=OSError("secret")):
            failed = self.controller.handle("a", "send", "ORD-123")
            self.assertIn("saving failed", failed.status)
            count = len(session.agent.state.messages)
            self.assertIn("blocked", self.controller.handle("a", "new").status)
            self.assertIn("blocked", self.controller.handle("a", "release").status)
        retried = self.controller.handle("a", "retry")
        self.assertIn("not executed again", retried.status)
        self.assertEqual(len(session.agent.state.messages), count)

    def test_processing_failure_requires_explicit_discard(self):
        self.controller.handle("a", "new")
        session = self.controller.sessions["a"]
        with patch.object(session.agent, "handle_message", side_effect=ValueError("secret")):
            failed = self.controller.handle("a", "send", "Hello")
        self.assertIn("Processing failed", failed.status)
        self.assertNotIn("secret", failed.status)
        self.assertIn("confirmation", self.controller.handle("a", "reload").status)
        restored = self.controller.handle("a", "reload", confirm=True)
        self.assertEqual(restored.history, [])
        self.assertIsNone(session._blocked)

    def test_release_cleanup_and_failed_open_preserve_history(self):
        old = self.controller.handle("a", "send", "Track my order")
        failed = self.controller.handle("a", "open", "absent")
        self.assertEqual(failed.conversation_id, old.conversation_id)
        self.assertEqual(failed.history, old.history)
        self.controller.release("a")
        self.assertNotIn(old.conversation_id, self.controller.owners)
        self.assertEqual(self.controller.handle("b", "open", old.conversation_id).history,
                         old.history)

    def test_configuration_fails_before_model_loading(self):
        root = Path(self.temp.name)
        config = RuntimeConfig(root/"absent", root/"adapter", root/"knowledge.json",
                               root/"history.sqlite3")
        with self.assertRaisesRegex(ValueError, "Classifier not found"):
            config.validate()

    def test_gradio_build_if_installed(self):
        import importlib.util
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Install the chat extra for the Gradio construction check.")
        from agent1.chat import build_app
        app = build_app(self.runtime)
        self.addCleanup(app.close)
        config = app.get_config_file()
        self.assertEqual(config["title"], "Agent1 · Retail assistant")
        self.assertTrue(any(c["type"] == "chatbot" for c in config["components"]))
        self.assertEqual(len(self.controller.sessions), 0)
        # Exercise the actual Gradio-bound send callback and component encoding.
        send = next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "run")
        first = send.fn("browser", "Track my order and check blue backpack", None, False)
        self.assertIn("order number", first[0][-1]["content"])
        second = send.fn("browser", "ORD-123", None, False)
        self.assertIn("shipped", second[0][-1]["content"])
        for component, value in zip(send.outputs, second):
            if not isinstance(value, dict) and value.__class__.__name__ != "Dropdown":
                component.postprocess(value)


if __name__ == "__main__":
    unittest.main()

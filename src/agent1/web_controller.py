"""Local browser-session controller, independent of Gradio and model loading."""
from dataclasses import dataclass
from threading import RLock

from agent1.session import UnsavedTurnError, SessionBlockedError


class InterfaceError(ValueError):
    """A safe, locally authored message for the browser."""


@dataclass
class View:
    history: list
    conversation_id: str
    status: str
    choices: list
    clear_message: bool = False


class ChatController:
    """One process / one local operator; browser tabs have independent state.

    A global lock serializes all model and session mutations. Conversation
    ownership prevents two tabs from overwriting the same saved checkpoint.
    The saved conversation list is intentionally shared, not an account system.
    """
    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = RLock()
        self.sessions = {}
        self.owners = {}

    def release(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
            if session is not None:
                cid = session.conversation_id
                if self.owners.get(cid) == key:
                    self.owners.pop(cid, None)

    def _claim(self, key, session, previous):
        if self.owners.get(previous) == key:
            self.owners.pop(previous, None)
        self.owners[session.conversation_id] = key

    def handle(self, key, action, value="", confirm=False):
        with self.lock:
            if key not in self.sessions:
                self.sessions[key] = self.runtime.new_session()
            session = self.sessions[key]
            status = "Choose a saved conversation or send a message to start."
            clear = False
            try:
                if action == "new":
                    previous = session.conversation_id
                    session.new()
                    self._claim(key, session, previous)
                    status = "New conversation."
                elif action == "open":
                    if not value:
                        raise InterfaceError("Choose a saved conversation first.")
                    owner = self.owners.get(value)
                    if owner is not None and owner != key:
                        raise InterfaceError("This conversation is open in another tab. "
                                         "Use Release conversation there first.")
                    previous = session.conversation_id
                    session.open(value)
                    self._claim(key, session, previous)
                    status = "Saved conversation restored."
                elif action == "send":
                    if not isinstance(value, str) or not value.strip():
                        raise InterfaceError("Enter a message.")
                    if session.agent is None:
                        session.new()
                        self._claim(key, session, None)
                    session.send(value.strip())
                    clear = True
                    status = "Saved."
                elif action == "retry":
                    session.retry_save()
                    status = "Saved. The operation was not executed again."
                elif action == "reload":
                    if not confirm:
                        raise InterfaceError("Check the discard confirmation before reloading.")
                    session.reload_saved()
                    status = "Last saved checkpoint restored; unsaved changes discarded."
                elif action == "release":
                    # Never silently lose a completed turn that has not been saved.
                    session._ready()
                    self.release(key)
                    session = self.runtime.new_session()
                    self.sessions[key] = session
                    status = "Conversation released. You can open it in another tab."
                elif action != "refresh":
                    raise InterfaceError("Unknown interface action.")
            except UnsavedTurnError:
                status = ("Answer produced, but saving failed. Use Retry save. "
                          "Do not resend the message, or confirm Discard and reload.")
                clear = True
            except SessionBlockedError:
                status = ("Conversation is blocked after a failure. Use Retry save "
                          "for a save failure, or confirm Discard and reload.")
            except Exception as exc:
                # Do not expose connection strings or database credentials in UI.
                if action == "send" and session._blocked == "processing":
                    status = (f"Processing failed ({type(exc).__name__}); no automatic retry. "
                              "Confirm Discard and reload before continuing.")
                elif isinstance(exc, InterfaceError):
                    status = str(exc)
                else:
                    status = f"Operation failed ({type(exc).__name__}). No automatic retry."
            try:
                choices = [
                    (f"{row['preview'] or '(empty)'} — {row['conversation_id']}",
                     row["conversation_id"])
                    for row in session.list_conversations(limit=200)
                ]
            except Exception as exc:
                choices = []
                status += f" Conversation list unavailable ({type(exc).__name__})."
            history = [
                message for message in session.history()
                if message["role"] in {"user", "assistant"}
            ]
            return View(history, session.conversation_id or "", status, choices, clear)

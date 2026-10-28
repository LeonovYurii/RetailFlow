"""Single-user synchronous sessions with explicit save-failure recovery."""
from uuid import uuid4


class UnsavedTurnError(RuntimeError):
    def __init__(self, answer):
        super().__init__('The turn completed but was not confirmed saved. Use retry_save(); do not resend it.')
        self.answer = answer


class SessionBlockedError(RuntimeError):
    pass


class ChatSession:
    """factory(store) returns a NEW PersistentAgent sharing your loaded models.

    Successful turns are saved automatically. Failed processing blocks further
    turns until reload_saved(); failed saving retains the answer and allows a
    save-only retry. No automatic retry of model or tool execution occurs.
    One active caller/writer per conversation is assumed.
    """
    def __init__(self, store, agent_factory):
        self.store = store
        self.agent_factory = agent_factory
        self.agent = None
        self.conversation_id = None
        self._blocked = None
        self._unsaved_answer = None

    def _ready(self):
        if self._blocked:
            raise SessionBlockedError('Session blocked: use retry_save() after a save error, or reload_saved() to discard the unsaved turn.')

    def _new_agent(self):
        agent = self.agent_factory(self.store)
        if agent is self.agent or agent.checkpoint_store is not self.store:
            raise ValueError('Factory must return a fresh PersistentAgent using the supplied store.')
        return agent

    def new(self):
        self._ready()
        agent = self._new_agent()
        cid = 'chat-' + uuid4().hex
        agent.save_conversation(cid)
        self.agent, self.conversation_id = agent, cid
        return cid

    def open(self, conversation_id):
        self._ready()
        agent = self._new_agent()
        agent.load_conversation(conversation_id)
        self.agent, self.conversation_id = agent, conversation_id
        return conversation_id

    def send(self, text):
        self._ready()
        if self.agent is None:
            raise RuntimeError('Start or open a conversation first.')
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Message must be a nonempty string.')
        try:
            answer = self.agent.handle_message(text)
        except Exception:
            self._blocked = 'processing'
            raise
        self._blocked = 'save'
        self._unsaved_answer = answer
        try:
            self.agent.save_conversation(self.conversation_id)
        except Exception as exc:
            raise UnsavedTurnError(answer) from exc
        self._blocked = None
        self._unsaved_answer = None
        return answer

    def retry_save(self):
        if self._blocked != 'save':
            raise RuntimeError('There is no completed unsaved turn to retry.')
        try:
            self.agent.save_conversation(self.conversation_id)
        except Exception as exc:
            raise UnsavedTurnError(self._unsaved_answer) from exc
        answer = self._unsaved_answer
        self._blocked = None
        self._unsaved_answer = None
        return answer

    def reload_saved(self):
        """Explicitly discard unsaved in-memory progress; never executes a turn."""
        if self.conversation_id is None:
            raise RuntimeError('No conversation is open.')
        agent = self._new_agent()
        agent.load_conversation(self.conversation_id)
        self.agent = agent
        self._blocked = None
        self._unsaved_answer = None

    def list_conversations(self, limit=20):
        return self.store.list_conversations(limit)

    def history(self):
        if self.agent is None:
            return []
        return [{'role':m.role.value,'content':m.content} for m in self.agent.state.messages]

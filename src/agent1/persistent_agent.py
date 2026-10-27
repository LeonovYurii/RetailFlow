"""Explicit save/restore around the existing synchronous Agent.

No model weights or model instances are serialized. Save after a completed
handle_message call. Restoring does not execute tasks. This is a learning-stage
checkpoint API, not exactly-once crash recovery for external write operations.
"""
from agent1.agent import Agent
from agent1.analysis.action_plan import RequestType


class PersistentAgent(Agent):
    def __init__(self, *args, checkpoint_store, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_store = checkpoint_store

    @staticmethod
    def _validate_checkpoint_state(state, runtime):
        plan = state.pending_action_plan
        if plan is not None:
            if not state.get_missing_arguments():
                raise ValueError('Only pending plans waiting for arguments can be checkpointed.')
            if runtime['fresh_request']:
                raise ValueError('A pending action cannot also be a fresh request.')
            key = ('multi_request_text' if plan.request_type == RequestType.MULTI_ACTION
                   else 'single_request_text')
            if not runtime[key].strip():
                raise ValueError('Pending plan is missing its runner request text.')
            if plan.request_type == RequestType.MULTI_ACTION and any(
                step.depends_on or step.condition for step in plan.steps
            ):
                raise ValueError('Dependent or conditional pending plans are unsupported.')

    def save_conversation(self, conversation_id):
        runtime = {
            'fresh_request': self._fresh_request,
            'single_request_text': self.action_runner._request_text,
            'multi_request_text': self.multi_action_runner.request_text,
        }
        self._validate_checkpoint_state(self.state, runtime)
        self.checkpoint_store.save_checkpoint(conversation_id, self.state, runtime)

    def load_conversation(self, conversation_id):
        """Replace this instance's conversation; validate before changing it.

        Unsaved in-memory history is replaced. Save it under its own ID first if
        needed, or use a fresh PersistentAgent instance as shown in notebook 18.
        """
        state, runtime = self.checkpoint_store.load_checkpoint(conversation_id)
        self._validate_checkpoint_state(state, runtime)
        self.state = state
        # All three objects MUST reference the same restored state.
        self.action_runner.state = state
        self.multi_action_runner.state = state
        self.action_runner._request_text = runtime['single_request_text']
        self.multi_action_runner.request_text = runtime['multi_request_text']
        self._fresh_request = runtime['fresh_request']
        # These are per-call diagnostics; durable tool results remain in messages.
        self.action_runner.last_tool_result = None
        self.multi_action_runner.last_results = []
        self.last_routing_decision = state.pending_routing_decision
        self.last_action_plan = (
            state.pending_action_plan.model_copy(deep=True)
            if state.pending_action_plan is not None else None
        )
        self.last_pending_decision = None
        self.last_pending_error = None
        if self.response_service is not None:
            self.response_service.reset_trace()
        return self

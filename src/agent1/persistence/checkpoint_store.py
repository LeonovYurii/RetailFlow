"""Versioned Agent checkpoints, stored separately from step-1 state snapshots."""
import json
import sqlite3
from contextlib import closing

from agent1.persistence.sqlite_store import SQLiteConversationStore, _encode, _decode


def _validate_runtime(runtime):
    expected = {'fresh_request', 'single_request_text', 'multi_request_text'}
    if not isinstance(runtime, dict) or set(runtime) != expected:
        raise ValueError('Invalid checkpoint runtime fields.')
    if type(runtime['fresh_request']) is not bool:
        raise ValueError('fresh_request must be boolean.')
    for key in ('single_request_text', 'multi_request_text'):
        if not isinstance(runtime[key], str):
            raise ValueError(f'{key} must be a string.')


class SQLiteCheckpointStore(SQLiteConversationStore):
    """Explicit last-save-wins checkpoints; inherited save/load remain state-only.

    save_checkpoint/load_checkpoint preserve state and runtime together in one row.
    Existing conversation_snapshots rows are neither changed nor auto-upgraded.
    """

    def __init__(self, database_path):
        super().__init__(database_path)
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('''
                    CREATE TABLE IF NOT EXISTS agent_checkpoints (
                        conversation_id TEXT PRIMARY KEY,
                        checkpoint_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

    def save_checkpoint(self, conversation_id, state, runtime):
        self._check_id(conversation_id)
        _validate_runtime(runtime)
        payload = json.dumps({'checkpoint_version': 1, 'state': _encode(state),
                              'runtime': runtime}, ensure_ascii=False, allow_nan=False)
        _decode(json.loads(payload)['state'])
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('''
                    INSERT INTO agent_checkpoints (conversation_id, checkpoint_json)
                    VALUES (?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        checkpoint_json = excluded.checkpoint_json,
                        updated_at = CURRENT_TIMESTAMP
                ''', (conversation_id, payload))

    def load_checkpoint(self, conversation_id):
        self._check_id(conversation_id)
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                'SELECT checkpoint_json FROM agent_checkpoints WHERE conversation_id = ?',
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f'No Agent checkpoint for {conversation_id!r}.')
        data = json.loads(row[0])
        if data.get('checkpoint_version') != 1:
            raise ValueError('Unsupported Agent checkpoint version.')
        _validate_runtime(data['runtime'])
        return _decode(data['state']), data['runtime']

"""MongoDB alternative to SQLiteSessionStore, using one document per checkpoint.

PyMongo is imported only by from_uri(); SQLite users do not need it installed.
Requires a running MongoDB deployment. No server or account is created here.
"""
import json
from datetime import datetime, timezone
from agent1.persistence.sqlite_store import SQLiteConversationStore, _encode, _decode
from agent1.persistence.checkpoint_store import _validate_runtime
from agent1.persistence.sqlite_session_store import check_limit, summary


class MongoCheckpointStore:
    def __init__(self, collection):
        # Injection also allows isolated adapter tests without a MongoDB server.
        if not collection.write_concern.acknowledged:
            raise ValueError('Checkpoint writes must be acknowledged (not w=0).')
        self.collection = collection
        self._owned_client = None

    @classmethod
    def from_uri(cls, uri, database='agent1', collection='agent_checkpoints'):
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        try:
            client.admin.command('ping')
            store = cls(client[database][collection])
        except Exception:
            client.close()
            raise
        store._owned_client = client
        return store

    def close(self):
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def save_checkpoint(self, conversation_id, state, runtime):
        SQLiteConversationStore._check_id(conversation_id)
        _validate_runtime(runtime)
        # Normalize identically to SQLite; do not persist arbitrary Python objects.
        data = json.loads(json.dumps({'checkpoint_version':1, 'state':_encode(state),
                                    'runtime':runtime}, ensure_ascii=False, allow_nan=False))
        _decode(data['state'])
        # Key names such as '0.order_id' become explicit entries in MongoDB.
        # This avoids dotted dictionary keys while retaining structured state.
        data['state']['pending_arguments'] = [
            {'name':name, 'value':value}
            for name,value in data['state']['pending_arguments'].items()
        ]
        data['mongo_format_version'] = 1
        now = datetime.now(timezone.utc)
        data['updated_at'] = now
        result = self.collection.update_one(
            {'_id':conversation_id},
            {'$set':data, '$setOnInsert':{'created_at':now}}, upsert=True,
        )
        if not result.acknowledged:
            raise RuntimeError('MongoDB did not acknowledge the checkpoint write.')

    @staticmethod
    def _state(document):
        if document.get('checkpoint_version') != 1 or document.get('mongo_format_version') != 1:
            raise ValueError('Unsupported MongoDB checkpoint version.')
        state = dict(document['state'])
        entries = state['pending_arguments']
        if not isinstance(entries,list):
            raise ValueError('Invalid MongoDB pending arguments.')
        arguments = {}
        for entry in entries:
            name = entry['name']
            if not isinstance(name,str) or name in arguments:
                raise ValueError('Invalid or duplicate pending argument name.')
            arguments[name] = entry['value']
        state['pending_arguments'] = arguments
        return state

    def load_checkpoint(self, conversation_id):
        SQLiteConversationStore._check_id(conversation_id)
        document = self.collection.find_one({'_id':conversation_id})
        if document is None:
            raise KeyError(f'No Agent checkpoint for {conversation_id!r}.')
        state = self._state(document)
        _validate_runtime(document['runtime'])
        return _decode(state), document['runtime']

    def list_conversations(self, limit=20):
        check_limit(limit)
        documents = self.collection.find({}).sort([('updated_at',-1),('_id',1)]).limit(limit)
        return [summary(d['_id'],d['created_at'].isoformat(),d['updated_at'].isoformat(),
                        self._state(d)) for d in documents]

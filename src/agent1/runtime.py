"""Standalone runtime composition. Importing this module never loads a model."""
from dataclasses import dataclass
from pathlib import Path
import json
import os


@dataclass(frozen=True)
class RuntimeConfig:
    classifier: Path
    adapter: Path
    knowledge: Path
    sqlite_path: Path
    backend: str = "sqlite"
    mongo_database: str = "agent1_demo"

    def validate(self):
        if self.backend not in {"sqlite", "mongodb"}:
            raise ValueError("Choose sqlite or mongodb.")
        if not self.classifier.is_file():
            raise ValueError(f"Classifier not found: {self.classifier}. Supply --classifier.")
        if not (self.adapter / "adapter_config.json").is_file():
            raise ValueError(f"Adapter config not found in {self.adapter}. Supply --adapter.")
        if not any((self.adapter / name).is_file() for name in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"Adapter weights not found in {self.adapter}.")
        if not self.knowledge.is_file():
            raise ValueError(f"Knowledge file not found: {self.knowledge}.")
        data = json.loads(self.knowledge.read_text(encoding="utf-8"))
        if not isinstance(data.get("documents"), list):
            raise ValueError("Knowledge file must contain a documents list.")
        if self.backend == "mongodb" and not os.environ.get("MONGODB_URI"):
            raise ValueError("Set MONGODB_URI before selecting MongoDB.")


class Runtime:
    def __init__(self, store, agent_factory):
        self.store = store
        self.agent_factory = agent_factory

    def new_session(self):
        from agent1.session import ChatSession
        return ChatSession(self.store, self.agent_factory)

    def close(self):
        close = getattr(self.store, "close", None)
        if close:
            close()


def load_runtime(config):
    """Load shared models once; each session receives fresh mutable services."""
    config.validate()
    from agent1.classification.ensemble import IntentClassifierEnsemble
    from agent1.generation.response_generator import ResponseGenerator
    from agent1.generation.response_service import ResponseService
    from agent1.analysis.request_analyzer import RequestAnalyzer
    from agent1.routing.intent_router import IntentRouter
    from agent1.routing.intent_mapping import INTENT_TO_ACTION
    from agent1.conversation.context_builder import ContextBuilder
    from agent1.rag.knowledge import KnowledgeStore
    from agent1.persistent_agent import PersistentAgent

    if config.backend == "sqlite":
        from agent1.persistence.sqlite_session_store import SQLiteSessionStore
        store = SQLiteSessionStore(config.sqlite_path)
    else:
        from agent1.persistence.mongo_store import MongoCheckpointStore
        store = MongoCheckpointStore.from_uri(
            os.environ["MONGODB_URI"], database=config.mongo_database
        )
    try:
        ensemble = IntentClassifierEnsemble.load(config.classifier)
        ensemble.semantic_classifier.device = "cpu"
        ensemble.semantic_classifier.embedding_model = None
        knowledge = KnowledgeStore(config.knowledge)
        generator = ResponseGenerator(config.adapter)

        def analyzer_llm(system_prompt, user_prompt):
            with generator.model.disable_adapter():
                return generator.generate([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ], max_new_tokens=1024)

        def make_agent(checkpoint_store):
            return PersistentAgent(
                ensemble_model=ensemble,
                router=IntentRouter(INTENT_TO_ACTION),
                analyzer=RequestAnalyzer(
                    llm=analyzer_llm, knowledge_intent_lookup=knowledge.select
                ),
                context_builder=ContextBuilder(),
                response_service=ResponseService(generator, knowledge),
                checkpoint_store=checkpoint_store,
            )
        return Runtime(store, make_agent)
    except BaseException:
        close = getattr(store, "close", None)
        if close:
            close()
        raise

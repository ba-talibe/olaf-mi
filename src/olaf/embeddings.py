from __future__ import annotations
import sys
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointIdsList, PointStruct, VectorParams,
)
import uuid

# Shared across all EmbeddingService instances in the same process.
# fastembed model loading (~500 MB download + model init) happens once.
_model_cache: dict[str, Any] = {}

_VECTOR_SIZE: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}


# Models fastembed doesn't ship but that publish an ONNX export on Hugging Face: registered
# with fastembed as custom models (mean pooling + normalisation, as sentence-transformers does).
_CUSTOM_ONNX_MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-small": {"dim": 384, "model_file": "onnx/model.onnx"},
    "intfloat/multilingual-e5-base": {"dim": 768, "model_file": "onnx/model.onnx"},
}


def _register_custom_onnx(model_name: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    spec = _CUSTOM_ONNX_MODELS.get(model_name)
    if spec is None or any(m["model"] == model_name for m in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=spec["dim"],
        model_file=spec["model_file"],
    )


# Points are tagged with their kind ("class", "individual", "property"). Points indexed before
# properties were have none: they are classes or individuals, so concepts are "not a property".
_PROPERTIES_ONLY = Filter(must=[FieldCondition(key="kind", match=MatchValue(value="property"))])
_CONCEPTS_ONLY = Filter(must_not=[FieldCondition(key="kind", match=MatchValue(value="property"))])


def entity_text(entity: dict) -> str:
    """Text embedded for an entity: 'label. definition', with '(domain → range)' after a property's label."""
    text = entity["label"]
    if entity.get("kind") == "property" and (entity.get("domain_label") or entity.get("range_label")):
        text += f" ({entity.get('domain_label') or '?'} → {entity.get('range_label') or '?'})"
    return f"{text}. {entity['definition']}" if entity.get("definition") else text


def preload_model(model_name: str) -> None:
    """Load (and download, the first time) the model into the process-wide cache.
    Called at server startup so no client connection has to wait for it."""
    EmbeddingService._load(model_name)


class EmbeddingService:
    def __init__(self, model_name: str, client: QdrantClient, collection: str):
        self.model_name = model_name
        self.client = client
        self.collection = collection
        self._known_collections: set[str] = set()
        # Load the model now, so a missing/unsupported model disables embeddings at
        # connection time (with a warning) instead of failing every concept_create.
        self._model_instance()
        self._ensure_collection()

    def _model_instance(self):
        return self._load(self.model_name)

    @staticmethod
    def _load(model_name: str):
        if model_name not in _model_cache:
            try:
                from fastembed import TextEmbedding
                _register_custom_onnx(model_name)
                _model_cache[model_name] = ("fastembed", TextEmbedding(model_name))
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    _model_cache[model_name] = ("sentence_transformers", SentenceTransformer(model_name))
                except Exception as e:
                    raise RuntimeError(f"Could not load embedding model {model_name}: {e}") from e
        return _model_cache[model_name]

    def _vector_size(self) -> int:
        if self.model_name in _VECTOR_SIZE:
            return _VECTOR_SIZE[self.model_name]
        return len(self._embed("probe"))

    def _ensure_collection(self, name: str | None = None) -> None:
        name = name or self.collection
        if name in self._known_collections:
            return
        if name not in self.collections():
            self.client.create_collection(
                name,
                vectors_config=VectorParams(size=self._vector_size(), distance=Distance.COSINE),
            )
        # Searches filter on the kind (no-op if the index already exists).
        self.client.create_payload_index(name, "kind", PayloadSchemaType.KEYWORD)
        self._known_collections.add(name)

    def collections(self) -> set[str]:
        return {c.name for c in self.client.get_collections().collections}

    def drop_collection(self, name: str) -> None:
        if name in self.collections():
            self.client.delete_collection(name)
        self._known_collections.discard(name)

    def switch_collection(self, collection_name: str) -> None:
        self.collection = collection_name
        self._ensure_collection()

    def _embed(self, text: str, is_query: bool = False) -> list[float]:
        kind, model = self._model_instance()
        # e5 models are trained with these prefixes; neither backend adds them on its own.
        prefix = ("query: " if is_query else "passage: ") if "e5" in self.model_name.lower() else ""
        if kind == "fastembed":
            return list(next(iter(model.embed([f"{prefix}{text}"]))).tolist())
        return model.encode(f"{prefix}{text}").tolist()

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        kind, model = self._model_instance()
        prefix = "passage: " if "e5" in self.model_name.lower() else ""
        texts = [f"{prefix}{t}" for t in texts]
        if kind == "fastembed":
            return [v.tolist() for v in model.embed(texts)]
        return [v.tolist() for v in model.encode(texts)]

    def _concept_point_id(self, uri: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))

    def upsert_concept(
        self,
        uri: str,
        label: str,
        definition: str,
        source_chunk_id: str | None = None,
        kind: str | None = None,
    ) -> None:
        text = f"{label}. {definition}" if definition else label
        vector = self._embed(text, is_query=False)
        payload: dict = {"uri": uri, "label": label, "definition": definition}
        if source_chunk_id is not None:
            payload["source_chunk_id"] = source_chunk_id
        if kind is not None:
            payload["kind"] = kind
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=self._concept_point_id(uri), vector=vector, payload=payload)],
        )

    def upsert_entity(self, entity: dict) -> None:
        """Index one entry of OntologyStore.ontology_entities in the active collection."""
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=self._concept_point_id(entity["uri"]),
                                vector=self._embed(entity_text(entity)), payload=entity)],
        )

    def index_collection(self, name: str, entities: list[dict], batch_size: int = 64) -> int:
        """Make collection `name` hold exactly these entities (entries of OntologyStore.seed_entities /
        ontology_entities, kept as payload). Used to index a seed, or rebuild an ontology's index,
        as a whole. Points are replaced in place and stale ones removed at the end, so a failure
        midway leaves the previous index usable; the collection is only recreated when its vector
        size no longer matches the model."""
        if name in self.collections():
            vectors = self.client.get_collection(name).config.params.vectors
            if getattr(vectors, "size", None) != self._vector_size():
                self.drop_collection(name)
        self._ensure_collection(name)
        for start in range(0, len(entities), batch_size):
            batch = entities[start:start + batch_size]
            texts = [entity_text(e) for e in batch]
            self.client.upsert(
                collection_name=name,
                points=[
                    PointStruct(id=self._concept_point_id(e["uri"]), vector=vector, payload=e)
                    for e, vector in zip(batch, self._embed_many(texts))
                ],
            )
        keep = {self._concept_point_id(e["uri"]) for e in entities}
        stale, offset = [], None
        while True:
            points, offset = self.client.scroll(name, limit=1000, offset=offset, with_payload=False, with_vectors=False)
            stale += [p.id for p in points if str(p.id) not in keep]
            if offset is None:
                break
        for start in range(0, len(stale), 1000):
            self.client.delete(name, points_selector=PointIdsList(points=stale[start:start + 1000]))
        return len(entities)

    def search_concepts(
        self, query: str, top_k: int = 10, seed_collections: dict[str, str] | None = None,
        properties: bool = False,
    ) -> list[dict]:
        """Search the active ontology's collection for classes and individuals, or for properties.
        With seed_collections ({seed_id: collection}), also search those and merge by score; each
        result then carries its `seed` id (None for the ontology). A term found both in the
        ontology (reused) and in a seed, or in several seeds, is listed once — the ontology first."""
        vector = self._embed(query, is_query=True)
        query_filter = _PROPERTIES_ONLY if properties else _CONCEPTS_ONLY
        results = self._search(self.collection, vector, top_k, query_filter)
        if seed_collections is None:
            return results
        for r in results:
            r["seed"] = None
        for seed_id, collection in seed_collections.items():
            try:
                results += [{**r, "seed": seed_id} for r in self._search(collection, vector, top_k, query_filter)]
            except Exception as e:
                print(f"Warning: seed collection {collection} not searched ({e})", file=sys.stderr)
        unique: dict[str, dict] = {}
        for r in results:
            unique.setdefault(r["uri"], r)
        return sorted(unique.values(), key=lambda r: r["score"], reverse=True)[:top_k]

    def _search(self, collection: str, vector: list[float], top_k: int, query_filter: Filter) -> list[dict]:
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self.client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        results = []
        for h in hits:
            payload = h.payload or {}
            result = {
                "uri": payload.get("uri", ""),
                "label": payload.get("label", ""),
                "definition": payload.get("definition", ""),
                "source_chunk_id": payload.get("source_chunk_id"),
                "score": round(h.score, 4) if hasattr(h, "score") and h.score is not None else 0.0,
                "match_type": "semantic",
            }
            if payload.get("kind") == "property":
                result.update({k: payload.get(k) for k in ("property_type", "domain", "range")})
            results.append(result)
        return results

    def delete_concept(self, uri: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=[self._concept_point_id(uri)]),
        )

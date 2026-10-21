from __future__ import annotations
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FieldMapping:
    text: str = "text"
    doc_id: str = "doc_id"
    chunk_index: str = "chunk_index"


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "chunks"
    # Prefix of the semantic index collections: {index_prefix}_{ontology_id}, {index_prefix}_seed_{seed_id}
    index_prefix: str = "olaf_index"
    field_mapping: FieldMapping = field(default_factory=FieldMapping)


@dataclass
class OxigraphConfig:
    url: str = "http://localhost:7878"


@dataclass
class OntologyConfig:
    base_uri: str = "http://olaf.local/ontology#"
    name: str = "Ontology"
    ontology_id: str = "main"


@dataclass
class EmbeddingConfig:
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enabled: bool = True


@dataclass
class ReasonerConfig:
    enabled: bool = True
    java: str = "java"
    memory_mb: int = 2048
    timeout_seconds: int = 120


@dataclass
class Config:
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    oxigraph: OxigraphConfig = field(default_factory=OxigraphConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            for candidate in ("config.toml", "olaf.toml"):
                p = Path(candidate)
                if p.exists():
                    path = p
                    break

        cfg = cls()
        if path is None or not Path(path).exists():
            cfg.qdrant.api_key = os.environ.get("QDRANT_API_KEY", cfg.qdrant.api_key)
            return cfg

        with open(path, "rb") as f:
            data = tomllib.load(f)

        if q := data.get("qdrant"):
            cfg.qdrant.url = q.get("url", cfg.qdrant.url)
            cfg.qdrant.api_key = q.get("api_key", cfg.qdrant.api_key)
            cfg.qdrant.collection = q.get("collection", cfg.qdrant.collection)
            if "concepts_collection" in q and "index_prefix" not in q:
                print("Warning: [qdrant] concepts_collection is deprecated, rename it to index_prefix "
                      "(the index holds classes, individuals and properties).", file=sys.stderr)
            cfg.qdrant.index_prefix = q.get("index_prefix", q.get("concepts_collection", cfg.qdrant.index_prefix))
            if fm := q.get("field_mapping"):
                cfg.qdrant.field_mapping.text = fm.get("text", cfg.qdrant.field_mapping.text)
                cfg.qdrant.field_mapping.doc_id = fm.get("doc_id", cfg.qdrant.field_mapping.doc_id)
                cfg.qdrant.field_mapping.chunk_index = fm.get("chunk_index", cfg.qdrant.field_mapping.chunk_index)

        if o := data.get("oxigraph"):
            cfg.oxigraph.url = o.get("url", cfg.oxigraph.url)

        if n := data.get("ontology"):
            cfg.ontology.base_uri = n.get("base_uri", cfg.ontology.base_uri)
            cfg.ontology.name = n.get("name", cfg.ontology.name)
            cfg.ontology.ontology_id = n.get("ontology_id", cfg.ontology.ontology_id)

        if e := data.get("embedding"):
            cfg.embedding.model = e.get("model", cfg.embedding.model)
            cfg.embedding.enabled = e.get("enabled", cfg.embedding.enabled)

        if r := data.get("reasoner"):
            cfg.reasoner.enabled = r.get("enabled", cfg.reasoner.enabled)
            cfg.reasoner.java = r.get("java", cfg.reasoner.java)
            cfg.reasoner.memory_mb = r.get("memory_mb", cfg.reasoner.memory_mb)
            cfg.reasoner.timeout_seconds = r.get("timeout_seconds", cfg.reasoner.timeout_seconds)

        cfg.qdrant.api_key = os.environ.get("QDRANT_API_KEY", cfg.qdrant.api_key)

        return cfg

from __future__ import annotations
import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any

import jsonschema
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from qdrant_client import QdrantClient

from .chunks import ChunkStore
from .config import Config, FieldMapping
from .embeddings import EmbeddingService, preload_model
from .ontology import RDF_FORMATS, OntologyStore, seed_graph
from .reasoner import Reasoner


def _text(content: Any) -> list[types.TextContent]:
    body = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
    return [types.TextContent(type="text", text=body)]


def _err(msg: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=f"Error: {msg}")]


async def _preload_embeddings(config: Config) -> None:
    if not config.embedding.enabled:
        return
    print(f"Loading embedding model {config.embedding.model}…", file=sys.stderr)
    try:
        await asyncio.to_thread(preload_model, config.embedding.model)
    except Exception as e:
        print(f"Warning: embeddings disabled ({e})", file=sys.stderr)


def _coerce_arguments(arguments: dict, schema: dict | None) -> dict:
    """LLMs sometimes send scalars as strings ("False", "10"): convert them to the type the
    tool's input schema declares — otherwise "False" would be taken as true."""
    properties = (schema or {}).get("properties", {})
    coerced = dict(arguments)
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        kind, text = properties.get(key, {}).get("type"), value.strip().lower()
        if kind == "boolean" and text in ("true", "false", "1", "0", "yes", "no"):
            coerced[key] = text in ("true", "1", "yes")
        elif kind == "integer" and re.fullmatch(r"-?\d+", text):
            coerced[key] = int(text)
    return coerced


def _ontology_collection(config: Config, ontology_id: str) -> str:
    return f"{config.qdrant.index_prefix}_{ontology_id}"


def _seed_collection_prefix(config: Config) -> str:
    return f"{config.qdrant.index_prefix}_seed_"


async def _index_seed(onto: OntologyStore, embed: EmbeddingService, config: Config, seed_id: str) -> int:
    """(Re)build the seed's Qdrant collection from its classes, individuals and properties in Oxigraph."""
    entities = [{**e, "seed_id": seed_id} for e in await onto.seed_entities(seed_id)]
    collection = _seed_collection_prefix(config) + seed_id
    return await asyncio.to_thread(embed.index_collection, collection, entities)


async def _make_shared_infra(config: Config) -> tuple[OntologyStore, QdrantClient, ChunkStore]:
    """Create and bootstrap infrastructure objects shared across all SSE connections."""
    await _preload_embeddings(config)
    onto = OntologyStore(url=config.oxigraph.url)
    await onto.bootstrap(config.ontology.ontology_id, config.ontology.base_uri, config.ontology.name)
    qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
    chunks = ChunkStore(config.qdrant.url, config.qdrant.collection, config.qdrant.field_mapping, config.qdrant.api_key)
    return onto, qdrant, chunks


def create_server(
    config: Config,
    onto: OntologyStore,
    qdrant: QdrantClient,
    chunks: ChunkStore,
) -> Server:
    """
    Create an MCP server instance.

    For SSE deployments, pass pre-created shared onto/qdrant/chunks so a single
    OntologyStore (stateless) and QdrantClient serve all connections.
    For stdio, create them first with _make_shared_infra and pass them in.
    """
    # Per-session state: the active ontology context, chunk collection and seed selection
    # (seed_ids None = every seed in the store).
    # Using a dict so the call_tool closure can mutate it without `nonlocal`.
    session: dict[str, Any] = {
        "ontology_id": config.ontology.ontology_id,
        "base_uri": config.ontology.base_uri,
        "chunks": chunks,
        "seed_ids": None,
    }

    embed: EmbeddingService | None = None
    if config.embedding.enabled:
        try:
            embed = EmbeddingService(config.embedding.model, qdrant, _ontology_collection(config, config.ontology.ontology_id))
        except Exception as e:
            print(f"Warning: embeddings disabled ({e})", file=sys.stderr)

    reasoner: Reasoner | None = None
    if config.reasoner.enabled:
        reasoner = Reasoner(config.reasoner.java, config.reasoner.memory_mb, config.reasoner.timeout_seconds)

    def _index_failed(e: Exception) -> str:
        print(f"Warning: semantic index not updated ({e})", file=sys.stderr)
        return f"Ontology updated, but the semantic index (concept/property_semantic_search) was not: {e}"

    def _index(action) -> str | None:
        """Update the semantic index. The ontology change is already written at this point,
        so a failure is reported as a warning instead of turning the call into an error."""
        try:
            action()
            return None
        except Exception as e:
            return _index_failed(e)

    async def _reindex(uri: str) -> str | None:
        """Re-index an entity (class, individual or property) from its current state in the ontology."""
        try:
            for entry in await onto.ontology_entities(session["ontology_id"], [uri]):
                embed.upsert_entity(entry)
            return None
        except Exception as e:
            return _index_failed(e)

    server = Server("olaf")
    schemas: dict[str, dict] = {}  # tool name → input schema, filled on the first call

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="chunk_collection_switch",
                description=(
                    "Switch the Qdrant collection the chunk tools read from, for this session only "
                    "(default: the server's configured collection). Optionally override the payload "
                    "field names if this collection uses different ones. Returns chunk counts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {"type": "string", "description": "Qdrant collection name"},
                        "field_mapping": {
                            "type": "object",
                            "description": "Payload field names (default: the server's [qdrant.field_mapping])",
                            "properties": {
                                "text": {"type": "string"},
                                "doc_id": {"type": "string"},
                                "chunk_index": {"type": "string"},
                            },
                        },
                    },
                    "required": ["collection"],
                },
            ),
            types.Tool(
                name="chunk_list",
                description=(
                    "List chunks from the Qdrant collection. "
                    "Filter by doc_id and/or status. Returns id, doc_id, chunk_index, "
                    "text_preview (200 chars), and processing status."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Filter to one document"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "processed", "all"],
                            "description": "Filter by olaf processing status (default: all)",
                        },
                        "limit": {"type": "integer", "description": "Max results (default: 20)"},
                        "offset": {"type": "integer", "description": "Pagination offset"},
                    },
                },
            ),
            types.Tool(
                name="chunk_read",
                description="Read the full text of a chunk by its Qdrant point ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Qdrant point ID (integer or UUID string)"},
                    },
                    "required": ["chunk_id"],
                },
            ),
            types.Tool(
                name="chunk_read_batch",
                description=(
                    "Read the full text of multiple chunks in a single call. "
                    "More efficient than calling chunk_read repeatedly. "
                    "Returns a list of {id, doc_id, chunk_index, text, status} objects."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Qdrant point IDs to retrieve",
                        },
                    },
                    "required": ["chunk_ids"],
                },
            ),
            types.Tool(
                name="chunk_mark_processed",
                description="Mark a chunk as processed after extracting ontology elements from it.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                    },
                    "required": ["chunk_id"],
                },
            ),
            types.Tool(
                name="concept_list",
                description=(
                    "List all OWL classes in the active ontology. "
                    "Returns uri, label, definition, and parent_uri for each class. "
                    "Use root_only=true to get only top-level classes (no parent). "
                    "Useful for surveying the current ontology state before adding concepts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "root_only": {
                            "type": "boolean",
                            "description": "Return only root classes (no rdfs:subClassOf parent). Default: false.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 100)",
                        },
                    },
                },
            ),
            types.Tool(
                name="concept_create",
                description=(
                    "Create an OWL class in the ontology. "
                    "Always call concept_search and concept_semantic_search first to check for duplicates. "
                    "Returns the URI and whether a new class was created (created=false if URI already exists). "
                    "If the concept already exists and source_chunk_id is given, it is added to the concept's "
                    "existing sources (no duplicate is created) — always pass source_chunk_id even when you "
                    "expect created=false, so provenance stays complete across chunks."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Human-readable class name, e.g. 'Motor Vehicle'"},
                        "definition": {"type": "string", "description": "Natural language definition"},
                        "parent_uri": {"type": "string", "description": "Parent class URI for rdfs:subClassOf. A seed term must be declared with seed_reuse first."},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative labels (rdfs:altLabel)",
                        },
                        "language": {"type": "string", "description": "BCP-47 language tag (default: en)"},
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the chunk this concept was extracted from"},
                    },
                    "required": ["label", "definition"],
                },
            ),
            types.Tool(
                name="concept_get",
                description=(
                    "Get all triples for a concept URI: type, label, definition, aliases, subClassOf, restrictions. "
                    "Also returns source_chunk_ids: every chunk this concept was extracted from, and, for a class "
                    "with OWL restrictions, `restrictions` in restriction_add / restriction_delete terms."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="concept_search",
                description=(
                    "Search for existing concepts by label substring (SPARQL). "
                    "Call this before concept_create to avoid duplicates. "
                    "For broader similarity matching, also call concept_semantic_search. "
                    "Each result includes source_chunk_ids: every chunk that concept was extracted from."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also search the selected seed ontologies' (seed_select) classes; each result then has a "
                                           "'seed' field (seed id, null for the active ontology). Default: false",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="concept_semantic_search",
                description=(
                    "Search for existing concepts by semantic similarity (vector search in Qdrant). "
                    "Finds concepts that are similar in meaning even with different wording. "
                    "Call alongside concept_search before concept_create to catch near-duplicates. "
                    "Scores are only comparable to each other (with some models, even unrelated "
                    "concepts score high): judge a match by its label and definition, not by a threshold."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also search the selected seed ontologies' (seed_select) classes and individuals; each result then has a "
                                           "'seed' field (seed id, null for the active ontology). Default: false",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="concept_update",
                description="Update a concept's label, definition, or aliases.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "label": {"type": "string"},
                        "definition": {"type": "string"},
                        "aliases_add": {"type": "array", "items": {"type": "string"}},
                        "aliases_remove": {"type": "array", "items": {"type": "string"}},
                        "language": {"type": "string"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="individual_create",
                description=(
                    "Create an owl:NamedIndividual — a specific named entity that is an instance of a class. "
                    "Use for unique, identifiable real-world entities (e.g. 'GDPR', 'Paris Agreement'). "
                    "Always call concept_search first to confirm the class exists. "
                    "Only one class_uri per call — to also type the individual as an instance of a "
                    "second class (multiple inheritance, e.g. both 'Developing Country' and 'Small Island "
                    "Developing State'), call relation_add afterwards with property_uri = "
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type and object_value = the extra class URI. "
                    "Returns {uri, created} — created=false if the individual already exists. "
                    "If it already exists and source_chunk_id is given, it is added to the individual's "
                    "existing sources — always pass source_chunk_id even when you expect created=false."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Human-readable name, e.g. 'Paris Agreement'"},
                        "class_uri": {"type": "string", "description": "URI of the owl:Class this individual is an instance of. A seed term must be declared with seed_reuse first."},
                        "definition": {"type": "string", "description": "Description of this specific individual"},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative labels (rdfs:altLabel)",
                        },
                        "language": {"type": "string", "description": "BCP-47 language tag (default: en)"},
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the source chunk"},
                    },
                    "required": ["label", "class_uri"],
                },
            ),
            types.Tool(
                name="concept_merge",
                description=(
                    "Merge two concepts: all triples from merge_uri are copied to keep_uri, "
                    "all references to merge_uri are re-pointed to keep_uri, then merge_uri is deleted. "
                    "merge_uri must be the ontology's own. keep_uri may be a reused seed term, to replace a "
                    "local duplicate of it: then only the references and source chunks are carried over, "
                    "since the seed term keeps its source definition."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keep_uri": {"type": "string", "description": "URI to keep"},
                        "merge_uri": {"type": "string", "description": "URI to absorb (will be deleted)"},
                    },
                    "required": ["keep_uri", "merge_uri"],
                },
            ),
            types.Tool(
                name="property_create",
                description=(
                    "Create an OWL property (ObjectProperty links two classes; DatatypeProperty links a class "
                    "to a literal). If it already exists and source_chunk_id is given, it is added to the "
                    "property's existing sources — always pass source_chunk_id even when you expect created=false."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "e.g. 'has engine'"},
                        "type": {
                            "type": "string",
                            "enum": ["object", "datatype"],
                        },
                        "domain_uri": {"type": "string", "description": "rdfs:domain class URI. A seed term must be declared with seed_reuse first."},
                        "range_uri": {
                            "type": "string",
                            "description": "rdfs:range class URI (object) or XSD datatype URI (datatype)",
                        },
                        "parent_uri": {"type": "string", "description": "rdfs:subPropertyOf URI"},
                        "language": {"type": "string"},
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the chunk this property was extracted from"},
                    },
                    "required": ["label", "type"],
                },
            ),
            types.Tool(
                name="property_update",
                description=(
                    "Update a property's domain, range, or parent property after creation. "
                    "Pass an empty string to clear a field, or omit it to leave it unchanged."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "domain_uri": {"type": "string", "description": "rdfs:domain class URI. A seed term must be declared with seed_reuse first."},
                        "range_uri": {"type": "string", "description": "rdfs:range class/datatype URI"},
                        "parent_uri": {"type": "string", "description": "rdfs:subPropertyOf URI"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="property_get",
                description="Get all details of a property by URI, including source_chunk_ids (every chunk it was extracted from).",
                inputSchema={
                    "type": "object",
                    "properties": {"uri": {"type": "string"}},
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="property_search",
                description="Search for properties by label substring. Each result includes source_chunk_ids.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also search the selected seed ontologies' (seed_select) properties; each result then has a "
                                           "'seed' field (seed id, null for the active ontology). Default: false",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="property_semantic_search",
                description=(
                    "Search for existing properties by semantic similarity (vector search in Qdrant on "
                    "'label (domain → range)'). Finds properties similar in meaning even with different "
                    "wording, e.g. 'is equipped with' for 'has engine'. Call alongside property_search "
                    "before property_create to catch near-duplicates, and with include_seeds=true to reuse "
                    "a seed's property instead of creating one. Scores are only comparable to each other: "
                    "judge a match by its label, domain and range, not by a threshold."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "e.g. 'has engine' or 'vehicle has engine'"},
                        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also search the selected seed ontologies' (seed_select) properties; each result then has a "
                                           "'seed' field (seed id, null for the active ontology). Default: false",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="relation_add",
                description=(
                    "Add a triple (subject, property, object) to the ontology. "
                    "Use full URIs for all arguments — never type a URI from memory: get subject/object "
                    "URIs from concept_create/individual_create/concept_search/concept_get output, and get "
                    "property_uri from property_create/property_search output, seed_reuse, or the standard "
                    "vocabularies (rdf, rdfs, owl, skos, xsd). The object of a linking property (owl:sameAs, "
                    "rdfs:seeAlso, rdfs:isDefinedBy, skos:exactMatch/closeMatch/broadMatch/narrowMatch/"
                    "relatedMatch) may be any URI, e.g. a Wikidata entity. "
                    "Fails with an error if subject_uri, property_uri, or object_value (as a URI) is unknown: "
                    "an entity of this ontology that doesn't exist yet (create it first), a seed term not yet "
                    "declared (seed_reuse first) or any other external URI. A seed class or property cannot "
                    "be the subject: its definition belongs to its source ontology — link from the local "
                    "entity instead (<local> rdfs:subClassOf <seed>). "
                    "Common property URIs: rdfs:subClassOf = http://www.w3.org/2000/01/rdf-schema#subClassOf, "
                    "rdf:type = http://www.w3.org/1999/02/22-rdf-syntax-ns#type. "
                    "Pass source_chunk_id to record which chunk this relation was extracted from — if the "
                    "same triple is added again from a different chunk, the new chunk id is added to its "
                    "existing sources instead of being lost (adding the same triple is otherwise a no-op)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_value": {"type": "string", "description": "Target URI or literal value"},
                        "is_literal": {"type": "boolean", "description": "True if object_value is a literal (default: false)"},
                        "datatype": {
                            "type": "string",
                            "description": "XSD datatype URI when is_literal=true, e.g. http://www.w3.org/2001/XMLSchema#integer",
                        },
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the chunk this relation was extracted from"},
                    },
                    "required": ["subject_uri", "property_uri", "object_value"],
                },
            ),
            types.Tool(
                name="relation_sources",
                description="Get the chunk IDs a specific (subject, property, object) triple was extracted from.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_value": {"type": "string", "description": "Target URI or literal value"},
                        "is_literal": {"type": "boolean", "description": "True if object_value is a literal (default: false)"},
                        "datatype": {
                            "type": "string",
                            "description": "XSD datatype URI when is_literal=true, e.g. http://www.w3.org/2001/XMLSchema#integer",
                        },
                    },
                    "required": ["subject_uri", "property_uri", "object_value"],
                },
            ),
            types.Tool(
                name="relation_delete",
                description=(
                    "Remove a triple (subject, property, object) from the ontology. "
                    "Use to correct a wrong relation, e.g. before re-adding it with property_update-style edits."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_value": {"type": "string", "description": "Target URI or literal value"},
                        "is_literal": {"type": "boolean", "description": "True if object_value is a literal (default: false)"},
                        "datatype": {
                            "type": "string",
                            "description": "XSD datatype URI when is_literal=true, e.g. http://www.w3.org/2001/XMLSchema#integer",
                        },
                    },
                    "required": ["subject_uri", "property_uri", "object_value"],
                },
            ),
            types.Tool(
                name="relation_search",
                description="Find triples matching a pattern. All parameters are optional — omit to get all triples.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "object_uri": {"type": "string"},
                        "limit": {"type": "integer", "description": "Max results (default: 20)"},
                    },
                },
            ),
            types.Tool(
                name="restriction_add",
                description=(
                    "Add an OWL restriction to a class via a blank node. "
                    "Examples: 'Car must have some Engine' → some, "
                    "'Person must have exactly 1 birthDate' → exactly+cardinality=1. "
                    "Creates: class rdfs:subClassOf [ a owl:Restriction; owl:onProperty prop; owl:someValuesFrom value ]."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "class_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "restriction_type": {
                            "type": "string",
                            "enum": ["some", "all", "has_value", "exactly", "min", "max"],
                            "description": "some=someValuesFrom, all=allValuesFrom, has_value=hasValue, exactly/min/max=cardinality",
                        },
                        "value": {
                            "type": "string",
                            "description": "Target class URI, or literal string for has_value with is_literal_value=true. Leave empty for unqualified cardinality.",
                        },
                        "cardinality": {"type": "integer", "description": "Required for exactly/min/max"},
                        "is_literal_value": {"type": "boolean", "description": "True if value is a literal (for has_value)"},
                    },
                    "required": ["class_uri", "property_uri", "restriction_type", "value"],
                },
            ),
            types.Tool(
                name="restriction_delete",
                description=(
                    "Delete OWL restrictions of a class on a property (the blank-node restrictions relation_delete "
                    "cannot reach): all of them, or only those of a given restriction_type and/or value — the "
                    "terms concept_get lists under `restrictions`. Returns how many were deleted."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "class_uri": {"type": "string"},
                        "property_uri": {"type": "string"},
                        "restriction_type": {
                            "type": "string",
                            "enum": ["some", "all", "has_value", "exactly", "min", "max"],
                            "description": "Only restrictions of this type (default: any)",
                        },
                        "value": {"type": "string", "description": "Only restrictions on this class URI / literal (default: any)"},
                        "is_literal_value": {"type": "boolean", "description": "True if value is a literal (has_value)"},
                    },
                    "required": ["class_uri", "property_uri"],
                },
            ),
            types.Tool(
                name="entity_delete",
                description=(
                    "Delete a class, individual or property of the ontology entirely: every triple it appears in "
                    "(as subject, property or object), their provenance, its restrictions and the restrictions "
                    "pointing to it, and its entry in the semantic index. For a reused seed term, removes its "
                    "declaration and uses from the ontology (the seed is unchanged). For a wrong or spurious entity only — "
                    "to fix one relation use relation_delete, for a duplicate use concept_merge. Irreversible."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"uri": {"type": "string", "description": "URI of the entity to delete"}},
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="disjoint_add",
                description=(
                    "Declare classes pairwise disjoint (owl:disjointWith): no individual can belong to two "
                    "of them. Use for sibling classes that cannot overlap, e.g. Person / Organisation / Document. "
                    "Without disjointness the reasoner (ontology_check) can almost never detect a contradiction. "
                    "Rejected if two of the classes are in a subclass relation. "
                    "Each pair is stored once as <a> owl:disjointWith <b>; undo it with relation_delete."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "class_uris": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "URIs of the classes to declare mutually disjoint (2 or more)",
                        },
                    },
                    "required": ["class_uris"],
                },
            ),
            types.Tool(
                name="seed_term_get",
                description=(
                    "Look at a seed term as its seed defines it, before reusing it: label, definition, "
                    "parents, subclasses (subproperties for a property), an individual's classes, a "
                    "property's domain and range. Use it to check that the term means what you think — "
                    "a label alone (e.g. '6T') is not enough."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"uri": {"type": "string", "description": "Seed term URI"}},
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="seed_reuse",
                description=(
                    "Reuse a seed term (class, individual or property) in the ontology under its own seed URI, "
                    "instead of creating an equivalent local one. Copies its minimal declaration (OWL type, "
                    "labels, definitions, parents, domain/range for a property, rdfs:isDefinedBy its source "
                    "ontology) and indexes it. Call it when a seed term has exactly the meaning you need, "
                    "then use its URI like any other (parent_uri, class_uri, domain/range, relation_add). "
                    "For a narrower meaning, reuse the seed term and create a local subclass of it. A reused "
                    "term cannot be modified (concept_update, property_update, axioms with it as subject). "
                    "Idempotent: created=false if already reused; pass source_chunk_id anyway."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string", "description": "Seed term URI, from concept_search / "
                                                                 "property_search / *_semantic_search with include_seeds=true"},
                        "source_chunk_id": {"type": "string", "description": "Qdrant point ID of the chunk that motivates the reuse"},
                    },
                    "required": ["uri"],
                },
            ),
            types.Tool(
                name="seed_select",
                description=(
                    "Select the seeds this session works with, by id (see seed_list): include_seeds "
                    "searches, seed_reuse, and ontology_check / ontology_infer / ontology_export with "
                    "include_seeds then only see these. An empty list selects none. "
                    "Default (never called): every seed in the store."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "seed_ids": {"type": "array", "items": {"type": "string"},
                                     "description": "Ids of the seeds to use, e.g. ['fibo-be', 'sustaingraph']"},
                    },
                    "required": ["seed_ids"],
                },
            ),
            types.Tool(
                name="seed_list",
                description="List all seed ontologies loaded in the store (shared across all ontologies), "
                            "with whether each is selected for this session (seed_select).",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="seed_load",
                description=(
                    "Load a seed ontology into the global seed store (graph urn:olaf:seed:{seed_id}) and index "
                    "its classes, individuals and properties for concept_semantic_search / property_semantic_search. "
                    "Pass either 'url' (public HTTP/HTTPS URI, at most 200 MB) or 'content' (raw RDF string). "
                    "Passing the seed_id of an existing seed replaces it; without seed_id, the id derived "
                    "from the URL must be new. "
                    "Seeds are shared across all ontologies."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "HTTP/HTTPS URI of an RDF ontology to fetch"},
                        "content": {"type": "string", "description": "Raw RDF content to load directly"},
                        "seed_id": {
                            "type": "string",
                            "description": "Short identifier, e.g. 'foaf' (letters, digits, '_', '-', '.'). "
                                           "Default: derived from the URL (http://xmlns.com/foaf/0.1/ → 'foaf')",
                        },
                        "format": {
                            "type": "string",
                            "enum": list(RDF_FORMATS),
                            "description": "RDF serialization (default: guessed from the response's "
                                           "Content-Type, the URL's extension or the content; else Turtle)",
                        },
                    },
                },
            ),
            types.Tool(
                name="ontology_list",
                description=(
                    "List all ontologies stored in Oxigraph. "
                    "Returns id, name, base_uri, class count, and whether it is the active ontology."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="ontology_create",
                description=(
                    "Create a new empty ontology. "
                    "Use ontology_switch afterwards to start working with it."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ontology_id": {"type": "string", "description": "Short identifier, e.g. 'legal', 'medical-v2' (letters, digits, '_', '-', '.'; not starting with 'seed_')"},
                        "name": {"type": "string", "description": "Human-readable name"},
                        "base_uri": {"type": "string", "description": "Namespace URI (defaults to configured base_uri)"},
                    },
                    "required": ["ontology_id", "name"],
                },
            ),
            types.Tool(
                name="ontology_switch",
                description=(
                    "Switch the active ontology for this session. "
                    "All subsequent tool calls will operate on the selected ontology."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ontology_id": {"type": "string", "description": "Identifier of the ontology to activate"},
                    },
                    "required": ["ontology_id"],
                },
            ),
            types.Tool(
                name="ontology_summary",
                description=(
                    "Get a summary of the current ontology: class/property/individual/restriction counts, "
                    "materialized inferences (inferred_triples), the outcome of the last ontology_check since the "
                    "server started (last_check: time, consistent, issues — later changes are not reflected), "
                    "root classes, chunk processing progress."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="ontology_export",
                description="Export the ontology as a Turtle string.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Also include seed graphs (default: false)",
                        },
                    },
                },
            ),
            types.Tool(
                name="sparql_query",
                description=(
                    "Run a read-only SPARQL 1.1 query (SELECT, ASK, CONSTRUCT or DESCRIBE) against the "
                    "triplestore. INSERT/DELETE/LOAD/CLEAR and other updates are rejected. "
                    "There is no implicit default graph: wrap patterns in GRAPH <urn:olaf:{ontology_id}> "
                    "for an ontology (e.g. urn:olaf:main), GRAPH ?g with FILTER(STRSTARTS(STR(?g), "
                    "'urn:olaf:seed:')) for seeds. Data model: classes are owl:Class, individuals "
                    "owl:NamedIndividual, properties owl:ObjectProperty/owl:DatatypeProperty; labels are "
                    "rdfs:label (language-tagged), aliases rdfs:altLabel, definitions skos:definition; "
                    "provenance is <urn:olaf:extractedFrom> <urn:olaf:chunk:{chunk_id}> on entities and on "
                    "rdf:Statement reification nodes for relations; triples inferred by the reasoner "
                    "(ontology_infer) have no source chunk, and their reification node carries "
                    "<urn:olaf:inferredBy> instead. "
                    "SELECT returns {variables, rows, row_count, truncated}; ASK returns {boolean}; "
                    "CONSTRUCT/DESCRIBE return Turtle. Syntax errors are returned verbatim — fix and retry."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Full SPARQL query, including PREFIX declarations"},
                        "limit": {"type": "integer", "description": "Max SELECT rows returned (default: 100, max: 1000)"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="ontology_orphans",
                description=(
                    "Find classes and individuals with no relation to the rest of the ontology "
                    "(nothing beyond their own type/label/definition triples). "
                    "Call before finishing a session and connect or justify every isolated entity."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max results (default: 100)"},
                    },
                },
            ),
            types.Tool(
                name="ontology_check",
                description=(
                    "Check the ontology for logical errors. Runs (1) the Pellet OWL 2 DL reasoner: global "
                    "consistency and unsatisfiable classes (classes that can have no instance), each with its "
                    "explanation, the minimal set of axioms causing it; and (2) closed-world SPARQL checks the "
                    "reasoner does not report: subclass cycles, untyped individuals, relations whose subject/object "
                    "does not match the property's domain/range, object/datatype property misuse, and "
                    "unknown_terms — RDF/RDFS/OWL/SKOS terms that do not exist (e.g. rdfs:subClassof), with "
                    "the closest real term as `suggestion`. "
                    "`entities` maps the local names used in explanations to their URIs. "
                    "Fix each problem by removing or correcting at least one axiom of its explanation "
                    "(relation_delete, property_update, concept_merge…), then check again."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Reason over the seed graphs too (default: false)",
                        },
                        "max_explanations": {
                            "type": "integer",
                            "description": "Explanations per problem (default: 1, max: 5)",
                        },
                        "limit": {"type": "integer", "description": "Max results per SPARQL check (default: 50)"},
                    },
                },
            ),
            types.Tool(
                name="ontology_infer",
                description=(
                    "Infer what the ontology entails with the Pellet reasoner: subclass relations, class "
                    "membership of individuals, equivalent classes, sub-properties, property assertions and "
                    "same-individual links that are not asserted. The ontology must be consistent "
                    "(ontology_check). action=preview (default) lists the new inferences, each with why it holds: "
                    "`taxonomic` ones follow from the asserted class hierarchy alone (their chain of axioms is "
                    "given), the others are explained by the reasoner (up to max_explained). An absurd "
                    "inference reveals a wrong axiom in its explanation — fix that axiom, not the inference. "
                    "Each inference says whether it is already `materialized`. "
                    "action=materialize writes the inferences into the ontology, replacing those written before; "
                    "each one is marked on its rdf:Statement node with <urn:olaf:inferredBy>, and ontology_check "
                    "ignores them. action=clear removes them. concept_merge also removes them."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["preview", "materialize", "clear"],
                            "description": "preview (default), materialize or clear",
                        },
                        "include_seeds": {
                            "type": "boolean",
                            "description": "Reason over the seed graphs too, e.g. to infer that a class is a "
                                           "subclass of a seed class (default: false). Only inferences about "
                                           "the ontology's own entities are kept.",
                        },
                        "max_explained": {
                            "type": "integer",
                            "description": "preview: non-taxonomic inferences to explain with the reasoner "
                                           "(default: 20, max: 100 — each takes a few seconds)",
                        },
                    },
                },
            ),
        ]

    # Input is validated here, after _coerce_arguments, rather than by the MCP library before it.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        try:
            if not schemas:
                schemas.update({t.name: t.inputSchema for t in await list_tools()})
            arguments = _coerce_arguments(arguments or {}, schemas.get(name))
            if name in schemas:
                try:
                    jsonschema.validate(arguments, schemas[name])
                except jsonschema.ValidationError as e:
                    return _err(f"Input validation error: {e.message}")
            oid = session["ontology_id"]
            chunks: ChunkStore = session["chunks"]
            match name:
                case "chunk_collection_switch":
                    fm = arguments.get("field_mapping")
                    session["chunks"] = chunks.for_collection(
                        arguments["collection"],
                        FieldMapping(**{**vars(config.qdrant.field_mapping), **fm}) if fm else None,
                    )
                    return _text({
                        "collection": arguments["collection"],
                        "chunks_total": session["chunks"].count_total(),
                        "chunks_processed": session["chunks"].count_processed(),
                    })

                case "chunk_list":
                    return _text(chunks.list_chunks(
                        doc_id=arguments.get("doc_id"),
                        status=arguments.get("status", "all"),
                        limit=arguments.get("limit", 20),
                        offset=arguments.get("offset"),
                    ))

                case "chunk_read":
                    chunk = chunks.get_chunk(arguments["chunk_id"])
                    return _text(chunk) if chunk else _err(f"Chunk not found: {arguments['chunk_id']}")

                case "chunk_read_batch":
                    return _text(chunks.get_chunks_batch(arguments["chunk_ids"]))

                case "chunk_mark_processed":
                    ok = chunks.mark_processed(arguments["chunk_id"])
                    return _text("ok" if ok else "failed")

                case "concept_list":
                    return _text(await onto.concept_list(
                        oid,
                        root_only=arguments.get("root_only", False),
                        limit=arguments.get("limit", 100),
                    ))

                case "concept_create":
                    source_chunk_id = arguments.get("source_chunk_id")
                    result = await onto.concept_create(
                        oid,
                        label=arguments["label"],
                        definition=arguments["definition"],
                        parent_uri=arguments.get("parent_uri"),
                        aliases=arguments.get("aliases"),
                        lang=arguments.get("language", "en"),
                        source_chunk_id=source_chunk_id,
                    )
                    if result["created"] and embed and (warning := _index(lambda: embed.upsert_concept(
                        result["uri"], arguments["label"], arguments["definition"], source_chunk_id, kind="class"
                    ))):
                        result["warning"] = warning
                    return _text(result)

                case "concept_get":
                    result = await onto.concept_get(oid, arguments["uri"])
                    return _text(result) if result else _err(f"Concept not found: {arguments['uri']}")

                case "concept_search":
                    return _text(await onto.concept_search_sparql(
                        oid, arguments["query"], arguments.get("top_k", 10),
                        include_seeds=arguments.get("include_seeds", False), seed_ids=session["seed_ids"],
                    ))

                case "concept_semantic_search" | "property_semantic_search":
                    if not embed:
                        return _err("Semantic search is disabled (embedding model not loaded)")
                    seed_collections = None
                    if arguments.get("include_seeds", False):
                        prefix = _seed_collection_prefix(config)
                        seed_collections = {c[len(prefix):]: c for c in embed.collections() if c.startswith(prefix)
                                            and (session["seed_ids"] is None or c[len(prefix):] in session["seed_ids"])}
                    return _text(embed.search_concepts(
                        arguments["query"], arguments.get("top_k", 10), seed_collections,
                        properties=name == "property_semantic_search",
                    ))

                case "concept_update":
                    await onto.concept_update(
                        oid,
                        uri=arguments["uri"],
                        label=arguments.get("label"),
                        definition=arguments.get("definition"),
                        aliases_add=arguments.get("aliases_add"),
                        aliases_remove=arguments.get("aliases_remove"),
                        lang=arguments.get("language", "en"),
                    )
                    if embed and (arguments.get("label") or arguments.get("definition")):
                        if warning := await _reindex(arguments["uri"]):
                            return _text({"status": "ok", "warning": warning})
                    return _text("ok")

                case "individual_create":
                    source_chunk_id = arguments.get("source_chunk_id")
                    result = await onto.individual_create(
                        oid,
                        label=arguments["label"],
                        class_uri=arguments["class_uri"],
                        definition=arguments.get("definition"),
                        aliases=arguments.get("aliases"),
                        lang=arguments.get("language", "en"),
                        source_chunk_id=source_chunk_id,
                    )
                    if result["created"] and embed and (warning := _index(lambda: embed.upsert_concept(
                        result["uri"], arguments["label"], arguments.get("definition", ""), source_chunk_id,
                        kind="individual",
                    ))):
                        result["warning"] = warning
                    return _text(result)

                case "concept_merge":
                    await onto.concept_merge(oid, arguments["keep_uri"], arguments["merge_uri"])
                    if embed:
                        if warning := (_index(lambda: embed.delete_concept(arguments["merge_uri"]))
                                       or await _reindex(arguments["keep_uri"])):
                            return _text({"status": "ok", "warning": warning})
                    return _text("ok")

                case "property_create":
                    result = await onto.property_create(
                        oid,
                        label=arguments["label"],
                        prop_type=arguments["type"],
                        domain_uri=arguments.get("domain_uri"),
                        range_uri=arguments.get("range_uri"),
                        parent_uri=arguments.get("parent_uri"),
                        lang=arguments.get("language", "en"),
                        source_chunk_id=arguments.get("source_chunk_id"),
                    )
                    if result["created"] and embed and (warning := await _reindex(result["uri"])):
                        result["warning"] = warning
                    return _text(result)

                case "property_update":
                    await onto.property_update(
                        oid,
                        uri=arguments["uri"],
                        domain_uri=arguments.get("domain_uri"),
                        range_uri=arguments.get("range_uri"),
                        parent_uri=arguments.get("parent_uri"),
                    )
                    # domain/range are part of the indexed text
                    if embed and (warning := await _reindex(arguments["uri"])):
                        return _text({"status": "ok", "warning": warning})
                    return _text("ok")

                case "property_get":
                    result = await onto.property_get(oid, arguments["uri"])
                    return _text(result) if result else _err(f"Property not found: {arguments['uri']}")

                case "property_search":
                    return _text(await onto.property_search(
                        oid, arguments["query"], arguments.get("top_k", 10),
                        include_seeds=arguments.get("include_seeds", False), seed_ids=session["seed_ids"],
                    ))

                case "relation_add":
                    await onto.relation_add(
                        oid,
                        subject_uri=arguments["subject_uri"],
                        property_uri=arguments["property_uri"],
                        object_value=arguments["object_value"],
                        is_literal=arguments.get("is_literal", False),
                        datatype=arguments.get("datatype"),
                        source_chunk_id=arguments.get("source_chunk_id"),
                    )
                    return _text("ok")

                case "relation_sources":
                    return _text(await onto.relation_sources(
                        oid,
                        subject_uri=arguments["subject_uri"],
                        property_uri=arguments["property_uri"],
                        object_value=arguments["object_value"],
                        is_literal=arguments.get("is_literal", False),
                        datatype=arguments.get("datatype"),
                    ))

                case "relation_delete":
                    await onto.relation_delete(
                        oid,
                        subject_uri=arguments["subject_uri"],
                        property_uri=arguments["property_uri"],
                        object_value=arguments["object_value"],
                        is_literal=arguments.get("is_literal", False),
                        datatype=arguments.get("datatype"),
                    )
                    return _text("ok")

                case "relation_search":
                    return _text(await onto.relation_search(
                        oid,
                        subject_uri=arguments.get("subject_uri"),
                        property_uri=arguments.get("property_uri"),
                        object_uri=arguments.get("object_uri"),
                        limit=arguments.get("limit", 20),
                    ))

                case "restriction_add":
                    bnode = await onto.restriction_add(
                        oid,
                        class_uri=arguments["class_uri"],
                        property_uri=arguments["property_uri"],
                        restriction_type=arguments["restriction_type"],
                        value=arguments["value"],
                        cardinality=arguments.get("cardinality"),
                        is_literal_value=arguments.get("is_literal_value", False),
                    )
                    return _text({"restriction_bnode": bnode})

                case "restriction_delete":
                    return _text(await onto.restriction_delete(
                        oid,
                        class_uri=arguments["class_uri"],
                        property_uri=arguments["property_uri"],
                        restriction_type=arguments.get("restriction_type"),
                        value=arguments.get("value"),
                        is_literal_value=arguments.get("is_literal_value", False),
                    ))

                case "entity_delete":
                    result = await onto.entity_delete(oid, arguments["uri"])
                    if embed and (warning := _index(lambda: embed.delete_concept(arguments["uri"]))):
                        result["warning"] = warning
                    return _text(result)

                case "disjoint_add":
                    return _text(await onto.disjoint_add(oid, arguments["class_uris"]))

                case "seed_term_get":
                    return _text(await onto.seed_term(arguments["uri"], seed_ids=session["seed_ids"]))

                case "seed_reuse":
                    result = await onto.seed_reuse(oid, arguments["uri"], arguments.get("source_chunk_id"),
                                                   seed_ids=session["seed_ids"])
                    if result["created"] and embed and (warning := await _reindex(result["uri"])):
                        result["warning"] = warning
                    return _text(result)

                case "seed_list":
                    selected = session["seed_ids"]
                    return _text([{**s, "selected": selected is None or s["id"] in selected}
                                  for s in await onto.seed_list()])

                case "seed_select":
                    known = {s["id"] for s in await onto.seed_list()}
                    seed_ids = list(dict.fromkeys(arguments["seed_ids"]))
                    if missing := [i for i in seed_ids if i not in known]:
                        return _err(f"Unknown seed(s): {', '.join(missing)} (loaded: {', '.join(sorted(known)) or 'none'})")
                    session["seed_ids"] = seed_ids
                    return _text({"selected": seed_ids})

                case "seed_load":
                    result = await onto.seed_load(
                        seed_id=arguments.get("seed_id"),
                        url=arguments.get("url"),
                        content=arguments.get("content"),
                        fmt=arguments.get("format"),
                    )
                    if not embed:
                        result["warning"] = "Embeddings disabled: seed not indexed for semantic search"
                    else:
                        try:
                            result["indexed_entities"] = await _index_seed(onto, embed, config, result["seed_id"])
                        except Exception as e:
                            print(f"Warning: seed not indexed ({e})", file=sys.stderr)
                            result["warning"] = f"Seed loaded, but not indexed for semantic search: {e}"
                    return _text(result)

                case "ontology_list":
                    return _text(await onto.ontology_list(oid))

                case "ontology_create":
                    base_uri = arguments.get("base_uri") or session["base_uri"]
                    return _text(await onto.ontology_create(
                        ontology_id=arguments["ontology_id"],
                        name=arguments["name"],
                        base_uri=base_uri,
                    ))

                case "ontology_switch":
                    result = await onto.ontology_switch(arguments["ontology_id"])
                    session["ontology_id"] = result["active_id"]
                    session["base_uri"] = result["base_uri"]
                    if embed:
                        embed.switch_collection(_ontology_collection(config, result['active_id']))
                    return _text(result)

                case "ontology_summary":
                    summary = await onto.summary(oid)
                    try:
                        summary["chunks_total"] = chunks.count_total()
                        summary["chunks_processed"] = chunks.count_processed()
                    except Exception as e:
                        summary["chunks_warning"] = f"Qdrant unavailable: {e}"
                    return _text(summary)

                case "ontology_export":
                    return _text(await onto.export_ttl(oid, include_seeds=arguments.get("include_seeds", False),
                                                       seed_ids=session["seed_ids"]))

                case "ontology_orphans":
                    return _text(await onto.orphans(oid, limit=arguments.get("limit", 100)))

                case "ontology_check":
                    report: dict[str, Any] = {}
                    if reasoner:
                        try:
                            data = await onto.reasoner_input(oid, include_seeds=arguments.get("include_seeds", False),
                                                             seed_ids=session["seed_ids"])
                            max_explanations = max(1, min(int(arguments.get("max_explanations", 1)), 5))
                            report["reasoner"] = await reasoner.check(data, max_explanations=max_explanations)
                        except Exception as e:
                            report["reasoner"] = {"error": str(e)}
                    else:
                        report["reasoner"] = {"error": "Reasoner disabled in config ([reasoner] enabled = false)"}
                    report["integrity"] = await onto.integrity_checks(oid, limit=arguments.get("limit", 50))
                    r = report["reasoner"]
                    report["issues"] = (
                        (0 if r.get("consistent", True) else 1)
                        + len(r.get("unsatisfiable_classes", []))
                        + sum(len(v) for v in report["integrity"].values())
                    )
                    onto.record_check(oid, r.get("consistent"), report["issues"])
                    return _text(report)

                case "ontology_infer":
                    if not reasoner:
                        return _err("Reasoner disabled in config ([reasoner] enabled = false)")
                    action = arguments.get("action", "preview")
                    if action == "clear":
                        return _text({"removed": await onto.inferences_clear(oid)})
                    if action not in ("preview", "materialize"):
                        return _err(f"Unknown action: {action} (expected preview, materialize or clear)")
                    include_seeds = arguments.get("include_seeds", False)
                    data = await onto.reasoner_input(oid, include_seeds=include_seeds, seed_ids=session["seed_ids"])
                    result = await reasoner.infer(data)
                    if not result["consistent"]:
                        return _err("The ontology is inconsistent, so nothing can be inferred from it: "
                                    "fix it first (ontology_check).")
                    triples = result["inferences"]
                    if include_seeds:  # keep inferences about the ontology's entities, not the seeds'
                        own = await onto.reasoner_input(oid)
                        subjects = {line.split(" ", 1)[0][1:-1] for line in own.decode().splitlines() if line}
                        triples = [t for t in triples if t[0] in subjects]
                    if action == "materialize":
                        replaced = len(await onto.inferences_list(oid))
                        stored = await onto.inferences_store(oid, [(s, p, f"<{o}>") for s, p, o in triples], by="pellet")
                        return _text({"stored": stored, "replaced": replaced})
                    max_explained = max(0, min(int(arguments.get("max_explained", 20)), 100))
                    stored = {(s, p, o[1:-1]) for s, p, o in await onto.inferences_list(oid)}
                    items = await reasoner.explain_inferences(data, triples, max_explained)
                    for item in items:  # already materialized by an earlier run
                        item["materialized"] = (item["subject"], item["predicate"], item["object"]) in stored
                    return _text({"count": len(triples), "materialized": len(stored), "inferences": items})

                case "sparql_query":
                    limit = max(1, min(int(arguments.get("limit", 100)), 1000))
                    return _text(await onto.sparql_query(arguments["query"], limit=limit))

                case _:
                    return _err(f"Unknown tool: {name}")

        except Exception as e:
            return _err(str(e))

    return server


def _make_sse_app(config: Config):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    onto = OntologyStore(url=config.oxigraph.url)
    qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
    chunks = ChunkStore(config.qdrant.url, config.qdrant.collection, config.qdrant.field_mapping, config.qdrant.api_key)

    @asynccontextmanager
    async def lifespan(app):
        await onto.bootstrap(config.ontology.ontology_id, config.ontology.base_uri, config.ontology.name)
        await _preload_embeddings(config)
        yield

    transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        server = create_server(config, onto=onto, qdrant=qdrant, chunks=chunks)
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=transport.handle_post_message),
        ],
    )


async def _run() -> None:
    config = Config.load()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "sse":
        import uvicorn
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8000"))
        app = _make_sse_app(config)
        uv_config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uv_server = uvicorn.Server(uv_config)
        await uv_server.serve()
    else:
        onto, qdrant, chunks = await _make_shared_infra(config)
        server = create_server(config, onto, qdrant, chunks)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


async def _make_onto(config: Config) -> OntologyStore:
    onto = OntologyStore(url=config.oxigraph.url)
    await onto.bootstrap(config.ontology.ontology_id, config.ontology.base_uri, config.ontology.name)
    return onto


def _exit_unless_orphan(qdrant: QdrantClient, collection: str, error: ValueError) -> None:
    """Nothing to drop in Oxigraph: remove the Qdrant collection if it was left behind,
    otherwise report the error and exit with a failure status."""
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
        print(f"{error} Collection Qdrant orpheline '{collection}' supprimée.")
        return
    print(f"Erreur : {error}", file=sys.stderr)
    sys.exit(1)


def _cmd_drop(ontology_id: str) -> None:
    async def _inner():
        config = Config.load()
        onto = await _make_onto(config)
        qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        collection = _ontology_collection(config, ontology_id)
        try:
            await onto.drop_ontology(ontology_id)
        except ValueError as e:
            _exit_unless_orphan(qdrant, collection, e)
            return
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)
        print(f"Ontologie '{ontology_id}' supprimée (graphe Oxigraph et collection Qdrant).")
    asyncio.run(_inner())


def _cmd_drop_seed(seed_id: str) -> None:
    async def _inner():
        config = Config.load()
        onto = await _make_onto(config)
        try:
            seed_graph(seed_id)
        except ValueError:  # loaded before ids were checked: could not be indexed, no collection
            try:
                await onto.seed_drop(seed_id)
            except ValueError as e:
                print(f"Erreur : {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Seed '{seed_id}' supprimée (graphe Oxigraph).")
            return
        qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        collection = _seed_collection_prefix(config) + seed_id
        try:
            await onto.seed_drop(seed_id)
        except ValueError as e:
            _exit_unless_orphan(qdrant, collection, e)
            return
        if qdrant.collection_exists(collection):
            qdrant.delete_collection(collection)
        print(f"Seed '{seed_id}' supprimée (graphe Oxigraph et collection Qdrant).")
    asyncio.run(_inner())


def _cmd_load_seed(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="olaf load-seed")
    parser.add_argument("source", help="HTTP(S) URL or local file path of the seed ontology")
    parser.add_argument("--id", dest="seed_id", help="Seed id (default: derived from the URL or file name; must then be new)")
    parser.add_argument("--format", choices=list(RDF_FORMATS),
                        help="RDF serialization (default: guessed from Content-Type, extension or content)")
    parser.add_argument("--no-index", action="store_true", help="Do not index the seed in Qdrant")
    args = parser.parse_args(argv)

    async def _inner():
        config = Config.load()
        onto = await _make_onto(config)
        if args.source.startswith(("http://", "https://")):
            result = await onto.seed_load(seed_id=args.seed_id, url=args.source, fmt=args.format,
                                          allow_private_hosts=True)
        else:
            with open(args.source, "rb") as f:
                data = f.read()
            result = await onto.seed_load(seed_id=args.seed_id, content=data, fmt=args.format,
                                          source_name=args.source)
        print(f"Seed '{result['seed_id']}' chargée dans {result['graph_uri']} "
              f"({result['triples_loaded']} triplets, {result['format']}).")
        if args.no_index:
            return
        if not config.embedding.enabled:
            print("Embeddings désactivés dans la config : seed non indexée dans Qdrant.", file=sys.stderr)
            return
        qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        collection = _seed_collection_prefix(config) + result["seed_id"]
        embed = EmbeddingService(config.embedding.model, qdrant, collection)
        count = await _index_seed(onto, embed, config, result["seed_id"])
        print(f"{count} entité(s) indexé(s) dans la collection Qdrant '{collection}'.")
    asyncio.run(_inner())


def _cmd_reindex(ontology_ids: list[str]) -> None:
    """Rebuild ontologies' Qdrant collections from Oxigraph (classes, individuals and properties)."""
    async def _inner():
        config = Config.load()
        if not config.embedding.enabled:
            print("Embeddings désactivés dans la config ([embedding] enabled = false).", file=sys.stderr)
            sys.exit(1)
        onto = await _make_onto(config)
        errors = []
        for ontology_id in ontology_ids:
            try:
                await onto.require_ontology(ontology_id)
            except ValueError as e:
                errors.append(str(e))
        if errors:  # checked before indexing anything, so that no empty collection is created
            print("Erreur : " + " ".join(errors), file=sys.stderr)
            sys.exit(1)
        qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        embed = None
        for ontology_id in ontology_ids:
            collection = _ontology_collection(config, ontology_id)
            embed = embed or EmbeddingService(config.embedding.model, qdrant, collection)
            entities = await onto.ontology_entities(ontology_id)
            count = await asyncio.to_thread(embed.index_collection, collection, entities)
            kinds = {k: sum(e["kind"] == k for e in entities) for k in ("class", "individual", "property")}
            print(f"Ontologie '{ontology_id}' : {count} entité(s) indexée(s) dans '{collection}' "
                  f"({kinds['class']} classes, {kinds['individual']} individus, {kinds['property']} propriétés).")
    asyncio.run(_inner())


def _cmd_index_seed(seed_ids: list[str]) -> None:
    """(Re)index seeds already in Oxigraph — all of them if no id is given."""
    async def _inner():
        config = Config.load()
        if not config.embedding.enabled:
            print("Embeddings désactivés dans la config ([embedding] enabled = false).", file=sys.stderr)
            sys.exit(1)
        onto = await _make_onto(config)
        ids = seed_ids or [s["id"] for s in await onto.seed_list()]
        qdrant = QdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
        embed = None
        failed = False
        for seed_id in ids:
            collection = _seed_collection_prefix(config) + seed_id
            try:
                await onto.require_seed(seed_id)  # before EmbeddingService, which creates the collection
                embed = embed or EmbeddingService(config.embedding.model, qdrant, collection)
                count = await _index_seed(onto, embed, config, seed_id)
                print(f"Seed '{seed_id}' : {count} entité(s) indexé(s) dans '{collection}'.")
            except Exception as e:
                failed = True
                print(f"Seed '{seed_id}' non indexée : {e}", file=sys.stderr)
        if failed:
            sys.exit(1)
    asyncio.run(_inner())


def _cmd_reset_chunks(argv: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="olaf reset-chunks")
    parser.add_argument("doc_id", nargs="?", help="Only reset this document's chunks")
    parser.add_argument("--collection", help="Qdrant collection (default: [qdrant].collection)")
    args = parser.parse_args(argv)

    config = Config.load()
    collection = args.collection or config.qdrant.collection
    chunks = ChunkStore(config.qdrant.url, collection, config.qdrant.field_mapping, config.qdrant.api_key)
    count = chunks.reset_processed(args.doc_id)
    scope = f"du document '{args.doc_id}' " if args.doc_id else ""
    print(f"{count} chunk(s) {scope}de la collection '{collection}' remis en attente.")


def main() -> None:
    if len(sys.argv) >= 2:
        match sys.argv[1]:
            case "drop":
                if len(sys.argv) < 3:
                    print("Usage: olaf drop <ontology_id>", file=sys.stderr)
                    sys.exit(1)
                _cmd_drop(sys.argv[2])
            case "drop-seed":
                if len(sys.argv) < 3:
                    print("Usage: olaf drop-seed <seed_id>", file=sys.stderr)
                    sys.exit(1)
                _cmd_drop_seed(sys.argv[2])
            case "load-seed":
                _cmd_load_seed(sys.argv[2:])
            case "index-seed":
                _cmd_index_seed(sys.argv[2:])
            case "reindex":
                if len(sys.argv) < 3:
                    print("Usage: olaf reindex <ontology_id> [ontology_id ...]", file=sys.stderr)
                    sys.exit(1)
                _cmd_reindex(sys.argv[2:])
            case "reset-chunks":
                _cmd_reset_chunks(sys.argv[2:])
            case _:
                asyncio.run(_run())
    else:
        asyncio.run(_run())


if __name__ == "__main__":
    main()

# OLAF

**Ontology Learning Agentic Framework** — MCP server for building OWL/RDFS ontologies from text, incrementally and collaboratively with an LLM, and for querying them.

---

## Overview

OLAF exposes a set of MCP tools that let an LLM agent construct a formal OWL/RDFS ontology from raw text chunks. The agent reads chunks, extracts concepts and relations, checks for duplicates, and builds up the ontology piece by piece — resuming at any point without losing work.

Once built, the ontology can be queried the same way: read-only tools and `sparql_query` let an agent answer natural-language questions and trace each fact back to its source chunks. See [`demos/`](demos/) for a building agent, a reasoning agent that repairs the built ontology, and a searching agent.

```
Text chunks (Qdrant)  ──►  LLM agent  ──►  OWL ontology (Oxigraph)
                              │  ▲
                         seeds│  │ concept_search / *_semantic_search
                              ▼  │
                          40 MCP tools
```

**Storage split:**
- **Oxigraph** — the ontologies (OWL/RDFS triples, named graphs, SPARQL), runs as a separate HTTP service
- **Qdrant** — your existing chunk collections (read + status tracking; each agent picks its collection with `chunk_collection_switch`) + one `olaf_index_{id}` collection per ontology and one `olaf_index_seed_{seed_id}` per seed for semantic search

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- A running [Qdrant](https://qdrant.tech/) instance with your chunk collection
- An MCP-compatible client (Claude Desktop, Claude Code, or any MCP host)

---

## Deployment

### Docker Compose (recommended)

```bash
git clone <repo>
cd olaf
cp config.example.toml config.toml
$EDITOR config.toml
docker compose up -d
```

This starts two services:
- **oxigraph** — RDF triplestore on port 7878, data persisted in a Docker volume
- **olaf** — MCP server on port 8000 (SSE transport). The image includes the Java runtime used by the reasoner (`ontology_check`).

On first start, the olaf service downloads the embedding model (a few hundred MB, ~1.1 GB for `multilingual-e5-base`) before accepting connections — wait for `Application startup complete` in `docker compose logs olaf`. The model is cached in the `olaf_models` volume, so later starts are fast.

### Without Docker Compose (Oxigraph / Qdrant already running)

If an Oxigraph or Qdrant instance is already running on this machine, `docker compose up` fails on the busy ports. Run only the OLAF image instead, on the host network so that `localhost` in `config.toml` reaches them:

```bash
docker build -t olaf .
docker run --rm --network host \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v olaf_models:/models \
  olaf
```

CLI commands run the same way, e.g. `docker run --rm --network host -v "$PWD/config.toml:/app/config.toml:ro" olaf olaf reset-chunks`.

### Connecting from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "olaf": {
      "url": "http://your-server:8000/sse"
    }
  }
}
```

For a local Docker deployment, use `http://localhost:8000/sse`.

---

## Configuration

`config.toml` (or `olaf.toml`) is loaded automatically from the working directory.

```toml
[qdrant]
url                 = "https://your-qdrant-instance.example.com"
collection          = "chunks"           # default chunk collection — each agent can pick its own
index_prefix        = "olaf_index"       # semantic index — collections olaf_index_{ontology_id}, olaf_index_seed_{seed_id}

[qdrant.field_mapping]
# default names of the payload fields in your chunk collections
text        = "text"
doc_id      = "doc_id"
chunk_index = "chunk_index"

[oxigraph]
url = "http://oxigraph:7878"   # Docker service name; use http://localhost:7878 if running locally

[ontology]
base_uri     = "http://olaf.local/ontology#"   # namespace for generated URIs
name         = "My Ontology"
ontology_id  = "main"                          # active ontology (urn:olaf:main)

[reasoner]
enabled         = true     # ontology_check: Pellet OWL reasoner (needs Java 11+, included in the image)
java            = "java"   # path to the java executable
memory_mb       = 2048     # JVM max heap
timeout_seconds = 120      # per reasoner call

[embedding]
model   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Any model fastembed supports (TextEmbedding.list_supported_models()), plus
# "intfloat/multilingual-e5-small" (384 d) and "intfloat/multilingual-e5-base" (768 d),
# loaded from their ONNX export on Hugging Face (~0.5 GB / ~1.1 GB, downloaded on first start).
# Concepts are indexed when they are created, and an existing index is not recomputed when
# the model changes (vector sizes differ): after changing it, build into a new ontology_id.
enabled = true
```

**The `olaf_status` and `olaf_processed_at` payload fields are added to your existing Qdrant points via `set_payload` — your other fields are never modified.**

---

## Tool reference

### Ontology management

| Tool | Description |
|------|-------------|
| `ontology_list` | List all ontologies in the store with id, name, base_uri, class count, and active flag. |
| `ontology_create` | Create a new empty ontology. Parameters: `ontology_id` (letters, digits, `_`, `-`, `.`; not starting with `seed_`, reserved for seeds' Qdrant collections), `name`, optional `base_uri`. |
| `ontology_switch` | Switch the active ontology for this session. All subsequent tools operate on the selected ontology. |
| `ontology_summary` | Counts of classes, properties, individuals, restrictions, root classes, and chunk processing progress. Includes `active_ontology`, the number of materialized inferences (`inferred_triples`) and the outcome of the last `ontology_check` since the server started (`last_check`: time, consistent, issues — later changes are not reflected until the next check). |
| `ontology_export` | Export the active ontology as Turtle. Pass `include_seeds=true` to append all seed graphs. |
| `ontology_orphans` | List classes/individuals with no relation to the rest of the graph beyond their own type/label/definition triples. |
| `ontology_check` | Check the ontology for logical errors: Pellet OWL 2 DL reasoning (consistency + unsatisfiable classes, each with its explanation) plus closed-world SPARQL checks (subclass cycles, untyped individuals, domain/range violations, object/datatype property misuse, unknown vocabulary terms such as `rdfs:subClassof`). See [Consistency checking](#consistency-checking). |
| `ontology_infer` | What the ontology entails but does not assert (Pellet): subclass relations, class membership of individuals, equivalent classes, sub-properties, property assertions, same-individual links. `action=preview` (default) lists them with why each holds; `materialize` writes them into the ontology, marked as inferred; `clear` removes them. See [Inference](#inference). |

### Chunks

| Tool | Description |
|------|-------------|
| `chunk_collection_switch` | Switch the Qdrant collection the chunk tools read from, for this session only (default: `[qdrant].collection`). Optional `field_mapping` overrides the payload field names. Returns chunk counts. |
| `chunk_list` | List chunks from Qdrant. Filter by `doc_id` and/or `status` (`pending`/`processed`/`all`). Returns `id`, `doc_id`, `chunk_index`, `text_preview` (200 chars), `status`. |
| `chunk_read` | Read the full text of a chunk by its Qdrant point ID. |
| `chunk_read_batch` | Read the full text of several chunks in one call. Returns `{id, doc_id, chunk_index, text, status}` for each. |
| `chunk_mark_processed` | Mark a chunk as processed after extracting ontology elements from it. |

### Concepts (`owl:Class`) and individuals

| Tool | Description |
|------|-------------|
| `concept_list` | List the classes of the active ontology with `uri`, `label`, `definition` and `parent_uri`. `root_only=true` returns only top-level classes. |
| `concept_search` | Substring search on `rdfs:label` and `rdfs:altLabel` via SPARQL. Returns `source_chunk_id` when available. Call before `concept_create`. With `include_seeds=true`, also searches the seeds' classes (`rdfs:label` or `skos:prefLabel`); each result then has a `seed` field. |
| `concept_semantic_search` | Vector similarity search in the active ontology's Qdrant collection. Finds near-duplicates even with different wording. Returns `source_chunk_id` when available. Call alongside `concept_search`. With `include_seeds=true`, also searches every seed collection and merges results by score; each result then has a `seed` field (`null` for the active ontology). |
| `concept_create` | Create an `owl:Class`. Generates a CamelCase URI from the label. Optional `source_chunk_id` to record which chunk the concept was extracted from. Returns `{uri, created}` — `created=false` if the URI already exists. |
| `concept_get` | Get all triples for a concept: type, label, definition, aliases, `subClassOf`, `source_chunk_ids`, and its OWL restrictions as `restrictions` (in `restriction_add` / `restriction_delete` terms). |
| `concept_update` | Update label, definition, or aliases (`aliases_add` / `aliases_remove`). |
| `individual_create` | Create an `owl:NamedIndividual` (a specific named entity such as "GDPR") as an instance of `class_uri`. Returns `{uri, created}`. If it already exists, `source_chunk_id` is added to its sources. |
| `concept_merge` | Merge two concepts: all triples from `merge_uri` move to `keep_uri`, all references re-pointed, `merge_uri` deleted. The kept concept keeps its label and definition: the merged one's labels become `rdfs:altLabel`, and its definition is used only if the kept one has none. A relation between the two (e.g. `merge_uri` subClassOf `keep_uri`) is dropped rather than turned into a self-loop. `merge_uri` must be the ontology's own; `keep_uri` may be a reused seed term, to replace a local duplicate of it — then only the references and source chunks are carried over (see [Reusing seed terms](#reusing-seed-terms)). The provenance of the re-pointed relations is kept (`relation_sources` on the new triples). |

### Properties (`owl:ObjectProperty` / `owl:DatatypeProperty`)

| Tool | Description |
|------|-------------|
| `property_create` | Create a property. `type` = `"object"` (links two classes) or `"datatype"` (class → literal). Generates a lowerCamelCase URI. |
| `property_update` | Update `domain_uri`, `range_uri`, or `parent_uri` after creation. Pass `""` to clear a field, omit to leave unchanged. |
| `property_get` | Get all triples for a property. |
| `property_search` | Substring search on property labels. With `include_seeds=true`, also searches the seeds' properties (`rdfs:label` or `skos:prefLabel`); each result then has a `seed` field. |
| `property_semantic_search` | Vector similarity search on the properties of the active ontology's Qdrant collection (embedded as `label (domain → range)`). Finds near-duplicate properties with different wording. With `include_seeds=true`, also searches the seeds' properties, to reuse one instead of creating it. Results include `property_type`, `domain`, `range` and, with seeds, `seed`. |

### Relations & Restrictions

| Tool | Description |
|------|-------------|
| `relation_add` | Insert any triple `(subject, property, object)`. Use full URIs. Set `is_literal=true` for literal objects; optionally pass `datatype` (XSD URI). Rejects unknown subject/property/object URIs (see [Reusing seed terms](#reusing-seed-terms)), a seed class or property as subject, and RDF/RDFS/OWL/SKOS terms that do not exist (e.g. `rdfs:subClassof`, with the closest real term as a hint). |
| `relation_delete` | Remove a triple `(subject, property, object)`. Same parameters as `relation_add`. |
| `relation_search` | Find triples by pattern. All three parameters are optional. |
| `relation_sources` | Get the chunk IDs a `(subject, property, object)` triple was extracted from. |
| `restriction_add` | Add an `owl:Restriction` blank node to a class. Supports `some`, `all`, `has_value`, `exactly`, `min`, `max`. |
| `restriction_delete` | Delete the restrictions of a class on a property — all, or those of a given `restriction_type` and/or `value`. Reaches the blank nodes `relation_delete` cannot. |
| `entity_delete` | Delete a class, individual or property of the ontology entirely: every triple it appears in, their provenance, its restrictions and those pointing to it, its semantic-index entry. Only entities of the ontology's namespace, or seed terms it reuses (their declaration and uses are removed; the seed is unchanged). Irreversible. |
| `disjoint_add` | Declare 2+ classes pairwise disjoint (`owl:disjointWith`). Rejects non-classes and classes in a subclass relation. Undo a pair with `relation_delete`. |

`restriction_add` produces:
```turtle
:Car rdfs:subClassOf [
    a owl:Restriction ;
    owl:onProperty :hasEngine ;
    owl:someValuesFrom :Engine
] .
```

### Consistency checking

`ontology_check` combines two kinds of checks, because OWL semantics alone catch few of the mistakes an extraction agent makes:

- **Reasoner (open world).** The [Pellet](https://github.com/stardog-union/pellet) reasoner bundled with `owlready2` runs on the ontology's logical content (provenance triples and `owl:imports` are stripped). It reports whether the ontology is **consistent** and lists **unsatisfiable classes** — classes that can have no instance. Each problem comes with an *explanation*: the minimal set of axioms causing it, e.g. `Organisation disjointWith Person`, `Employee subClassOf Organisation`, `Employee subClassOf Person`. `entities` maps the local names used in explanations to full URIs. If the ontology is inconsistent, unsatisfiable classes are not computed: fix the inconsistency first and check again.
- **Integrity checks (closed world, SPARQL).** In OWL, `rdfs:domain`/`rdfs:range` do not *constrain* — they *infer* types — and a relation between two classes (punning) is not constrained at all; a subclass cycle silently means equivalence. These checks report what the reasoner considers fine but is almost always an extraction error — and **unknown vocabulary terms**: a typo such as `rdfs:subClassof` makes a triple no tool or reasoner understands, so it silently means nothing; each one comes with the closest real term as `suggestion`. `rdfs:altLabel`, which OLAF has always used for aliases, is accepted although the standard term is `skos:altLabel`.

The reasoner can only find contradictions the ontology states. **Without `owl:disjointWith` axioms, an ontology is almost never inconsistent** — declare sibling classes that cannot overlap disjoint with `disjoint_add`.

[`olaf_reasoning_agent`](demos/olaf_reasoning_agent/) runs these checks on a built ontology and has an LLM repair what they report.

Requires Java 11+ (included in the Docker image). Settings live under `[reasoner]` in `config.toml` (`enabled`, `java`, `memory_mb`, `timeout_seconds`). When the reasoner is disabled or Java is missing, `ontology_check` still runs the integrity checks and reports the reasoner error.

### Inference

`ontology_infer` runs Pellet on a **consistent** ontology and returns what it entails but does not assert: indirect superclasses, the inherited types of individuals, and what follows from domains, ranges, equivalences, restrictions and property axioms. Trivial entailments (`X ⊑ owl:Thing`, `X ≡ X`…) are left out, and so is the type a class gets when it is used as an individual in a relation between classes (punning).

- **`preview`** lists the inferences, each with why it holds. *Taxonomic* ones follow from the asserted class hierarchy alone and come with their chain of `rdf:type` / `rdfs:subClassOf` axioms; the reasoner explains the others (`max_explained`, a few seconds each). An absurd inference — e.g. an agent inferred to be a language model — reveals a wrong axiom in its explanation: fix that axiom, not the inference.
- **`materialize`** writes them into the ontology graph, replacing those written before. Each inferred triple gets a reification node marked `<urn:olaf:inferredBy> "pellet"` (and no source chunk), so it can be told apart from what was extracted. Asserting an inferred triple later (`relation_add`, …) removes its mark.
- **`clear`** removes them. `concept_merge` does too, as a merge changes what can be inferred.

`ontology_check` and `ontology_orphans` ignore materialized inferences: an inferred `rdf:type` would otherwise hide the very domain/range violation that caused it. With `include_seeds=true`, the seeds take part in the reasoning, but only inferences about the ontology's own entities are kept.

[`olaf_reasoning_agent`](demos/olaf_reasoning_agent/) has an LLM review the new inferences, then materializes them.

### Querying

| Tool | Description |
|------|-------------|
| `sparql_query` | Run a read-only SPARQL query (`SELECT` / `ASK` / `CONSTRUCT` / `DESCRIBE`; updates are rejected). No implicit default graph: target `GRAPH <urn:olaf:{id}>` or `urn:olaf:seed:*`. Optional `limit` (default 100, max 1000) for SELECT rows. |

### Seeds

Seeds are **global** — shared across all ontologies in the store.

| Tool | Description |
|------|-------------|
| `seed_list` | List all seed ontologies loaded in the store (id, graph URI, triple count, `selected` for this session). |
| `seed_select` | Select the seeds this session works with, by id (`[]` = none; default: all). `include_seeds` searches, `seed_reuse`, and `ontology_check` / `ontology_infer` / `ontology_export` with `include_seeds` then only see these. The demo agents call it from `[olaf] seed_ids` in their config. |
| `seed_load` | Load a seed ontology via `url` (HTTP URI) or `content` (raw RDF string), with an optional `seed_id` (default: derived from the URL, e.g. `http://xmlns.com/foaf/0.1/` → `foaf`, and then it must be new) and `format` (`ttl`, `xml`, `nt`, `n3`, `jsonld`; default: guessed from the Content-Type, extension or content). Stored in `urn:olaf:seed:{seed_id}`, accessible to all ontologies; passing an existing `seed_id` replaces it. Over MCP, the URL must resolve to a public address (checked at every redirect) — internal hosts go through the CLI — and the download is capped at 200 MB. Its classes, individuals and properties are indexed in the Qdrant collection `{index_prefix}_seed_{seed_id}`. |
| `seed_term_get` | A seed term as its seed defines it: label, definition, parents, subclasses (or subproperties), an individual's classes, a property's domain and range — to check its meaning before `seed_reuse`. Read-only; follows `seed_select`. |
| `seed_reuse` | Reuse a seed term (class, individual or property) in the active ontology under its own URI: copies its minimal declaration and indexes it. Idempotent (`created=false` if already reused). See below. |

To load a **local file**, use the CLI (`olaf load-seed`, below): an MCP call runs on the server, which cannot read the client's files.

### Reusing seed terms

When a seed term has the meaning the ontology needs, the ontology reuses its IRI rather than minting a local one — the usual practice for interoperable ontologies (W3C *Best Practices for Publishing Linked Data*, "reuse vocabularies"; NeOn methodology). OLAF follows the MIREOT approach (Courtot et al., 2011): instead of importing the whole seed, `seed_reuse` copies into the ontology the term's minimal declaration —
- its OWL type (`owl:Class`, `owl:NamedIndividual`, `owl:ObjectProperty` / `owl:DatatypeProperty`; an `rdf:Property` gets one from its range);
- its labels (`rdfs:label`, `skos:prefLabel` → `rdfs:label`) and definitions (`skos:definition`, `rdfs:comment` → `skos:definition`);
- its parents (`rdfs:subClassOf` / `rdfs:subPropertyOf`), an individual's classes, a property's domain and range;
- `rdfs:isDefinedBy` its source ontology (the term's own, else the seed ontology whose IRI prefixes the term's, else the term's namespace), and the source chunk if given.

Labels and definitions are copied once per distinct text; a term without a label in its seed gets its IRI's local name (`LegalEntity` → "Legal Entity").

The reused term then shows up like any other in `concept_get`, `concept_list`, `ontology_summary`, the export, the checks and the semantic index. A narrower notion is a local class with the seed term as `parent_uri`.

A reused term is never redefined — that would be *ontology hijacking* (Hogan et al., 2009): `concept_update`, `property_update`, `restriction_add` on it, `concept_merge` with it as `merge_uri`, and `relation_add` with a seed class or property as subject are refused (a reused individual can be a subject). `disjoint_add` writes the local class as subject.

All write tools check the URIs they reference: a URI of the ontology's namespace must exist in it; an external URI must be declared in it (i.e. reused) or belong to a standard vocabulary (rdf, rdfs, owl, skos, xsd). A seed term not yet reused is refused with a hint to call `seed_reuse`; any other unknown URI (a typo, an invented one) is refused. Two exceptions, which are not uses of a term: the object of a linking property (`owl:sameAs`, `rdfs:seeAlso`, `rdfs:isDefinedBy`, `skos:exactMatch`/`closeMatch`/`broadMatch`/`narrowMatch`/`relatedMatch`) may be any URI, e.g. a Wikidata entity; and a datatype property's range may be any datatype (e.g. `geo:wktLiteral`). To use another external vocabulary (Dublin Core, schema.org…), load it as a seed.

A local duplicate of a seed term — class or property — is replaced with `seed_reuse` then `concept_merge(keep_uri=<seed URI>, merge_uri=<local URI>)`: its uses (as object or as property) move to the seed term, its own statements, their provenance and its restrictions are dropped.

### Semantic index

Ontologies and seeds are indexed the same way, one Qdrant point per entity, tagged with its `kind`:
- classes (`owl:Class`, `rdfs:Class`) and individuals (`owl:NamedIndividual`) as `label. definition` — searched by `concept_semantic_search`;
- properties (`owl:ObjectProperty`, `owl:DatatypeProperty`, `rdf:Property`) as `label (domain → range). definition` — searched by `property_semantic_search`.

The label is `rdfs:label` or `skos:prefLabel` (English or untagged preferred, else the IRI's local name), the definition `skos:definition` or `rdfs:comment`. Scores are only comparable to each other — with some embedding models (e.g. the e5 family), even unrelated entries score above 0.8 — so judge a match by its content, not by a threshold. OWL vocabulary terms a seed redeclares (`owl:Thing`…) are skipped. An ontology is indexed as it is built (`concept_create`, `individual_create`, `property_create`, updates, merges, deletions); `olaf reindex` rebuilds its index from Oxigraph, e.g. to index the properties of an ontology built before they were.

---

## CLI commands

Administrative operations not exposed to the MCP agent:

```bash
# Delete an ontology: Oxigraph graph and Qdrant collection (does not affect seeds)
olaf drop <ontology_id>

# Load a seed from a URL or a local file (Turtle, RDF/XML, N-Triples, N3, JSON-LD) into Oxigraph,
# and index its classes, individuals and properties in Qdrant. --id of an existing seed replaces
# it; without --id, the id is derived from the URL or file name and must be new.
# Unlike seed_load over MCP, URLs on private/internal hosts are allowed.
olaf load-seed <url|file> [--id SEED_ID] [--format ttl|xml|nt|n3|jsonld] [--no-index]

# Rebuild ontologies' semantic index (Qdrant) from Oxigraph: classes, individuals and properties
olaf reindex <ontology_id> [ontology_id ...]

# (Re)index seeds already in Oxigraph in Qdrant — all of them if no id is given
olaf index-seed [seed_id ...]

# Delete a global seed (Oxigraph graph and Qdrant collection)
olaf drop-seed <seed_id>

# Mark processed chunks as pending again (all, or one document's), to rebuild an ontology.
# Removes only the olaf_status / olaf_processed_at payload fields.
olaf reset-chunks [doc_id] [--collection NAME]
```

An unknown ontology or seed id is an error (exit code 1) for `drop`, `drop-seed`, `reindex` and `index-seed`; nothing is created or deleted. `drop` / `drop-seed` still remove a Qdrant collection left behind by an id whose graph is already gone.

---

## Typical agent workflow

```
1. ontology_list()                  → discover existing ontologies
2. ontology_switch("my-project")    → or ontology_create("my-project", "My Project")
   chunk_collection_switch("docs")   → optional: chunks from another collection than the server default
3. ontology_summary()               → understand current state
4. seed_list()                      → check available seeds
5. seed_load(url="https://…/domain.ttl", seed_id="domain")
                                    → optionally load a reference ontology (local file: olaf load-seed)
   seed_select(["domain", …])       → optionally restrict the session to some seeds (default: all)

For each chunk:
6. chunk_list(doc_id=..., status="pending")
7. chunk_read(chunk_id)
8. concept_search(query, include_seeds=true) +           → run in parallel, deduplicate by URI;
   concept_semantic_search(query, include_seeds=true)      property_search / property_semantic_search
                                                           for relations
9. a seed term fits? seed_term_get(uri) → is every X a Y? → seed_reuse(uri), then use its URI
   (same meaning: reuse it; narrower: local class with the seed term as parent_uri)
10. concept_create / individual_create / property_create / relation_add / restriction_add / disjoint_add
11. chunk_mark_processed(chunk_id)

12. ontology_orphans()               → connect or justify any isolated entity
13. ontology_check()                 → fix each reported problem, check again until clean
14. ontology_infer()                 → review the inferences, fix the axioms behind absurd ones,
    ontology_infer(action="materialize")  then store them
15. ontology_export()
```

### Querying workflow

Used by [`olaf_searching_agent`](demos/olaf_searching_agent/), with read-only tools only:

```
1. ontology_switch("my-project")
2. concept_search(term) +            → locate the entities named in the question
   concept_semantic_search(term)
3. concept_get / relation_search     → explore their neighbourhood
4. sparql_query(query)               → lists, counts, joins, hierarchies, paths
5. chunk_read_batch(source_chunk_ids) → ground the answer in the source text
```

---

## Named graphs

Oxigraph uses named graphs to separate ontologies and seeds:

| Graph | Contents |
|-------|----------|
| `urn:olaf:{id}` | An ontology (e.g. `urn:olaf:main`, `urn:olaf:legal`) |
| `urn:olaf:seed:{id}` | A seed ontology — global, shared across all ontologies |

The active ontology is set via `ontology_id` in config (default: `main`) and can be changed at runtime with `ontology_switch`.

---

## Architecture

```
                  ┌─────────────────────┐
Claude Desktop ──►│  olaf (port 8000)   │
Cloud agent    ──►│  MCP / SSE          │
                  │                     │
                  │  config.py          │
                  │  ontology.py  ──────┼──► Oxigraph HTTP (port 7878)
                  │  chunks.py    ──────┼──► Qdrant
                  │  embeddings.py──────┼──► Qdrant (olaf_index_*)
                  │  reasoner.py  ──────┼──► Pellet (Java subprocess)
                  │  server.py          │
                  └─────────────────────┘
```

```
src/olaf/
├── config.py       Config dataclasses, TOML loading (stdlib tomllib)
├── ontology.py     OntologyStore — HTTP SPARQL backend, all reads/writes
├── chunks.py       ChunkStore — Qdrant wrapper for existing chunk collection
├── embeddings.py   EmbeddingService — fastembed + olaf_index_* Qdrant collections (semantic index)
├── reasoner.py     Reasoner — runs Pellet (bundled with owlready2) for ontology_check
└── server.py       MCP server, tool registration, call dispatch, CLI commands
```

**URI generation** — concept URIs are derived from labels:
- Classes: `CamelCase` → `http://olaf.local/ontology#MotorVehicle`
- Properties: `lowerCamelCase` → `http://olaf.local/ontology#hasEngine`

Change `base_uri` in config to use your own namespace.

---

## Dependencies

| Package | Role |
|---------|------|
| `mcp` | MCP server SDK (Anthropic) |
| `httpx` | HTTP client — calls Oxigraph SPARQL endpoint and loads remote seeds |
| `qdrant-client` | Qdrant vector database client |
| `fastembed` | Lightweight ONNX-based embeddings (no PyTorch) |
| `starlette` | ASGI framework for SSE transport |
| `uvicorn` | ASGI server |
| `owlready2` | Provides the Pellet OWL reasoner jars used by `ontology_check` (run with Java) |

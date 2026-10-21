# olaf_searching_agent

Conversational agent that answers natural-language questions over an OLAF ontology. It searches the ontology with the OLAF MCP tools, writes and runs SPARQL queries when a question needs them, reads the source chunks to back up its answer, and replies in natural language.

## Architecture

```
question ──▶ agent.py ──SSE──▶ OLAF MCP server ──SPARQL──▶ Oxigraph (ontology)
                │                               ──────────▶ Qdrant (chunks + concept vectors)
LiteLLM ──API──▶ Claude / OpenAI / Ollama
                │
answer   ◀──────┘
```

For each question the agent runs a tool loop, capped at `max_iterations` rounds:

1. **Locate** the entities in the question with `concept_search` and `concept_semantic_search`, plus `property_search` and `property_semantic_search` for relations.
2. **Explore** around them with `concept_get` and `relation_search`.
3. **Query** with `sparql_query` when the question needs lists, counts, joins, hierarchies or paths. The LLM writes the SPARQL; when a query fails or returns nothing, it reads the error and tries a corrected version.
4. **Ground** the answer by following provenance (`urn:olaf:extractedFrom`) back to the chunks and reading them with `chunk_read_batch`. Triples inferred by the reasoner (materialized by [`olaf_reasoning_agent`](../olaf_reasoning_agent/), marked `<urn:olaf:inferredBy>`) have no source chunk: when an answer rests on one, the agent says it is inferred and cites the chunks of the asserted facts it follows from.
5. **Answer** in the language of the question, citing concepts and chunks (`[chunk <id>, doc <doc_id>]`).

The conversation is kept between questions, so follow-ups such as "and which of those are in France?" work. To keep the context small, tool results older than the last `keep_recent_turns` LLM turns are cut to a short preview; the LLM re-runs a tool if it needs the full result again.

### Read-only access

The agent shows the LLM only the read-only OLAF tools: `chunk_list`, `chunk_read`, `chunk_read_batch`, `concept_list`, `concept_get`, `concept_search`, `concept_semantic_search`, `property_get`, `property_search`, `property_semantic_search`, `relation_search`, `relation_sources`, `seed_list`, `seed_term_get`, `ontology_list`, `ontology_summary`, `ontology_export` and `sparql_query`. If the LLM calls any other tool, the agent refuses the call. `sparql_query` refuses updates on its own (only `SELECT` / `ASK` / `CONSTRUCT` / `DESCRIBE`), and Oxigraph's `/query` endpoint cannot run them anyway.

## Prerequisites

- An OLAF MCP server in SSE mode that includes the `sparql_query` tool (see the root `docker-compose.yml`)
- An ontology built with [`olaf_building_agent`](../olaf_building_agent/), so that source chunks are linked
- An API key for your LLM. SPARQL generation works much better with a strong model (Claude Sonnet or GPT-4o class).

## Setup

```bash
cd demos/olaf_searching_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url         = "http://localhost:8000/sse"   # OLAF SSE endpoint
ontology_id = "demo"                        # ontology to search
collection  = "chunks"                      # collection it was built from (default: server's)

[litellm]
model       = "claude-sonnet-4-6"
max_tokens  = 4096
temperature = 0

[agent]
max_iterations        = 20     # LLM ↔ tool rounds per question, then a forced final answer
max_tool_result_chars = 20000  # truncate large tool results before sending them to the LLM
keep_recent_turns     = 4      # older tool results are cut to a preview of…
pruned_result_chars   = 500    # …this many characters
show_sparql           = true   # print generated SPARQL to stderr
log_level             = "INFO"
```

### OpenAI-compatible providers (Scaleway…)

Scaleway Generative APIs expose an OpenAI-compatible endpoint. To use one, prefix the model with `openai/`, set `api_base`, and set `api_key_env` to the name of the environment variable that holds the key:

```toml
[litellm]
model       = "openai/llama-3.3-70b-instruct"   # a model from the Scaleway catalogue that supports tool calling
api_base    = "https://api.scaleway.ai/v1"
api_key_env = "SCW_SECRET_KEY"
```

```bash
export SCW_SECRET_KEY=...   # IAM API key secret
```

The agent does not read `.env` files: export the variable in the shell that runs it.

When the provider answers HTTP 429 (rate limit), the call is retried after 15 s, 30 s, then 60 s, up to `[litellm] rate_limit_retries` times (default 6). Timeouts, connection errors and 5xx answers are retried too, after 10 s, up to `[litellm] transient_retries` times (default 2) — if timeouts keep coming back, the model is too slow for the work asked per call: raise `timeout`, or use a faster model.

## Running

```bash
python agent.py                                    # interactive session
python agent.py -q "Quels instruments financent des projets d'énergie renouvelable ?"
python agent.py --no-show-sparql --log-level WARNING
python agent.py --config my.toml
```

In the interactive session, `/reset` starts a new conversation and `exit` (or Ctrl-D) quits.

Tool calls and generated SPARQL go to stderr. The answer goes to stdout:

```
11:32:01 INFO     Active ontology: demo
11:32:01 INFO     Loaded 16 read-only OLAF tools.

> Quels instruments financent des projets d'énergie renouvelable ?
11:32:03 INFO     Tool call: concept_search(query='renewable energy')
11:32:03 INFO     Tool call: concept_semantic_search(query='instrument financing renewable energy projects')
11:32:06 INFO     Tool call: sparql_query(query='PREFIX rdfs: <…> SELECT ?instr ?label WHERE { GRAPH…')
┌── SPARQL ──────────────────────────────────────────────────
│ PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
│ SELECT DISTINCT ?instr ?label WHERE {
│   GRAPH <urn:olaf:demo> { ?instr <http://olaf.local/ontology#funds> ?p . … }
│ }
└────────────────────────────────────────────────────────────
11:32:09 INFO     Tool call: chunk_read_batch(chunk_ids="['0001b051-4dde-41f0-b713-d37b87391b31', …]")

Deux types d'instruments financent des projets d'énergie renouvelable : …
```

## The `sparql_query` tool

This tool was added to the OLAF server for this demo, and any MCP client can use it. Its input is `query` plus an optional `limit` (100 rows by default, 1000 at most). What it returns depends on the query form:

| Query form | Result |
|---|---|
| `SELECT` | `{form, variables, rows, row_count, truncated}` |
| `ASK` | `{form, boolean}` |
| `CONSTRUCT` / `DESCRIBE` | `{form, turtle, truncated}` |

There is no implicit default graph: queries must target `GRAPH <urn:olaf:{ontology_id}>` or the seed graphs `urn:olaf:seed:*`. When Oxigraph rejects a query, its syntax error is passed back as is, so the LLM can fix the query.

# olaf_building_agent

Autonomous agent that reads text chunks from Qdrant and builds an OWL/RDFS ontology using OLAF MCP tools and LiteLLM.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                                      ──gRPC───▶  Qdrant (chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The code drives the pipeline; the LLM does the ontology work through OLAF tool calls. Each step below is a **fresh LLM conversation**, so the prompt size depends on `batch_size`, not on how far the build has gone — which keeps token usage per call bounded (and within provider rate limits).

1. **Extraction** — pending chunks are processed by batches of `batch_size`. Each batch gets the chunk texts plus a digest: the current ontology's classes and properties, so the LLM reuses them instead of re-creating them, and — from the seeds selected by `[olaf] seed_ids` (all loaded seeds if omitted) — the seed terms closest in meaning to the batch's chunks (semantic search with each chunk's text, `digest_seed_terms`) and the seeds' general classes, those with several subclasses (`digest_seed_parents`), candidate parents of new classes. Once the LLM is done, the pipeline marks the chunks processed: an interrupted run resumes where it stopped.

   **Seeds.** The pipeline sets `include_seeds=true` on the LLM's four search tools, whatever the LLM passes. The prompt applies the usual reuse rules (see the root README, *Reusing seed terms*): a seed term with the same meaning is reused under its own URI (`seed_reuse`), not copied into a local one; a narrower notion becomes a local subclass of the seed term — after asking what kind of thing the new class is, reading the seed term with `seed_term_get`, and passing the is-a test (is *every* instance of the class an instance of the seed class?); a specific named organisation, state or programme becomes an individual of the seed class. Without a definition or a clear context, nothing is linked: a missing parent can be added later, a wrong `rdfs:subClassOf` propagates to every inference.
2. **Export** — the ontology is written to `export_path`.

Curation is not done here: run [`olaf_reasoning_agent`](../olaf_reasoning_agent/) on the ontology once it is built — it merges duplicates, repairs logical problems, adds disjointness and materializes inferences. Export again at its end (its own `export_path`) for the curated version.

Within a step, the LLM loops on tool calls until it replies with a plain-text summary (at most `max_iterations` rounds); older tool results are pruned from its context as it goes.

## Prerequisites

- OLAF MCP server running in SSE mode (see root `docker-compose.yml`)
- Qdrant with a populated chunk collection
- An LLM API key (Anthropic, OpenAI, etc.)

## Setup

```bash
cd demos/olaf_building_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url           = "http://localhost:8000/sse"   # OLAF SSE endpoint
ontology_id   = "demo"                        # ontology to build (created if missing)
ontology_name = "Demo Ontology"
base_uri      = "http://olaf.local/ontology#" # namespace, used when the ontology is created
collection    = "chunks"                      # Qdrant collection to process (default: server's)
seed_ids      = ["fibo-be", "sustaingraph"]   # seeds to use (omit: all loaded seeds; []: none)

[litellm]
model       = "anthropic/claude-haiku-4-5-20251001"  # or claude-sonnet-4-6, gpt-4o, ollama/llama3…
max_tokens  = 4096
temperature = 0

[agent]
batch_size     = 5                       # chunks per extraction batch
max_batches    = 0                       # 0 = all pending chunks
max_iterations = 30                      # max LLM rounds per task
digest_seed_terms   = 30                 # seed terms closest to each batch, listed in its digest
digest_seed_parents = 30                 # seeds' general classes listed (candidate parents)
export_path    = "ontology_output.ttl"   # Turtle file written at the end of the run
log_level      = "INFO"                  # DEBUG | INFO | WARNING
```

See [`config.example.toml`](config.example.toml) for all settings (digest size, context pruning, retries).

**Model strings for LiteLLM** — always prefix with the provider:

| Provider  | Model string                              | API key env var    |
|-----------|-------------------------------------------|--------------------|
| Anthropic | `anthropic/claude-haiku-4-5-20251001`     | `ANTHROPIC_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4-6`             | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai/gpt-4o`                           | `OPENAI_API_KEY`   |
| Ollama    | `ollama/llama3`                           | *(none)*           |


### OpenAI-compatible providers (Scaleway…)

Prefix the model with `openai/`, set `api_base`, and set `api_key_env` to the **name** of the environment variable holding the key (the key itself stays out of the config):

```toml
[litellm]
model       = "openai/llama-3.3-70b-instruct"   # a model of the catalogue that supports tool calling
api_base    = "https://api.scaleway.ai/v1"
api_key_env = "SCW_SECRET_KEY"
```

The agent does not read `.env` files: export the variable in the shell that runs it, e.g. `set -a; source ../../.env; set +a`.

### Rate limits

When the provider answers HTTP 429 (e.g. a tokens-per-minute quota), the call is retried after 15 s, 30 s, then 60 s, up to `[litellm] rate_limit_retries` times (default 6). If it keeps happening, lower `batch_size` (each LLM call carries one batch) and `max_tokens` (some providers count the requested maximum against the quota). Timeouts, connection errors and 5xx answers are retried too, after 10 s, up to `[litellm] transient_retries` times (default 2) — if timeouts keep coming back, the model is too slow for the work asked per call: raise `timeout`, or use a faster model.

## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
python agent.py --log-level DEBUG
```

The agent logs each step and tool call to stderr:
```
11:32:01 INFO     Active ontology: demo (created)
11:32:01 INFO     [batch 1] iteration 1
11:32:05 INFO     Tool call: concept_create(label='Climate Risk', ...)
11:32:40 INFO     [batch 1] done: Added 6 classes, 3 properties…
11:32:40 INFO     Batch 1 done — chunks 5/120 processed, 6 classes, 3 object properties, 0 individuals.
…
12:01:13 INFO     Ontology exported to ontology_output.ttl
12:01:13 INFO     Finished — 84 classes, 41 object properties, 0 datatype properties, 23 individuals; chunks 120/120 processed.
```

To try a configuration on a small sample, set `max_batches` (e.g. `2`); the remaining chunks stay pending for a later run. To rebuild from scratch, reset the chunk statuses with `olaf reset-chunks [--collection NAME]` and use a new `ontology_id`.

## Export

At the end of the run, the ontology is exported as Turtle to `config.toml → [agent] export_path` (overwritten on each run).

To export manually from Oxigraph (replace `ontology_id` with your ontology_id):

```bash
curl -s -X POST http://oxigraph:7878/query \
  -H 'Content-Type: application/sparql-query' \
  -H 'Accept: text/turtle' \
  -d 'CONSTRUCT { ?s ?p ?o } WHERE { GRAPH <urn:olaf:ontology_id> { ?s ?p ?o } }' \
  > ontology.ttl
```

To delete an ontology from Oxigraph:

```bash
curl -s -X POST http://oxigraph:7878/update \
  -H 'Content-Type: application/sparql-update' \
  -d 'DROP GRAPH <urn:olaf:ontology_id>'
```

## Infrastructure (Docker)

The root `docker-compose.yml` runs OLAF and Oxigraph. If Qdrant runs as a standalone container on the host network, add `extra_hosts` to the `olaf` service and point Qdrant's URL to `host.docker.internal`:

```toml
# config.toml
[qdrant]
url = "http://host.docker.internal:6333"

[oxigraph]
url = "http://oxigraph:7878"   # Docker service name
```

```yaml
# docker-compose.yml
services:
  olaf:
    extra_hosts:
      - "host.docker.internal:host-gateway"   # required on Linux/WSL2
```

## Known issues

**LiteLLM async bug** — Some versions of LiteLLM have a bug where `acompletion` returns a coroutine object instead of a response for the Anthropic provider. The agent works around this by using `asyncio.to_thread(litellm.completion, ...)`, with a timeout per call set by `[litellm] timeout` (default 120 s). If you see `RAW RESPONSE: <coroutine object ...>` in the logs, upgrading LiteLLM may fix it; the workaround is already in place.

**URI generation** — Concept labels must be space-separated title-case words (`"Climate Risk"`, not `"ClimateRisk"` or `"climate_risk"`). The server slugifies the label into a PascalCase URI automatically: `"Climate Risk"` → `ClimateRisk`. Passing a single compound word produces incorrect results (`"ClimateRisk"` → `Climaterisk`).

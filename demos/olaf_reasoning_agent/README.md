# olaf_reasoning_agent

Agent that curates an existing OWL/RDFS ontology — typically one built by [`olaf_building_agent`](../olaf_building_agent/): it merges duplicate classes, repairs logical problems, declares missing disjointness, then reviews and materializes what the ontology entails, using the OLAF reasoner (`ontology_check`, `ontology_infer`) and LiteLLM.

## Architecture

```
agent.py  ──SSE──▶  OLAF MCP server  ──SPARQL──▶  Oxigraph (RDF)
                     ontology_check  ──JVM─────▶  Pellet reasoner
                                     ──gRPC────▶  Qdrant (source chunks)
LiteLLM   ──API──▶  Claude / OpenAI / Ollama
```

The code drives the curation; the LLM only decides and applies the changes:

0. **Deduplicate** (`dedup = true`) — classes whose concept embeddings are closer than `dedup_threshold` are reviewed by the LLM, `dedup_pairs_per_task` pairs at a time: merge (`concept_merge`) or keep. It comes first: a merge changes what the checks and the reasoner see. Needs embeddings on the server; otherwise it is skipped with a warning.
1. **Check** — `ontology_check` and `ontology_orphans` run. Problems reported:
   - by the Pellet reasoner: **inconsistency**, **unsatisfiable classes**, each with its explanation (the minimal set of axioms causing it);
   - by the SPARQL integrity checks: **subclass cycles**, **domain/range violations**, **untyped individuals**, **property kind mismatches**, **unknown vocabulary terms** (e.g. `rdfs:subClassof`);
   - and **orphans** (`fix_orphans = true`). The LLM connects an orphan only when the source text says explicitly what it is, and to the most specific class that fits — leaving it is the normal outcome: a wrong `rdfs:subClassOf` to a catch-all class would be false, and propagated to every inference.
2. **Fix** — the problems are split into groups of `problems_per_task`. For each group, the code gathers the context itself: the axioms at stake, the entities involved (label, definition, types, parents, all ancestors, domain/range) and the text of the chunks they were extracted from. For domain/range violations it also lists the classes the property could be widened to (common ancestors) and the property's other uses. A fresh LLM conversation gets it all in its task message and fixes the problems with a few write tools (`relation_delete`, `relation_add`, `property_update`, `concept_update`, `concept_merge`, `restriction_delete` for a wrong OWL restriction, `entity_delete` for a spurious entity); a few read tools (`concept_get`, `property_get`, `concept_search`, `relation_search`) cover what the context misses, such as where to attach an orphan.
3. **Check again** — for up to `max_rounds` rounds. The run stops early when nothing is left, or when a round changed nothing. Several rounds are often needed: the reasoner only lists unsatisfiable classes once the ontology is consistent.
4. **Enrich** (`enrich_disjointness = true`, once the ontology is consistent) — the reasoner can only detect contradictions the ontology states, chiefly through disjointness: without it, an ontology is almost never inconsistent and inferences stay taxonomic. Groups of sibling classes (same parent, or all root classes) with pairs neither disjoint nor in a subclass relation are handed to the LLM, `sibling_groups_per_task` at a time, with their definitions. It declares **pairs** (`disjoint_add` with two classes), only when no single thing could ever be an instance of both — not when one can be a kind, part or example of the other, nor for aspects that can coincide (purpose, sector, instrument, approach) — and justifies each pair in its summary. The check then runs again: when a disjointness declared by this run shows up in an explanation, the repair task flags it (⚠) as the likely wrong axiom, with the exact triple to undo it. Groups the LLM leaves alone are proposed again on the next run.
5. **Infer** (`infer = true`, once the ontology is consistent) — `ontology_infer` lists what the ontology entails but does not assert: indirect superclasses, inherited types of individuals, consequences of domains, ranges, equivalences… Each comes with why it holds (its chain of subclass axioms, or the reasoner's explanation).
   - *Review* (`review_inferences = true`) — the LLM reads them by groups of `inferences_per_task`. An absurd inference reveals a wrong axiom: in this demo's ontology, six agents were inferred to be *language models* because of a single wrong axiom, `LlmAgent ⊑ LargeLanguageModel` (an agent uses an LLM, it is not one). The LLM judges the axioms of each explanation one by one, each on its own ("is an A a kind of B?"): if all are true, the inference is kept, however surprising; otherwise it deletes the specific false link — never a correct, more general one higher up the chain — and quotes in its summary the definition or text that contradicts it. Axioms that are a class's only parent are flagged, so that it does not leave the class orphaned for an imprecision. It never touches the inference itself. Inferences already materialized by an earlier run are not reviewed again.
   - *Check again*, then *materialize* — the inferences are written into the ontology (replacing those of an earlier run), each marked `<urn:olaf:inferredBy> "pellet"` on its reification node, with no source chunk. `ontology_check` ignores them, so they never hide an error; the searching agent sees them and says when an answer rests on one.

Because the context is given upfront, the LLM has little to explore: each task should take a few LLM calls, and the prompt size depends on `problems_per_task`, not on the size of the ontology.

Before the first change — a merge, a fix, a disjointness, a review or the materialization — the whole ontology graph is saved as Turtle in `backup_dir` (see [Undoing a run](#undoing-a-run)).

## Prerequisites

- OLAF MCP server up to date (`entity_delete`, `restriction_delete`, `ontology_infer` — after a code update, rebuild the container: `docker compose up -d --build olaf`; with an older server, the tools it lacks are simply not offered to the LLM), running in SSE mode, with the reasoner enabled (`[reasoner] enabled = true`, Java 11+ — included in the Docker image). Without it, only the SPARQL checks run.
- The ontology to curate, and the Qdrant chunk collection it was built from (for the source text).
- For the deduplication: embeddings enabled on the server (`[embedding] enabled = true`).
- An LLM API key (Anthropic, OpenAI, etc.)

## Setup

```bash
cd demos/olaf_reasoning_agent
pip install -r requirements.txt
cp config.example.toml config.toml   # then edit config.toml
export ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration (`config.toml`)

```toml
[olaf]
url         = "http://localhost:8000/sse"   # OLAF SSE endpoint
ontology_id = "demo"                        # ontology to repair (must exist)
collection  = "chunks"                      # chunk collection it was built from (default: server's)
seed_ids    = ["fibo-be", "sustaingraph"]   # seeds the ontology was built with (omit: all; []: none)

[litellm]
model       = "claude-sonnet-4-6"
max_tokens  = 4096
temperature = 0

[agent]
max_rounds        = 3       # check → fix → check again
problems_per_task = 5       # problems per LLM conversation
dedup             = true    # first, merge near-duplicate classes
fix_orphans       = true    # also connect orphan classes/individuals
enrich_disjointness = true  # declare disjoint sibling classes
infer             = true    # then review and materialize the reasoner's inferences
max_iterations    = 12      # max LLM rounds per task
log_level         = "INFO"
```

`seed_ids` selects the seeds the session sees (`seed_select`): with `include_seeds = true`, the check and the inferences reason over these seeds together with the ontology. The agent does not attach classes to seed classes: its tools don't include `seed_reuse`, and it handles orphans (entities with no relation at all), not classes that merely lack a parent.

See [`config.example.toml`](config.example.toml) for all settings (context size, seeds, retries, backup, export). The `[litellm]` section works as in `olaf_building_agent`, including OpenAI-compatible providers (`api_base` + `api_key_env`).

## Running

```bash
python agent.py                       # uses config.toml in current directory
python agent.py --config my.toml
```

The agent logs each round, task, the model's reasoning text and every fix to stderr:
```
17:20:01 INFO     Active ontology: demo
17:20:04 INFO     Round 1 — consistent: True, unsatisfiable: 0, classes in cycles: 2, domain: 11, range: 8, untyped: 0, kind: 0, orphans: 8
17:20:04 INFO     Backup of the ontology before any change: backups/demo-20261007-172004.ttl
17:20:05 INFO     [round 1 · task 1/6] iteration 1
17:20:31 INFO     [round 1 · task 1/6] The text lists constraints among ontology concepts, so…
17:20:31 INFO     Tool call: relation_delete(subject_uri='http://olaf.local/ontology#OntologyConcept', property_uri='http://www.w3.org/2000/01/rdf-schema#subClassOf', object_value='http://olaf.local/ontology#OntologyConstraint')
17:20:31 INFO     Tool call: property_update(uri='http://olaf.local/ontology#produces', domain_uri='http://olaf.local/ontology#OntologytotoolCompilation')
17:20:58 INFO     [round 1 · task 1/6] done: Removed OntologyConcept subClassOf OntologyConstraint (the text…
…
17:31:40 INFO     Round 2 — consistent: True, unsatisfiable: 0, classes in cycles: 0, domain: 1, range: 0, untyped: 0, kind: 0, orphans: 3
…
17:40:12 INFO     Inferences: 27 entailed but not asserted — 27 from the class hierarchy alone, 0 already materialized by an earlier run.
17:40:13 INFO     [review · task 1/3] iteration 1
17:40:41 INFO     Tool call: relation_delete(subject_uri='http://olaf.local/ontology#LlmAgent', property_uri='http://www.w3.org/2000/01/rdf-schema#subClassOf', object_value='http://olaf.local/ontology#LargeLanguageModel')
…
17:46:02 INFO     Inferences materialized: 21 (replacing 0).
```

With `--log-level DEBUG`, the full task message of each conversation (problems, entities, source text) is logged too; library traces (LiteLLM, HTTP) stay at INFO so the log remains readable.

## Undoing a run

Each run that has something to fix writes `backups/<ontology_id>-<date>-<time>.ttl` before changing anything — a dump of the whole graph, provenance included (`[agent] backup_dir`; set it to `""` to disable). To put the ontology back as it was, replace the graph with it in Oxigraph:

```bash
curl -X PUT 'http://localhost:7878/store?graph=urn:olaf:<ontology_id>' \
  -H 'Content-Type: text/turtle' --data-binary @backups/<ontology_id>-<date>-<time>.ttl
```

Every change is also logged with its full arguments (`Tool call: relation_delete(subject_uri='…', …)`).

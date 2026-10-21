"""
olaf_building_agent — LLM agent that builds an OWL/RDFS ontology using OLAF MCP tools + LiteLLM.

The code drives the pipeline; the LLM does the ontology work through tool calls:
  1. Extraction — pending chunks are processed by batches of `batch_size`; each batch is a fresh
     LLM conversation given the chunks and a digest of the current ontology. The code marks
     the chunks processed once the batch is done.
  2. Export — the ontology is written to `export_path`.

Curation (deduplication, consistency repair, inference) is olaf_reasoning_agent's job: run it
on the ontology once it is built.

Usage:
    python agent.py                      # uses config.toml in current directory
    python agent.py --config my.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_common import LoopSettings, ToolLoop, logger, run, select_context, setup_logging  # noqa: E402

from prompts import BATCH_PROMPT, SYSTEM_PROMPT  # noqa: E402

# Steps done by the code itself (session context, reading/marking chunks, export) or by
# olaf_reasoning_agent (checks, inference, curation): hidden from the LLM, refused if called.
HIDDEN_TOOLS = {
    "ontology_create", "ontology_switch", "chunk_collection_switch",
    "chunk_list", "chunk_mark_processed", "ontology_export", "seed_load", "seed_select",
    "ontology_check", "ontology_orphans", "ontology_infer", "entity_delete",
}

# Searches that always cover the seeds selected for the run (unless none is selected).
SEED_SEARCH_TOOLS = {"concept_search", "concept_semantic_search", "property_search", "property_semantic_search"}

_PROPERTIES_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?p (SAMPLE(?l) AS ?label) (SAMPLE(?t) AS ?type) (SAMPLE(?d) AS ?domain) (SAMPLE(?r) AS ?range) WHERE {{
    GRAPH <urn:olaf:{ontology_id}> {{
        ?p a ?t . FILTER(?t IN (owl:ObjectProperty, owl:DatatypeProperty))
        OPTIONAL {{ ?p rdfs:label ?l }} OPTIONAL {{ ?p rdfs:domain ?d }} OPTIONAL {{ ?p rdfs:range ?r }}
    }}
}} GROUP BY ?p
"""

# General classes of the seeds — those with several subclasses there (LegalEntity, Polity,
# Loan…): the candidate parents of the ontology's new classes. Only classes the seed declares,
# which seed_reuse accepts.
_SEED_PARENTS_QUERY = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?p ?g (SAMPLE(?l) AS ?label) (COUNT(DISTINCT ?c) AS ?children) WHERE {
    GRAPH ?g {
        ?c rdfs:subClassOf ?p . ?p a owl:Class .
        FILTER(isIRI(?c) && isIRI(?p) && !STRSTARTS(STR(?p), "http://www.w3.org/"))
        OPTIONAL { ?p rdfs:label|skos:prefLabel ?l }
    }
    FILTER(STRSTARTS(STR(?g), "urn:olaf:seed:")) __SEEDS__
} GROUP BY ?p ?g HAVING(COUNT(DISTINCT ?c) >= 2) ORDER BY DESC(?children) ?p
"""


def _local_name(uri: str) -> str:
    return uri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]



class BuildingAgent:
    """Code-driven pipeline: extraction by batches of chunks, then export. Each batch is a
    fresh LLM conversation, so the context size does not grow with the corpus."""

    def __init__(self, session: ClientSession, config: dict, mcp_tools: list):
        self.session = session
        self.ontology_id = config["olaf"]["ontology_id"]
        agent_cfg = config.get("agent", {})
        self.batch_size = agent_cfg.get("batch_size", 5)
        self.max_batches = agent_cfg.get("max_batches", 0)
        self.digest_max_entries = agent_cfg.get("digest_max_entries", 150)
        self.digest_seed_terms = agent_cfg.get("digest_seed_terms", 30)
        self.digest_seed_parents = agent_cfg.get("digest_seed_parents", 30)
        self._seed_parents: list[str] | None = None  # digest lines, computed once per run
        self.export_path: str | None = agent_cfg.get("export_path")
        self.seed_ids: list[str] | None = config["olaf"].get("seed_ids")
        self.loop = ToolLoop(
            session, config["litellm"], mcp_tools,
            allowed={t.name for t in mcp_tools} - HIDDEN_TOOLS,
            settings=LoopSettings.from_config(agent_cfg, max_history_turns=20),
            refusal="Error: {tool} is not available for this task — the pipeline handles it.",
            forced_args={t: {"include_seeds": True} for t in SEED_SEARCH_TOOLS} if self.seed_ids != [] else None,
        )

    async def _call(self, tool_name: str, args: dict | None = None) -> str:
        """Tool call made by the pipeline itself; an error stops the run."""
        result = await self.session.call_tool(tool_name, args or {})
        text = result.content[0].text if result.content else ""
        if text.startswith("Error"):
            raise RuntimeError(f"{tool_name} failed: {text}")
        return text

    # ── Phase 1: extraction by batches ────────────────────────────────────

    async def seed_parents(self) -> list[str]:
        """Digest lines for the seeds' general classes (see _SEED_PARENTS_QUERY)."""
        if self._seed_parents is None:
            graphs = " ".join(f"<urn:olaf:seed:{s}>" for s in self.seed_ids or [])
            query = _SEED_PARENTS_QUERY.replace("__SEEDS__", f"VALUES ?g {{ {graphs} }}" if self.seed_ids else "")
            rows = json.loads(await self._call("sparql_query", {"query": query, "limit": self.digest_seed_parents}))["rows"]
            self._seed_parents = [
                f"- {r['label'] or _local_name(r['p'])} <{r['p']}> [seed {(r['g'] or '').removeprefix('urn:olaf:seed:')}, "
                f"{r['children']} subclasses]"
                for r in rows
            ]
        return self._seed_parents

    async def relevant_seed_terms(self, chunks: list[dict]) -> list[str]:
        """Digest lines for the seed classes and properties closest in meaning to the batch's
        chunks (semantic search with each chunk's text), best first."""
        best: dict[str, tuple[float, str]] = {}
        for chunk in chunks:
            query = (chunk.get("text") or "")[:2000]
            for tool, top_k in (("concept_semantic_search", 20), ("property_semantic_search", 10)):
                try:
                    hits = json.loads(await self._call(tool, {"query": query, "top_k": top_k, "include_seeds": True}))
                except RuntimeError as e:  # semantic search disabled on the server
                    logger.warning("No relevant seed terms for this batch: %s", e)
                    return []
                for h in hits:
                    if not h.get("seed") or h["uri"] in best and best[h["uri"]][0] >= h["score"]:
                        continue
                    kind = "property" if tool == "property_semantic_search" else "class/individual"
                    best[h["uri"]] = (h["score"], f"- {h['label']} <{h['uri']}> [{kind}, seed {h['seed']}]"
                                                  + (f" — {h['definition'][:120]}" if h.get("definition") else ""))
        ranked = sorted(best.values(), key=lambda v: v[0], reverse=True)
        return [line for _, line in ranked[:self.digest_seed_terms]]

    async def digest(self, chunks: list[dict]) -> str:
        """Compact view of the current ontology and of the seed terms the batch may reuse,
        given to each batch so the LLM reuses them instead of re-creating them."""
        n = self.digest_max_entries
        summary = json.loads(await self._call("ontology_summary"))
        classes = json.loads(await self._call("concept_list", {"limit": n}))
        labels = {c["uri"]: c["label"] for c in classes}
        props = json.loads(await self._call(
            "sparql_query", {"query": _PROPERTIES_QUERY.format(ontology_id=self.ontology_id), "limit": n}
        ))
        relevant, parents = [], []
        if self.seed_ids != []:  # the seeds selected for the session (seed_select), or all
            relevant = await self.relevant_seed_terms(chunks)
            parents = await self.seed_parents()

        lines = [
            "## Current ontology",
            f"{summary.get('classes', 0)} classes, {summary.get('object_properties', 0)} object properties, "
            f"{summary.get('datatype_properties', 0)} datatype properties, {summary.get('individuals', 0)} individuals.",
        ]
        if classes:
            lines.append(f"\n### Classes ({len(classes)} of {summary.get('classes', len(classes))} — "
                         "use concept_search for the others)")
            for c in classes:
                parent = c.get("parent_uri")
                lines.append(f"- {c['label']} <{c['uri']}>"
                             + (f" ⊂ {labels.get(parent) or f'<{parent}>'}" if parent else ""))
        if props["rows"]:
            lines.append(f"\n### Properties ({len(props['rows'])} of {props['row_count']} — "
                         "use property_search for the others)")
            for p in props["rows"]:
                kind = "object" if (p["type"] or "").endswith("ObjectProperty") else "datatype"
                lines.append(f"- {p['label'] or p['p']} <{p['p']}> [{kind}]"
                             + (f" domain <{p['domain']}>" if p["domain"] else "")
                             + (f" range <{p['range']}>" if p["range"] else ""))
        if relevant:
            lines.append("\n### Seed terms closest to these chunks — reuse one when it has the same meaning "
                         "(the searches find the others)")
            lines += relevant
        if parents:
            lines.append("\n### General seed classes — candidate parents: what kind of thing is each new class?")
            lines += parents
        return "\n".join(lines)

    async def extract(self) -> None:
        batch_number = 0
        while not self.max_batches or batch_number < self.max_batches:
            pending = json.loads(await self._call("chunk_list", {"status": "pending", "limit": self.batch_size}))
            if not pending:
                logger.info("No pending chunks left.")
                return
            batch_number += 1
            ids = [c["id"] for c in pending]
            chunks = json.loads(await self._call("chunk_read_batch", {"chunk_ids": ids}))
            chunks_text = "\n\n".join(
                f"### Chunk {c['id']} (document {c.get('doc_id', '')}, #{c.get('chunk_index', '')})\n{c.get('text', '')}"
                for c in chunks if "error" not in c
            )
            chunks = [c for c in chunks if "error" not in c]
            task = BATCH_PROMPT.format(batch_number=batch_number, digest=await self.digest(chunks), chunks=chunks_text)
            await self.loop.task(SYSTEM_PROMPT, task, f"batch {batch_number}")

            # If the batch failed (LLM error), the exception stops the run before this point:
            # its chunks stay pending and are picked up again on the next run.
            for chunk_id in ids:
                if await self._call("chunk_mark_processed", {"chunk_id": chunk_id}) != "ok":
                    raise RuntimeError(f"Could not mark chunk {chunk_id} as processed.")
            summary = json.loads(await self._call("ontology_summary"))
            logger.info(
                "Batch %d done — chunks %s/%s processed, %s classes, %s object properties, %s individuals.",
                batch_number, summary.get("chunks_processed", "?"), summary.get("chunks_total", "?"),
                summary.get("classes"), summary.get("object_properties"), summary.get("individuals"),
            )
        logger.info("Reached max_batches (%d) — remaining chunks stay pending.", self.max_batches)

    # ── Run ───────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.extract()

        if self.export_path:
            with open(self.export_path, "w", encoding="utf-8") as f:
                f.write(await self._call("ontology_export"))
            logger.info("Ontology exported to %s", self.export_path)

        summary = json.loads(await self._call("ontology_summary"))
        logger.info(
            "Finished — %s classes, %s object properties, %s datatype properties, %s individuals; "
            "chunks %s/%s processed. Next: run olaf_reasoning_agent to curate the ontology.",
            summary.get("classes"), summary.get("object_properties"), summary.get("datatype_properties"),
            summary.get("individuals"), summary.get("chunks_processed", "?"), summary.get("chunks_total", "?"),
        )


# ─── Entry point ──────────────────────────────────────────────────────────────


async def main_async(config: dict) -> None:
    olaf_url = config["olaf"]["url"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    async with sse_client(olaf_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await select_context(session, config["olaf"], create=True)
            tools_result = await session.list_tools()
            await BuildingAgent(session, config, tools_result.tools).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="olaf_building_agent — autonomous OWL/RDFS ontology construction"
    )
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    setup_logging(args.log_level or config.get("agent", {}).get("log_level", "INFO"))
    run(main_async(config))


if __name__ == "__main__":
    main()

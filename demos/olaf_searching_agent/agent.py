"""
olaf_searching_agent — LLM agent that answers natural-language questions over an OLAF ontology.

The LLM searches the ontology and its source chunks with the read-only OLAF MCP tools,
writes and runs SPARQL queries (sparql_query) when needed, and answers in natural language.
Python just: connects to OLAF, exposes the read-only tools to LiteLLM, and runs the loop.

Usage:
    python agent.py                                   # interactive session
    python agent.py --question "Which bonds fund renewable projects?"
    python agent.py --config my.toml --show-sparql
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tomllib
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_common import LoopSettings, ToolLoop, logger, run, setup_logging  # noqa: E402

from prompts import WRAP_UP_PROMPT, build_system_prompt  # noqa: E402

# Tools that only read the store. Anything else (create/update/merge/delete/load/mark…) is
# hidden from the LLM, and refused if it is called anyway.
READ_ONLY_TOOLS = {
    "chunk_list",
    "chunk_read",
    "chunk_read_batch",
    "concept_list",
    "concept_get",
    "concept_search",
    "concept_semantic_search",
    "property_get",
    "property_search",
    "property_semantic_search",
    "relation_search",
    "relation_sources",
    "seed_list",
    "seed_term_get",
    "ontology_list",
    "ontology_summary",
    "ontology_export",
    "sparql_query",
}

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
RESET_COMMANDS = {"/reset", "/new"}


# ─── Agent ────────────────────────────────────────────────────────────────────


class SearchingAgent:
    """Holds the conversation, so follow-up questions can refer to previous answers."""

    def __init__(self, session: ClientSession, config: dict, mcp_tools: list, show_sparql: bool):
        self.loop = ToolLoop(
            session, config["litellm"], mcp_tools,
            allowed=READ_ONLY_TOOLS,
            settings=LoopSettings.from_config(config.get("agent", {}), max_iterations=20, keep_recent_turns=4),
            refusal="Error: tool '{tool}' is not available — this agent has read-only access.",
            verbose=False,
            before_call=_print_sparql_call if show_sparql else None,
        )
        self.system_prompt = build_system_prompt(config["olaf"]["ontology_id"])
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

    async def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        answer = await self.loop.run(self.messages, "question")
        if answer is not None:
            return answer

        # Budget exhausted: force a final answer from what was gathered.
        logger.warning("Asking for a final answer from what was gathered.")
        self.messages.append({"role": "user", "content": WRAP_UP_PROMPT})
        response = await self.loop.complete(self.messages, tool_choice="none")
        answer = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer


def _print_sparql_call(tool_name: str, args: dict) -> None:
    if tool_name == "sparql_query":
        _print_sparql(args.get("query", ""))


def _print_sparql(query: str) -> None:
    print("┌── SPARQL " + "─" * 50, file=sys.stderr)
    for line in query.strip().splitlines():
        print(f"│ {line}", file=sys.stderr)
    print("└" + "─" * 60, file=sys.stderr)


# ─── Entry point ──────────────────────────────────────────────────────────────


async def interactive_loop(agent: SearchingAgent) -> None:
    print("OLAF searching agent — ask a question ('/reset' to start over, 'exit' to quit).")
    while True:
        try:
            # input() runs in a thread so the SSE connection keeps being served meanwhile.
            question = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            return
        if question.lower() in RESET_COMMANDS:
            agent.reset()
            print("Conversation reset.")
            continue
        try:
            print("\n" + await agent.ask(question))
        except Exception as exc:
            logger.error("Failed to answer: %s", exc)


async def main_async(config: dict, question: str | None, show_sparql: bool) -> None:
    olaf_url = config["olaf"]["url"]
    ontology_id = config["olaf"]["ontology_id"]
    logger.info("Connecting to OLAF at %s", olaf_url)

    async with sse_client(olaf_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Select the ontology for this connection (session state only, nothing is written).
            switch = await session.call_tool("ontology_switch", {"ontology_id": ontology_id})
            switch_text = switch.content[0].text if switch.content else ""
            if switch_text.startswith("Error"):
                raise SystemExit(f"Cannot use ontology '{ontology_id}': {switch_text}")
            logger.info("Active ontology: %s", ontology_id)

            # Source chunks must be read from the collection the ontology was built from.
            if collection := config["olaf"].get("collection"):
                args: dict = {"collection": collection}
                if config["olaf"].get("field_mapping"):
                    args["field_mapping"] = config["olaf"]["field_mapping"]
                switch = await session.call_tool("chunk_collection_switch", args)
                switch_text = switch.content[0].text if switch.content else ""
                if switch_text.startswith("Error"):
                    raise SystemExit(f"Cannot use collection '{collection}': {switch_text}")
                logger.info("Chunk collection: %s", collection)

            tools_result = await session.list_tools()
            read_only = [t for t in tools_result.tools if t.name in READ_ONLY_TOOLS]
            missing = READ_ONLY_TOOLS - {t.name for t in read_only}
            if "sparql_query" in missing:
                logger.warning("The OLAF server has no sparql_query tool — update the server.")
            logger.info("Loaded %d read-only OLAF tools.", len(read_only))

            agent = SearchingAgent(session, config, read_only, show_sparql)
            if question:
                print(await agent.ask(question))
            else:
                await interactive_loop(agent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="olaf_searching_agent — natural-language Q&A over an OLAF ontology"
    )
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--question", "-q", default=None, help="Ask one question and exit")
    parser.add_argument(
        "--show-sparql", action=argparse.BooleanOptionalAction, default=None,
        help="Print generated SPARQL queries (default: [agent] show_sparql in config)",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG / INFO / WARNING")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    agent_cfg = config.get("agent", {})
    setup_logging(args.log_level or agent_cfg.get("log_level", "INFO"))
    show_sparql = args.show_sparql if args.show_sparql is not None else agent_cfg.get("show_sparql", True)

    run(main_async(config, args.question, show_sparql))


if __name__ == "__main__":
    main()

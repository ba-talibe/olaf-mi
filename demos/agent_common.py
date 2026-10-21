"""
agent_common — code shared by the OLAF demo agents: the LLM call (LiteLLM, with retries), the
tool-calling loop over the OLAF MCP tools, context pruning, session setup and logging.

Each agent imports it from the parent directory:

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent_common import ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import litellm
from mcp import ClientSession

logger = logging.getLogger("agent")


# ─── LLM call ─────────────────────────────────────────────────────────────────


def endpoint_kwargs(llm_cfg: dict) -> dict:
    """Optional custom endpoint for OpenAI-compatible providers (Scaleway, vLLM, …).
    The key is read from the env var named by api_key_env, never stored in the config."""
    kwargs = {}
    if llm_cfg.get("api_base"):
        kwargs["api_base"] = llm_cfg["api_base"]
    if llm_cfg.get("api_key_env"):
        key = os.environ.get(llm_cfg["api_key_env"])
        if not key:
            raise SystemExit(f"Environment variable {llm_cfg['api_key_env']} is not set.")
        kwargs["api_key"] = key
    return kwargs


# Errors worth retrying as-is: the provider was slow or briefly unavailable.
_TRANSIENT_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def completion(llm_cfg: dict, **kwargs):
    """litellm.completion, retried on rate limits (HTTP 429) with a growing wait — quotas
    are usually per minute, so the wait goes up to a full minute — and on transient errors
    (timeout, connection, 5xx), which are retried a few times after a short pause."""
    rate_limit_retries = llm_cfg.get("rate_limit_retries", 6)
    transient_retries = llm_cfg.get("transient_retries", 2)
    rate_limited = transient = 0
    while True:
        try:
            return litellm.completion(**kwargs, **endpoint_kwargs(llm_cfg))
        except litellm.RateLimitError as exc:
            if rate_limited == rate_limit_retries:
                raise
            wait = min(15 * 2 ** rate_limited, 60)
            rate_limited += 1
            # The provider's message says which quota was hit (requests or tokens per minute…).
            logger.warning(
                "Rate limited by the LLM provider — retrying in %ds (%d/%d): %s",
                wait, rate_limited, rate_limit_retries, " ".join(str(exc).split())[:300],
            )
            time.sleep(wait)
        except _TRANSIENT_ERRORS as exc:
            if transient == transient_retries:
                raise
            transient += 1
            logger.warning(
                "LLM call failed (%s) — retrying in 10s (%d/%d): %s",
                type(exc).__name__, transient, transient_retries, " ".join(str(exc).split())[:300],
            )
            time.sleep(10)


# ─── Tools and context ────────────────────────────────────────────────────────


def mcp_tools_to_litellm(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI function-calling format used by LiteLLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def prune_history(
    messages: list[dict], keep_recent_turns: int, pruned_chars: int, max_turns: int | None = None
) -> list[dict]:
    """Copy of `messages` to send to the LLM. A turn is an assistant message with tool calls
    plus its tool results. Tool results older than the last `keep_recent_turns` turns are cut
    to a `pruned_chars` preview; with `max_turns`, turns beyond it are dropped altogether
    (oldest first, the leading system/user messages are kept). Whole turns are always kept or
    dropped together, so every tool call keeps its result, as the API requires."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
    if max_turns is not None and len(starts) > max_turns:
        head_end, first_kept = starts[0], starts[-max_turns]
        messages = messages[:head_end] + messages[first_kept:]
        starts = [s - (first_kept - head_end) for s in starts[-max_turns:]]

    keep = max(1, keep_recent_turns)
    cutoff = starts[-keep] if len(starts) > keep else 0
    pruned = []
    for i, m in enumerate(messages):
        content = m.get("content") or ""
        if i < cutoff and m["role"] == "tool" and len(content) > pruned_chars:
            m = {**m, "content": (
                content[:pruned_chars]
                + f"\n… [pruned from context: {len(content)} chars — call the tool again if you need it]"
            )}
        pruned.append(m)
    return pruned


def summarise(args: dict) -> str:
    """One-line representation of tool arguments for logging. URIs are kept whole, so that
    every change can be read (and undone) from the log; long texts are shortened."""
    parts = []
    for k, v in args.items():
        if isinstance(v, (bool, int, float)) or v is None:  # True, not 'True': the value sent is not a string
            parts.append(f"{k}={v!r}")
            continue
        s = " ".join(str(v).split())
        parts.append(f"{k}={s[:120]!r}…" if len(s) > 120 and not s.startswith("http") else f"{k}={s!r}")
    return ", ".join(parts)


def tool_text(result) -> str:
    return result.content[0].text if result.content else ""


@dataclass
class LoopSettings:
    """Limits of one tool-calling loop, read from the agent's [agent] config section."""
    max_iterations: int = 30
    max_tool_result_chars: int = 20_000
    keep_recent_turns: int = 3
    pruned_result_chars: int = 500
    max_history_turns: int | None = None

    @classmethod
    def from_config(cls, agent_cfg: dict, **defaults) -> "LoopSettings":
        base = cls(**defaults)
        return cls(**{f: agent_cfg.get(f, getattr(base, f)) for f in cls.__dataclass_fields__})


class ToolLoop:
    """The LLM ↔ OLAF tool loop: the LLM calls tools until it replies with plain text.
    Only the tools in `allowed` can be called; others are refused with `refusal`.
    `forced_args` ({tool: {arg: value}}) overrides what the LLM passes for those arguments."""

    def __init__(
        self,
        session: ClientSession,
        llm_cfg: dict,
        mcp_tools: list,
        allowed: set[str],
        settings: LoopSettings,
        refusal: str = "Error: {tool} is not available for this task.",
        verbose: bool = True,
        before_call: Callable[[str, dict], None] | None = None,
        forced_args: dict[str, dict] | None = None,
    ):
        self.session = session
        self.llm_cfg = llm_cfg
        self.allowed = allowed
        self.tools = mcp_tools_to_litellm([t for t in mcp_tools if t.name in allowed])
        self.settings = settings
        self.refusal = refusal
        self.verbose = verbose  # INFO-level progress (batch agents) vs DEBUG (interactive)
        self.before_call = before_call
        self.forced_args = forced_args or {}

    async def complete(self, messages: list[dict], tool_choice: str = "auto"):
        s = self.settings
        return await asyncio.to_thread(
            completion,
            self.llm_cfg,
            model=self.llm_cfg["model"],
            messages=prune_history(messages, s.keep_recent_turns, s.pruned_result_chars, s.max_history_turns),
            tools=self.tools,
            tool_choice=tool_choice,
            max_tokens=self.llm_cfg.get("max_tokens", 4096),
            temperature=self.llm_cfg.get("temperature", 0),
            timeout=self.llm_cfg.get("timeout", 120),
        )

    async def call_tool(self, tool_name: str, raw_args: str) -> str:
        """Tool call requested by the LLM; errors are returned to it so it can adapt."""
        try:
            tool_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_args}"
        if tool_name not in self.allowed:
            logger.warning("Refused tool: %s", tool_name)
            return self.refusal.format(tool=tool_name)

        tool_args = {**tool_args, **self.forced_args.get(tool_name, {})}
        logger.info("Tool call: %s(%s)", tool_name, summarise(tool_args))
        if self.before_call:
            self.before_call(tool_name, tool_args)
        try:
            content = tool_text(await self.session.call_tool(tool_name, tool_args))
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            return f"Error: {exc}"

        logger.debug("  → %s", content[:300])
        limit = self.settings.max_tool_result_chars
        if len(content) > limit:
            content = content[:limit] + f"\n… [truncated: {len(content)} chars in total — narrow your query]"
        return content

    async def run(self, messages: list[dict], label: str) -> str | None:
        """Loop on `messages` (extended in place) until the LLM replies without a tool call.
        Returns that reply, or None when max_iterations is reached."""
        level = logging.INFO if self.verbose else logging.DEBUG
        for iteration in range(1, self.settings.max_iterations + 1):
            logger.log(level, "[%s] iteration %d", label, iteration)
            response = await self.complete(messages)
            msg = response.choices[0].message
            if usage := getattr(response, "usage", None):
                logger.debug("Tokens — in: %s, out: %s",
                             getattr(usage, "prompt_tokens", "?"), getattr(usage, "completion_tokens", "?"))

            if not msg.tool_calls:
                text = (msg.content or "").strip()
                messages.append({"role": "assistant", "content": text})
                logger.log(level, "[%s] done: %s", label, text[:1000])
                return text
            if msg.content and msg.content.strip():  # the model's reasoning between tool calls
                logger.log(level, "[%s] %s", label, msg.content.strip()[:500])

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": await self.call_tool(tc.function.name, tc.function.arguments),
                })

        logger.warning("[%s] reached max_iterations (%d).", label, self.settings.max_iterations)
        return None

    async def task(self, system_prompt: str, task: str, label: str) -> str:
        """One task = one fresh conversation. Returns the LLM's closing summary."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
        logger.debug("[%s] task:\n%s", label, task)
        return await self.run(messages, label) or ""


# ─── Session and process setup ────────────────────────────────────────────────


async def call_or_exit(session: ClientSession, tool_name: str, args: dict) -> str:
    """Setup call: an error ends the program with its message."""
    text = tool_text(await session.call_tool(tool_name, args))
    if text.startswith("Error"):
        raise SystemExit(f"{tool_name} failed: {text}")
    return text


async def select_context(session: ClientSession, olaf_cfg: dict, create: bool = False) -> None:
    """Activate the configured ontology — created first if `create` and missing, otherwise it
    must exist — and select the chunk collection and the seeds (`seed_ids`; omitted: every seed
    in the store, empty: none). Session state only: other agents on the same server are unaffected."""
    ontology_id = olaf_cfg["ontology_id"]
    created = False
    if create:
        create_args = {"ontology_id": ontology_id, "name": olaf_cfg.get("ontology_name", ontology_id)}
        if olaf_cfg.get("base_uri"):
            create_args["base_uri"] = olaf_cfg["base_uri"]
        created = json.loads(await call_or_exit(session, "ontology_create", create_args)).get("created", False)
    await call_or_exit(session, "ontology_switch", {"ontology_id": ontology_id})
    logger.info("Active ontology: %s%s", ontology_id, " (created)" if created else "")

    if collection := olaf_cfg.get("collection"):
        args: dict = {"collection": collection}
        if olaf_cfg.get("field_mapping"):
            args["field_mapping"] = olaf_cfg["field_mapping"]
        counts = json.loads(await call_or_exit(session, "chunk_collection_switch", args))
        logger.info("Chunk collection: %s (%s/%s processed)",
                    collection, counts.get("chunks_processed", "?"), counts.get("chunks_total", "?"))
    else:
        logger.info("Chunk collection: server default ([qdrant].collection)")

    if (seed_ids := olaf_cfg.get("seed_ids")) is not None:
        await call_or_exit(session, "seed_select", {"seed_ids": seed_ids})
        logger.info("Seeds: %s", ", ".join(seed_ids) or "none")
    else:
        logger.info("Seeds: all loaded seeds")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Library output (full LLM requests, one line per HTTP request) drowns the agent's own log.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "mcp", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(coro: Awaitable) -> None:
    """asyncio.run, but a SystemExit raised inside the MCP session (wrapped by anyio in an
    exception group) exits with its message instead of a long traceback."""
    def first_exit(group: BaseExceptionGroup) -> SystemExit | None:
        for exc in group.exceptions:
            found = exc if isinstance(exc, SystemExit) else (
                first_exit(exc) if isinstance(exc, BaseExceptionGroup) else None
            )
            if found:
                return found
        return None

    try:
        asyncio.run(coro)
    except BaseExceptionGroup as group:
        if exit_exc := first_exit(group):
            raise exit_exc from None
        raise

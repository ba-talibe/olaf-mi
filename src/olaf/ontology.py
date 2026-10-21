from __future__ import annotations
import asyncio
import contextlib
import difflib
import hashlib
import os
import re
import uuid
from urllib.parse import urlparse

_PREFIXES = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

_RESTRICTION_PROP = {
    "some":      "owl:someValuesFrom",
    "all":       "owl:allValuesFrom",
    "has_value": "owl:hasValue",
    "exactly":   "owl:qualifiedCardinality",
    "min":       "owl:minQualifiedCardinality",
    "max":       "owl:maxQualifiedCardinality",
}
_UNQUALIFIED_RESTRICTION_PROP = {
    "exactly": "owl:cardinality",
    "min":     "owl:minCardinality",
    "max":     "owl:maxCardinality",
}

_CHUNK_URI_PREFIX = "urn:olaf:chunk:"
_SEED_PREFIX = "urn:olaf:seed:"
_SEED_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")

# RDF serializations Oxigraph can load into a single named graph, by short name.
RDF_FORMATS = {
    "ttl":    "text/turtle",
    "xml":    "application/rdf+xml",
    "nt":     "application/n-triples",
    "n3":     "text/n3",
    "jsonld": "application/ld+json",
}
_FORMAT_BY_EXTENSION = {
    ".ttl": "ttl", ".turtle": "ttl",
    ".owl": "xml", ".rdf": "xml", ".xml": "xml",
    ".nt": "nt", ".n3": "n3",
    ".jsonld": "jsonld", ".json": "jsonld",
}


def rdf_content_type(
    fmt: str | None = None,
    name: str | None = None,
    http_content_type: str | None = None,
    data: bytes = b"",
) -> str:
    """Media type of an RDF document: explicit format, then the server's Content-Type,
    then the file extension, then a look at the first bytes; Turtle by default."""
    if fmt:
        if fmt not in RDF_FORMATS:
            raise ValueError(f"Unknown RDF format {fmt!r} (expected one of {', '.join(RDF_FORMATS)})")
        return RDF_FORMATS[fmt]
    if http_content_type:
        media = http_content_type.split(";")[0].strip().lower()
        if media in RDF_FORMATS.values():
            return media
    if name:
        ext = os.path.splitext(urlparse(name).path)[1].lower()
        if ext in _FORMAT_BY_EXTENSION:
            return RDF_FORMATS[_FORMAT_BY_EXTENSION[ext]]
    head = data.lstrip()[:200]
    if head.startswith(b"<?xml") or head.startswith(b"<rdf:RDF"):
        return RDF_FORMATS["xml"]
    if head.startswith((b"{", b"[")):
        return RDF_FORMATS["jsonld"]
    return RDF_FORMATS["ttl"]


MAX_SEED_BYTES = 200 * 1024 * 1024


def _check_public_address(host: str, addresses: list[str]) -> None:
    import ipaddress
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(f"Seed URL host {host!r} resolves to a non-public address ({address}): "
                             "load it with the CLI (olaf load-seed) instead.")


async def _fetch_rdf(url: str, allow_private_hosts: bool) -> tuple[bytes, str | None]:
    """Download an RDF document (at most MAX_SEED_BYTES), following redirects by hand so that
    every hop is checked against private addresses unless allow_private_hosts."""
    import socket
    import httpx

    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(10):
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"seed URL must be an http or https URL (got {url!r})")
            if not allow_private_hosts:
                infos = await asyncio.get_running_loop().getaddrinfo(
                    parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
                _check_public_address(parsed.hostname, [info[4][0] for info in infos])
            async with client.stream("GET", url, headers={"Accept": _RDF_ACCEPT}) as resp:
                if resp.is_redirect:
                    url = str(resp.url.join(resp.headers["location"]))
                    continue
                resp.raise_for_status()
                if int(resp.headers.get("content-length") or 0) > MAX_SEED_BYTES:
                    raise ValueError(f"Seed too large ({resp.headers['content-length']} bytes, max {MAX_SEED_BYTES})")
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_SEED_BYTES:
                        raise ValueError(f"Seed too large (more than {MAX_SEED_BYTES} bytes)")
                    chunks.append(chunk)
                return b"".join(chunks), resp.headers.get("content-type")
    raise ValueError(f"Too many redirects for seed URL {url!r}")


_RDF_ACCEPT = (
    "text/turtle, application/rdf+xml;q=0.9, application/n-triples;q=0.8, "
    "application/ld+json;q=0.7, text/n3;q=0.6, */*;q=0.1"
)

# Vocabulary terms a seed may redeclare: not indexed as seed entities.
_BUILTIN_NAMESPACES = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2001/XMLSchema#",
)


_OWL = "http://www.w3.org/2002/07/owl#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"

# Types indexed for semantic search → (kind, property_type), in order of precedence.
_INDEXED_TYPES = {
    f"{_OWL}Class":            ("class", None),
    f"{_RDFS}Class":           ("class", None),
    f"{_OWL}NamedIndividual":  ("individual", None),
    f"{_OWL}ObjectProperty":   ("property", "object"),
    f"{_OWL}DatatypeProperty": ("property", "datatype"),
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property": ("property", None),
}
_LABEL_PREDICATES = (f"{_RDFS}label", f"{_SKOS}prefLabel")
_DEFINITION_PREDICATES = (f"{_SKOS}definition", f"{_RDFS}comment")


# Properties that link to the outside world: their object may be any URI (an entity of
# another dataset, a web page), not a term the ontology uses.
_LINK_PROPERTIES = {
    f"{_OWL}sameAs", f"{_RDFS}seeAlso", f"{_RDFS}isDefinedBy",
    f"{_SKOS}exactMatch", f"{_SKOS}closeMatch", f"{_SKOS}broadMatch", f"{_SKOS}narrowMatch", f"{_SKOS}relatedMatch",
}

# Namespaces whose terms may be used without being declared in the ontology.
_STANDARD_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2004/02/skos/core#",
    "http://www.w3.org/2001/XMLSchema#",
)


def _values(var: str, uris: list[str] | None) -> str:
    """A VALUES clause restricting ?var to `uris`, or nothing when uris is None."""
    if uris is None:
        return ""
    return f"VALUES ?{var} {{ {' '.join(f'<{_check_uri(u)}>' for u in uris)} }}"


def seed_graph(seed_id: str, strict: bool = True) -> str:
    """Graph of a seed. New seeds must have a plain id (it also names their Qdrant collection);
    strict=False accepts any id, for seeds loaded before ids were checked."""
    if strict and not _SEED_ID.fullmatch(seed_id):
        raise ValueError(f"Invalid seed id {seed_id!r}: use letters, digits, '_', '-' or '.' (max 64)")
    return _check_uri(f"{_SEED_PREFIX}{seed_id}")


def _seed_graphs(var: str, seed_ids: list[str] | None) -> str:
    """SPARQL condition: `var` is the graph of a seed — any seed, or only those of seed_ids
    (the seeds selected for the session; an empty list selects none)."""
    if seed_ids is None:
        return f'STRSTARTS(STR({var}), "{_SEED_PREFIX}")'
    if not seed_ids:
        return "false"
    return f"{var} IN ({', '.join(f'<{seed_graph(s, strict=False)}>' for s in seed_ids)})"


def check_ontology_id(ontology_id: str) -> str:
    """An ontology id names its graph (urn:olaf:{id}) and its Qdrant collection: it must not
    look like a seed's (urn:olaf:seed:{id}, {prefix}_seed_{id})."""
    if not _SEED_ID.fullmatch(ontology_id) or ontology_id.lower() == "seed" or ontology_id.lower().startswith("seed_"):
        raise ValueError(f"Invalid ontology id {ontology_id!r}: use letters, digits, '_', '-' or '.' "
                         "(max 64), not starting with 'seed_' (reserved for seeds)")
    return ontology_id


# Path segments that don't name an ontology: versions ("0.1", "v2") and generic file names.
_VERSION_SEGMENT = re.compile(r"v?\d+([._-]\d+)*", re.IGNORECASE)
_GENERIC_SEGMENTS = {"ontology", "onto", "index", "schema", "vocab", "vocabulary", "ns", "owl", "rdf", "core", "terms"}


def default_seed_id(source: str | None) -> str | None:
    """A readable seed id from a URL or file name: its last meaningful path segment
    (http://xmlns.com/foaf/0.1/ → 'foaf'), else the host's main label; None if nothing fits."""
    parsed = urlparse(source or "")
    for segment in reversed([s for s in parsed.path.split("/") if s]):
        stem = os.path.splitext(segment)[0] if not _VERSION_SEGMENT.fullmatch(segment) else segment
        if _VERSION_SEGMENT.fullmatch(stem) or stem.lower() in _GENERIC_SEGMENTS:
            continue
        if seed_id := re.sub(r"[^A-Za-z0-9_.-]", "-", stem)[:64].strip("-."):
            return seed_id
    host = (parsed.hostname or "").removeprefix("www.").split(".")[0]
    return re.sub(r"[^A-Za-z0-9_-]", "-", host)[:64].strip("-") or None


def _local_name(uri: str) -> str:
    """Readable fallback label for an entity without one: the IRI fragment or last path segment."""
    name = re.split(r"[#/:]", uri.rstrip("#/"))[-1]
    return re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", name).strip() or uri


def _slugify_class(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    return "".join(w.capitalize() for w in words) or "Unknown"


def _slugify_prop(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", text).split()
    if not words:
        return "unknown"
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


_INVALID_IRI = re.compile(r'[<>\s"{}|\\^`]')


def _check_uri(uri: str) -> str:
    if _INVALID_IRI.search(uri):
        raise ValueError(f"Invalid URI (contains illegal characters): {uri!r}")
    return uri


_READ_QUERY_FORMS = {"SELECT", "ASK", "CONSTRUCT", "DESCRIBE"}

# One prologue item: a comment line, a PREFIX or a BASE declaration.
_PROLOGUE_ITEM = re.compile(r"\s*(?:#[^\n]*(?:\n|$)|PREFIX\s+[^\s:]*:\s*<[^>]*>|BASE\s+<[^>]*>)", re.IGNORECASE)
_FIRST_KEYWORD = re.compile(r"\s*([A-Za-z]+)")


def _query_form(sparql: str) -> str | None:
    """Return the query form keyword (SELECT, ASK, INSERT…) after skipping the prologue."""
    pos = 0
    while (m := _PROLOGUE_ITEM.match(sparql, pos)) and m.end() > pos:
        pos = m.end()
    m = _FIRST_KEYWORD.match(sparql, pos)
    return m.group(1).upper() if m else None


def _fmt_http(binding: dict | None) -> str | None:
    if binding is None:
        return None
    return binding["value"]


def _chunk_uri(chunk_id: str) -> str:
    return f"{_CHUNK_URI_PREFIX}{chunk_id}"


def _chunk_id_from_uri(uri: str) -> str:
    return uri[len(_CHUNK_URI_PREFIX):] if uri.startswith(_CHUNK_URI_PREFIX) else uri


def _split_chunks(concatenated: str | None) -> list[str]:
    if not concatenated:
        return []
    return [_chunk_id_from_uri(c) for c in concatenated.split("|") if c]


# Marks, on a triple's rdf:Statement node, that the triple was materialized by the reasoner
# (ontology_infer) rather than extracted or asserted. Such triples are left out of
# ontology_check, and replaced on each materialization.
INFERRED_BY = "urn:olaf:inferredBy"

# Triples that are only there because the reasoner inferred them: marked, and not (also)
# extracted from a chunk — by any of the statement nodes describing the triple, as a merge
# can leave several. `?st`, `?s`, `?p`, `?o` are bound by the pattern.
_INFERRED_ONLY = f"""
    ?st <{INFERRED_BY}> ?_by ; rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o .
    FILTER NOT EXISTS {{
        ?_src rdf:subject ?s ; rdf:predicate ?p ; rdf:object ?o ; <urn:olaf:extractedFrom> ?_chunk
    }}
"""


# Terms of the standard vocabularies. A term of their namespaces outside these lists is a typo
# (e.g. rdfs:subClassof) that no tool or reasoner understands: refused on write, reported by the
# integrity checks.
_VOCABULARIES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": {
        "type", "Property", "Statement", "subject", "predicate", "object", "value", "List", "first",
        "rest", "nil", "Bag", "Seq", "Alt", "langString", "XMLLiteral", "HTML", "JSON", "PlainLiteral",
    },
    "http://www.w3.org/2000/01/rdf-schema#": {
        "Resource", "Class", "Literal", "Datatype", "Container", "ContainerMembershipProperty",
        "subClassOf", "subPropertyOf", "domain", "range", "label", "comment", "seeAlso",
        "isDefinedBy", "member",
        # Not an RDFS term (the standard one is skos:altLabel), but OLAF's own convention for
        # aliases since the start (concept_create, concept_merge…): accepted as such.
        "altLabel",
    },
    "http://www.w3.org/2002/07/owl#": {
        "AllDifferent", "AllDisjointClasses", "AllDisjointProperties", "allValuesFrom",
        "annotatedProperty", "annotatedSource", "annotatedTarget", "Annotation", "AnnotationProperty",
        "assertionProperty", "AsymmetricProperty", "Axiom", "backwardCompatibleWith",
        "bottomDataProperty", "bottomObjectProperty", "cardinality", "Class", "complementOf",
        "DataRange", "datatypeComplementOf", "DatatypeProperty", "deprecated", "DeprecatedClass",
        "DeprecatedProperty", "differentFrom", "disjointUnionOf", "disjointWith", "distinctMembers",
        "equivalentClass", "equivalentProperty", "FunctionalProperty", "hasKey", "hasSelf",
        "hasValue", "imports", "incompatibleWith", "intersectionOf", "InverseFunctionalProperty",
        "inverseOf", "IrreflexiveProperty", "maxCardinality", "maxQualifiedCardinality", "members",
        "minCardinality", "minQualifiedCardinality", "NamedIndividual", "NegativePropertyAssertion",
        "Nothing", "ObjectProperty", "onClass", "onDataRange", "onDatatype", "oneOf",
        "onProperties", "onProperty", "Ontology", "OntologyProperty", "priorVersion",
        "propertyChainAxiom", "propertyDisjointWith", "qualifiedCardinality", "ReflexiveProperty",
        "Restriction", "sameAs", "someValuesFrom", "sourceIndividual", "SymmetricProperty",
        "targetIndividual", "targetValue", "Thing", "topDataProperty", "topObjectProperty",
        "TransitiveProperty", "unionOf", "versionInfo", "versionIRI", "withRestrictions",
    },
    "http://www.w3.org/2004/02/skos/core#": {
        "Concept", "ConceptScheme", "Collection", "OrderedCollection", "prefLabel", "altLabel",
        "hiddenLabel", "definition", "note", "scopeNote", "example", "historyNote", "editorialNote",
        "changeNote", "notation", "broader", "narrower", "related", "broaderTransitive",
        "narrowerTransitive", "semanticRelation", "inScheme", "hasTopConcept", "topConceptOf",
        "member", "memberList", "mappingRelation", "exactMatch", "closeMatch", "broadMatch",
        "narrowMatch", "relatedMatch",
    },
}


def _unknown_term(uri: str) -> tuple[str, str | None] | None:
    """(term, closest known term) if `uri` is in a standard vocabulary's namespace but is not
    one of its terms, else None."""
    for ns, terms in _VOCABULARIES.items():
        if uri.startswith(ns):
            local = uri[len(ns):]
            if local in terms:
                return None
            close = difflib.get_close_matches(local, terms, n=1, cutoff=0.6) or [
                t for t in terms if t.lower() == local.lower()
            ]
            return uri, (ns + close[0]) if close else None
    return None


def _statements_of(subject_uri: str, property_uri: str, obj_term: str, var: str = "?st") -> str:
    """Pattern matching every rdf:Statement node that describes a triple. Provenance is
    looked up by content, not by the node's URI: the URI is a hash of the triple as it was
    first recorded, which concept_merge does not rename when it re-points the triple."""
    return f"{var} rdf:subject <{subject_uri}> ; rdf:predicate <{property_uri}> ; rdf:object {obj_term} ."


def _stmt_uri(subject_uri: str, property_uri: str, obj_term: str) -> str:
    """Deterministic URI identifying a (subject, property, object) triple, so its
    provenance (extractedFrom) can be looked up or extended without needing to
    round-trip a blank node identity across separate HTTP requests."""
    key = f"{subject_uri}\n{property_uri}\n{obj_term}"
    return f"urn:olaf:stmt:{hashlib.sha256(key.encode()).hexdigest()[:16]}"


# ── HTTP executor ──────────────────────────────────────────────────────────────

class _HttpExecutor:
    def __init__(self, url: str):
        import httpx
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30)

    async def execute_select(self, sparql: str, variables: list[str]) -> list[dict[str, str | None]]:
        resp = await self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        return [{v: _fmt_http(b.get(v)) for v in variables} for b in bindings]

    async def execute_ask(self, sparql: str) -> bool:
        resp = await self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
        )
        resp.raise_for_status()
        return bool(resp.json().get("boolean", False))

    async def execute_raw_query(self, sparql: str, accept: str):
        """Run an arbitrary read query and return the raw HTTP response.
        Oxigraph's error message is surfaced (instead of a bare status code) so a
        caller writing the query — typically an LLM — can fix it and retry."""
        resp = await self._client.post(
            f"{self._url}/query",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-query", "Accept": accept},
        )
        if resp.is_error:
            raise ValueError(f"SPARQL error ({resp.status_code}): {resp.text.strip()}")
        return resp

    async def execute_update(self, sparql: str) -> None:
        resp = await self._client.post(
            f"{self._url}/update",
            content=sparql.encode(),
            headers={"Content-Type": "application/sparql-update"},
        )
        resp.raise_for_status()

    async def load_graph(self, data: bytes, graph_uri: str, content_type: str = "text/turtle") -> None:
        """Replace the graph's content. Oxigraph's parse error is surfaced as is."""
        resp = await self._client.put(
            f"{self._url}/store",
            params={"graph": graph_uri},
            content=data,
            headers={"Content-Type": content_type},
        )
        if resp.is_error:
            raise ValueError(f"Oxigraph could not load the graph ({resp.status_code}): {resp.text.strip()}")

    async def dump_graph(self, graph_uri: str) -> bytes:
        resp = await self._client.get(
            f"{self._url}/store",
            params={"graph": graph_uri},
            headers={"Accept": "text/turtle"},
        )
        resp.raise_for_status()
        return resp.content


# ── OntologyStore ──────────────────────────────────────────────────────────────

class OntologyStore:
    """
    Stateless RDF store — all methods accept ontology_id explicitly.
    A single instance can safely serve multiple concurrent sessions.
    Per-session active ontology is tracked in server.py's session dict.
    """

    def __init__(self, url: str):
        self._ex = _HttpExecutor(url)
        self._meta: dict[str, tuple[str, str]] = {}
        # Outcome of the last ontology_check per ontology, since the server started.
        self._last_check: dict[str, dict] = {}

    def record_check(self, ontology_id: str, consistent: bool | None, issues: int) -> None:
        from datetime import datetime, timezone
        self._last_check[ontology_id] = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "consistent": consistent,
            "issues": issues,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _graph(ontology_id: str) -> str:
        graph = f"urn:olaf:{ontology_id}"
        _check_uri(graph)
        return graph

    async def _resolve_base(self, ontology_id: str) -> str:
        if ontology_id not in self._meta:
            rows = await self._ex.execute_select(f"""
            {_PREFIXES}
            SELECT ?base ?name WHERE {{
                GRAPH <urn:olaf:{ontology_id}> {{
                    ?base a owl:Ontology .
                    OPTIONAL {{ ?base rdfs:label ?name }}
                }}
            }}
            """, ["base", "name"])
            if not rows or not rows[0]["base"]:
                raise ValueError(
                    f"Ontology '{ontology_id}' not found. "
                    "Call ontology_create first, or check the configured ontology_id."
                )
            self._meta[ontology_id] = (rows[0]["base"].rstrip("#/"), rows[0]["name"] or "")
        return self._meta[ontology_id][0]

    async def _uri(self, local: str, ontology_id: str) -> str:
        return f"{await self._resolve_base(ontology_id)}#{local}"

    async def _add_source_chunk(self, graph: str, uri: str, chunk_id: str) -> None:
        """Record that `uri` was (also) extracted from `chunk_id`. Adding the same
        chunk twice is a no-op — RDF triples are a set — so callers don't need to
        check for existence first."""
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{ <{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(chunk_id)}> }}
        }}
        """)

    # ── Startup ────────────────────────────────────────────────────────────

    async def bootstrap(self, ontology_id: str, base_uri: str, name: str = "Ontology") -> None:
        """Ensure the default ontology declaration exists. Call once at server startup."""
        base = _check_uri(base_uri.rstrip("#/"))
        graph = self._graph(ontology_id)
        if not await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{base}> a owl:Ontology }} }}"
        ):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{
                GRAPH <{graph}> {{
                    <{base}> a owl:Ontology ;
                        rdfs:label "{_esc(name)}" .
                }}
            }}
            """)
        self._meta[ontology_id] = (base, name)

    # ── Multi-ontology management ──────────────────────────────────────────

    async def ontology_list(self, current_id: str) -> list[dict]:
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?g ?base ?name (COUNT(DISTINCT ?cls) AS ?classes) WHERE {{
            GRAPH ?g {{
                ?base a owl:Ontology .
                OPTIONAL {{ ?base rdfs:label ?name }}
                OPTIONAL {{ ?cls a owl:Class }}
            }}
            FILTER(STRSTARTS(STR(?g), "urn:olaf:"))
            FILTER(!CONTAINS(STR(?g), ":seed:"))
        }}
        GROUP BY ?g ?base ?name
        """, ["g", "base", "name", "classes"])
        return [
            {
                "id": (row["g"] or "")[len("urn:olaf:"):],
                "name": row["name"] or "",
                "base_uri": (row["base"] or "") + "#",
                "classes": int(row["classes"] or 0),
                "active": (row["g"] or "") == self._graph(current_id),
            }
            for row in rows
        ]

    async def ontology_create(self, ontology_id: str, name: str, base_uri: str) -> dict:
        check_ontology_id(ontology_id)
        base = _check_uri(base_uri.rstrip("#/"))
        graph = self._graph(ontology_id)

        if await self._ex.execute_ask(f"ASK {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}"):
            return {"id": ontology_id, "created": False}

        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                <{base}> a owl:Ontology ;
                    rdfs:label "{_esc(name)}" .
            }}
        }}
        """)
        self._meta[ontology_id] = (base, name)
        return {"id": ontology_id, "created": True, "base_uri": base + "#"}

    async def ontology_switch(self, ontology_id: str) -> dict:
        graph = self._graph(ontology_id)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?base ?name WHERE {{
            GRAPH <{graph}> {{
                ?base a owl:Ontology .
                OPTIONAL {{ ?base rdfs:label ?name }}
            }}
        }}
        """, ["base", "name"])
        if not rows:
            raise ValueError(f"Ontology '{ontology_id}' not found. Use ontology_create first.")

        base = rows[0]["base"].rstrip("#/") if rows[0]["base"] else await self._resolve_base(ontology_id)
        name = rows[0]["name"] or ""
        self._meta[ontology_id] = (base, name)
        return {"active_id": ontology_id, "name": name, "base_uri": base + "#"}

    async def require_ontology(self, ontology_id: str) -> None:
        if not await self._ex.execute_ask(f"ASK {{ GRAPH <{self._graph(ontology_id)}> {{ ?s ?p ?o }} }}"):
            existing = ", ".join(sorted(o["id"] for o in await self.ontology_list(""))) or "none"
            raise ValueError(f"Ontology '{ontology_id}' not found (existing: {existing}).")

    async def require_seed(self, seed_id: str) -> None:
        if not await self._ex.execute_ask(f"ASK {{ GRAPH <{seed_graph(seed_id, strict=False)}> {{ ?s ?p ?o }} }}"):
            existing = ", ".join(sorted(s["id"] for s in await self.seed_list())) or "none"
            raise ValueError(f"Seed '{seed_id}' not found (existing: {existing}).")

    async def drop_ontology(self, ontology_id: str) -> dict:
        """Drop an ontology graph. Seeds are independent and unaffected. Intended for CLI use only."""
        await self.require_ontology(ontology_id)
        graph = self._graph(ontology_id)
        await self._ex.execute_update(f"DROP SILENT GRAPH <{graph}>")
        self._meta.pop(ontology_id, None)
        return {"dropped": ontology_id}

    async def seed_list(self) -> list[dict]:
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?g (COUNT(*) AS ?triples) WHERE {{
            GRAPH ?g {{ ?s ?p ?o }}
            FILTER(STRSTARTS(STR(?g), "urn:olaf:seed:"))
        }}
        GROUP BY ?g
        """, ["g", "triples"])
        return [
            {
                "id": (row["g"] or "")[len("urn:olaf:seed:"):],
                "graph_uri": row["g"] or "",
                "triples": int(row["triples"] or 0),
            }
            for row in rows
        ]

    async def seed_drop(self, seed_id: str) -> dict:
        """Drop a global seed graph. Intended for CLI use only."""
        await self.require_seed(seed_id)
        await self._ex.execute_update(f"DROP SILENT GRAPH <{seed_graph(seed_id, strict=False)}>")
        return {"dropped": seed_id}

    # ── Concepts (owl:Class) ───────────────────────────────────────────────

    async def concept_create(
        self,
        ontology_id: str,
        label: str,
        definition: str,
        parent_uri: str | None = None,
        aliases: list[str] | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_class(label), ontology_id)
        # Guard against any pre-existing entity (class or individual) at this URI.
        if await self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}"):
            if source_chunk_id is not None:
                await self._add_source_chunk(graph, uri, source_chunk_id)
            return {"uri": uri, "created": False}

        if parent_uri:
            await self._require_existing(graph, await self._resolve_base(ontology_id), _check_uri(parent_uri))
        lines = [
            f"<{uri}> a owl:Class",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
            f'<{uri}> skos:definition "{_esc(definition)}"@{lang}',
        ]
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subClassOf <{parent_uri}>")
        for alias in (aliases or []):
            lines.append(f'<{uri}> rdfs:altLabel "{_esc(alias)}"@{lang}')
        if source_chunk_id is not None:
            lines.append(f"<{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(source_chunk_id)}>")

        joined_lines = " .\n                ".join(lines)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {joined_lines} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    async def concept_get(self, ontology_id: str, uri: str) -> dict | None:
        _check_uri(uri)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._graph(ontology_id)}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None

        result: dict = {"uri": uri, "triples": [], "source_chunk_ids": []}
        for row in rows:
            result["triples"].append({"predicate": row["p"] or "", "object": row["o"] or ""})

        for t in result["triples"]:
            if t["predicate"] == "http://www.w3.org/2000/01/rdf-schema#label" and "label" not in result:
                result["label"] = t["object"]
            if t["predicate"] == "http://www.w3.org/2004/02/skos/core#definition" and "definition" not in result:
                result["definition"] = t["object"]
            if t["predicate"] == "urn:olaf:extractedFrom":
                result["source_chunk_ids"].append(_chunk_id_from_uri(t["object"]))
        if restrictions := await self.restrictions_of(ontology_id, uri):
            result["restrictions"] = restrictions

        return result

    async def concept_search_sparql(
        self, ontology_id: str, query_str: str, top_k: int = 10, include_seeds: bool = False,
        seed_ids: list[str] | None = None,
    ) -> list[dict]:
        """Classes whose label contains `query_str`. With include_seeds, also the classes of the
        seeds (all, or those of seed_ids) — owl:Class or rdfs:Class, label rdfs:label or
        skos:prefLabel — each tagged with its seed id."""
        q = _esc(query_str.lower())
        graph = self._graph(ontology_id)
        if include_seeds:
            values = ""
            scope = f"FILTER(?g = <{graph}> || {_seed_graphs('?g', seed_ids)})"
            pattern = "?c a ?t ; rdfs:label|skos:prefLabel ?l . FILTER(?t IN (owl:Class, rdfs:Class))"
            definition = "skos:definition|rdfs:comment"
        else:
            values = f"VALUES ?g {{ <{graph}> }}"
            scope = ""
            pattern = "?c a owl:Class ; rdfs:label ?l ."
            definition = "skos:definition"
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c (MIN(IF(?g = <{graph}>, "", STR(?g))) AS ?src) (SAMPLE(?l) AS ?label) (SAMPLE(?d) AS ?definition)
               (SAMPLE(?ch) AS ?chunks) WHERE {{
            {values}
            GRAPH ?g {{
                {pattern}
                OPTIONAL {{ ?c {definition} ?d }}
                FILTER(CONTAINS(LCASE(STR(?l)), "{q}"))
            }}
            {scope}
            OPTIONAL {{
                SELECT ?c (GROUP_CONCAT(DISTINCT STR(?chunk); separator="|") AS ?ch) WHERE {{
                    GRAPH <{graph}> {{ ?c <urn:olaf:extractedFrom> ?chunk }}
                }}
                GROUP BY ?c
            }}
        }}
        GROUP BY ?c
        LIMIT {top_k}
        """, ["c", "src", "label", "definition", "chunks"])
        results = []
        for row in rows:
            result = {
                "uri": row["c"] or "",
                "label": row["label"] or "",
                "definition": row["definition"] or "",
                "source_chunk_ids": _split_chunks(row["chunks"]),
                "match_type": "sparql",
            }
            if include_seeds:
                src = row["src"] or ""
                result["seed"] = src[len(_SEED_PREFIX):] if src.startswith(_SEED_PREFIX) else None
            results.append(result)
        return results

    async def concept_update(
        self,
        ontology_id: str,
        uri: str,
        label: str | None = None,
        definition: str | None = None,
        aliases_add: list[str] | None = None,
        aliases_remove: list[str] | None = None,
        lang: str = "en",
    ) -> None:
        _check_uri(uri)
        self._require_local(await self._resolve_base(ontology_id), uri, "concept_update")
        graph = self._graph(ontology_id)
        if not await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a ?t FILTER(?t IN (owl:Class, owl:NamedIndividual)) }} }}"
        ):
            raise ValueError(f"No class or individual {uri!r} in this ontology: create it first (concept_create / "
                             "individual_create), or find its exact URI with concept_search.")
        if label is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:label ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:label ?o }} }}
            """)
            if label:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:label "{_esc(label)}"@{lang} }} }}
                """)
        if definition is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> skos:definition ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> skos:definition ?o }} }}
            """)
            if definition:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> skos:definition "{_esc(definition)}"@{lang} }} }}
                """)
        for alias in (aliases_add or []):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)
        for alias in (aliases_remove or []):
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:altLabel "{_esc(alias)}"@{lang} }} }}
            """)

    async def concept_merge(self, ontology_id: str, keep_uri: str, merge_uri: str) -> None:
        """Merge merge_uri into keep_uri. When keep_uri is a seed term (a local duplicate of it
        is being replaced by the reused term), only the references to merge_uri and its source
        chunks are carried over: its labels, definition and axioms would redefine the seed term."""
        _check_uri(keep_uri)
        _check_uri(merge_uri)
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        self._require_local(base, merge_uri, "concept_merge (as merge_uri)")
        await self._require_existing(graph, base, keep_uri)
        # Inferred marks are keyed by triple: re-pointing the triples would leave them stale.
        # A merge changes what can be inferred anyway — ontology_infer computes them again.
        await self.inferences_clear(ontology_id)
        if not keep_uri.startswith(base):
            await self._ex.execute_update(f"""
            INSERT {{ GRAPH <{graph}> {{ <{keep_uri}> <urn:olaf:extractedFrom> ?c }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{merge_uri}> <urn:olaf:extractedFrom> ?c }} }}
            """)
            # merge_uri's own statements are dropped, not carried over: so are their provenance
            # records (re-pointed, they would describe triples that don't exist) and its restrictions.
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }}
            WHERE  {{ GRAPH <{graph}> {{ ?st rdf:subject <{merge_uri}> ; a rdf:Statement ; ?sp ?so }} }}
            """)
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{merge_uri}> rdfs:subClassOf ?r . ?r ?rp ?ro }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{merge_uri}> rdfs:subClassOf ?r . FILTER(isBlank(?r)) ?r ?rp ?ro }} }}
            """)
        else:
            await self._merge_definition(graph, keep_uri, merge_uri)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?s ?p <{merge_uri}> }} }}
        INSERT {{ GRAPH <{graph}> {{ ?s ?p <{keep_uri}> }} }}
        WHERE  {{ GRAPH <{graph}> {{ ?s ?p <{merge_uri}> }} }}
        """)
        # Uses of merge_uri as a property (merging duplicate properties).
        await self._ex.execute_update(f"""
        DELETE {{ GRAPH <{graph}> {{ ?s <{merge_uri}> ?o }} }}
        INSERT {{ GRAPH <{graph}> {{ ?s <{keep_uri}> ?o }} }}
        WHERE  {{ GRAPH <{graph}> {{ ?s <{merge_uri}> ?o }} }}
        """)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ <{merge_uri}> ?p ?o }} }}
        WHERE  {{ GRAPH <{graph}> {{ <{merge_uri}> ?p ?o }} }}
        """)
        # A relation between the two merged concepts (e.g. merge_uri subClassOf keep_uri)
        # has become a self-loop on keep_uri: drop it, it would read as a subclass cycle
        # or, for an object property, punning that types keep_uri against its range.
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ <{keep_uri}> ?p <{keep_uri}> }} }}
        WHERE  {{ GRAPH <{graph}> {{ <{keep_uri}> ?p <{keep_uri}> }} }}
        """)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }}
        WHERE  {{ GRAPH <{graph}> {{
            ?st rdf:subject <{keep_uri}> ; rdf:object <{keep_uri}> ; ?sp ?so .
        }} }}
        """)

    async def _merge_definition(self, graph: str, keep_uri: str, merge_uri: str) -> None:
        """Copy merge_uri's own triples onto keep_uri (concept_merge between local entities)."""
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT {{
            GRAPH <{graph}> {{ <{keep_uri}> ?p ?o }}
        }}
        WHERE {{
            GRAPH <{graph}> {{
                <{merge_uri}> ?p ?o .
                FILTER(?p NOT IN (rdfs:label, skos:definition))
                FILTER NOT EXISTS {{ GRAPH <{graph}> {{ <{keep_uri}> ?p ?o }} }}
            }}
        }}
        """)
        # The kept concept keeps a single label and definition: the merged one's labels become
        # alternative labels, and its definition is taken only if the kept one has none.
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT {{ GRAPH <{graph}> {{ <{keep_uri}> rdfs:altLabel ?l }} }}
        WHERE  {{ GRAPH <{graph}> {{
            <{merge_uri}> rdfs:label ?l .
            FILTER NOT EXISTS {{ <{keep_uri}> rdfs:label ?kl . FILTER(STR(?kl) = STR(?l)) }}
        }} }}
        """)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT {{ GRAPH <{graph}> {{ <{keep_uri}> skos:definition ?d }} }}
        WHERE  {{ GRAPH <{graph}> {{
            <{merge_uri}> skos:definition ?d .
            FILTER NOT EXISTS {{ <{keep_uri}> skos:definition ?kd }}
        }} }}
        """)

    async def _count(self, graph: str) -> int:
        rows = await self._ex.execute_select(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}", ["n"]
        )
        return int(rows[0]["n"] or 0) if rows else 0

    async def entity_delete(self, ontology_id: str, uri: str) -> dict:
        """Delete a class, individual or property of the ontology: every triple it appears
        in (as subject, property or object), the provenance records of those triples, its
        own restrictions and the restrictions that point to it. For a reused seed term, this
        removes its declaration and its uses from the ontology; the seed is unchanged."""
        _check_uri(uri)
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        if not uri.startswith(base) and not await self._ex.execute_ask(
            f"ASK {{ GRAPH <{graph}> {{ <{uri}> a ?t }} }}"
        ):
            raise ValueError(f"{uri} is not in this ontology's namespace ({base}#) nor a seed term it reuses: "
                             "only the ontology's own entities can be deleted.")
        if not await self._ex.execute_ask(f"ASK {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}"):
            raise ValueError(f"Entity not found: {uri}")
        before = await self._count(graph)
        # Restrictions (blank nodes) the entity owns or that point to it, whole.
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?c rdfs:subClassOf ?r . ?r ?rp ?ro }} }}
        WHERE  {{ GRAPH <{graph}> {{
            ?c rdfs:subClassOf ?r . FILTER(isBlank(?r))
            {{ FILTER(?c = <{uri}>) }} UNION {{ ?r ?x <{uri}> }}
            ?r ?rp ?ro .
        }} }}
        """)
        for update in (
            f"DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }} WHERE {{ GRAPH <{graph}> {{ "
            f"?st rdf:subject|rdf:predicate|rdf:object <{uri}> . ?st a rdf:Statement ; ?sp ?so }} }}",
            f"DELETE {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }} WHERE {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}",
            f"DELETE {{ GRAPH <{graph}> {{ ?s <{uri}> ?o }} }} WHERE {{ GRAPH <{graph}> {{ ?s <{uri}> ?o }} }}",
            f"DELETE {{ GRAPH <{graph}> {{ ?s ?p <{uri}> }} }} WHERE {{ GRAPH <{graph}> {{ ?s ?p <{uri}> }} }}",
        ):
            await self._ex.execute_update(f"{_PREFIXES}\n{update}")
        return {"deleted": uri, "triples_removed": before - await self._count(graph)}

    @staticmethod
    def _restriction_pattern(class_uri: str, property_uri: str, restriction_type: str | None, value: str | None,
                             is_literal_value: bool) -> str:
        _check_uri(class_uri)
        _check_uri(property_uri)
        lines = [f"<{class_uri}> rdfs:subClassOf ?r . ?r a owl:Restriction ; owl:onProperty <{property_uri}> ."]
        if restriction_type:
            if restriction_type not in _RESTRICTION_PROP:
                raise ValueError(f"Unknown restriction_type '{restriction_type}'. Valid: {list(_RESTRICTION_PROP)}")
            preds = [_RESTRICTION_PROP[restriction_type]]
            if restriction_type in _UNQUALIFIED_RESTRICTION_PROP:
                preds.append(_UNQUALIFIED_RESTRICTION_PROP[restriction_type])
            lines.append(f"?r ?_tp ?_tv . FILTER(?_tp IN ({', '.join(preds)}))")
        if value:
            obj = f'"{_esc(value)}"' if is_literal_value else f"<{_check_uri(value)}>"
            if is_literal_value:
                lines.append(f"?r owl:hasValue ?_v . FILTER(STR(?_v) = {obj})")
            else:
                lines.append(f"?r ?_vp {obj} . FILTER(?_vp IN (owl:someValuesFrom, owl:allValuesFrom, owl:hasValue, owl:onClass))")
        return "\n            ".join(lines)

    async def restriction_delete(
        self,
        ontology_id: str,
        class_uri: str,
        property_uri: str,
        restriction_type: str | None = None,
        value: str | None = None,
        is_literal_value: bool = False,
    ) -> dict:
        """Delete the restrictions of a class on a property — all of them, or those of a
        given type and/or value. Returns how many were deleted."""
        graph = self._graph(ontology_id)
        pattern = self._restriction_pattern(class_uri, property_uri, restriction_type, value, is_literal_value)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT (COUNT(DISTINCT ?r) AS ?n) WHERE {{ GRAPH <{graph}> {{ {pattern} }} }}
        """, ["n"])
        count = int(rows[0]["n"] or 0) if rows else 0
        if count:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{class_uri}> rdfs:subClassOf ?r . ?r ?rp ?ro }} }}
            WHERE  {{ GRAPH <{graph}> {{ {pattern} ?r ?rp ?ro . }} }}
            """)
        return {"removed": count}

    async def restrictions_of(self, ontology_id: str, class_uri: str) -> list[dict]:
        """The OWL restrictions on a class, in restriction_add's / restriction_delete's terms."""
        names = {f"http://www.w3.org/2002/07/owl#{v.split(':')[1]}": k for k, v in _RESTRICTION_PROP.items()}
        names.update({f"http://www.w3.org/2002/07/owl#{v.split(':')[1]}": k for k, v in _UNQUALIFIED_RESTRICTION_PROP.items()})
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?r ?prop ?kind ?val ?cls WHERE {{ GRAPH <{self._graph(ontology_id)}> {{
            <{class_uri}> rdfs:subClassOf ?r . ?r a owl:Restriction ; owl:onProperty ?prop ; ?kind ?val .
            FILTER(?kind IN ({", ".join(f"<{k}>" for k in names)}))
            OPTIONAL {{ ?r owl:onClass ?cls }}
        }} }}
        """, ["r", "prop", "kind", "val", "cls"])
        out = []
        for r in rows:
            kind = names.get(r["kind"] or "", "")
            if kind in ("exactly", "min", "max"):
                out.append({"property": r["prop"], "restriction_type": kind, "cardinality": int(r["val"] or 0),
                            "value": r["cls"] or ""})
            else:
                out.append({"property": r["prop"], "restriction_type": kind, "value": r["val"]})
        return out

    async def concept_list(self, ontology_id: str, root_only: bool = False, limit: int = 100) -> list[dict]:
        root_filter = ""
        if root_only:
            root_filter = """
                FILTER NOT EXISTS {
                    ?c rdfs:subClassOf ?anyParent .
                    FILTER(!isBlank(?anyParent))
                    FILTER(?anyParent != owl:Thing)
                }"""

        graph = self._graph(ontology_id)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c (SAMPLE(?lbl) AS ?label) (SAMPLE(?def) AS ?definition) (SAMPLE(?par) AS ?parent) (SAMPLE(?chunks) AS ?chunksSample) WHERE {{
            GRAPH <{graph}> {{
                ?c a owl:Class .
                OPTIONAL {{ ?c rdfs:label ?lbl }}
                OPTIONAL {{ ?c skos:definition ?def }}
                OPTIONAL {{
                    ?c rdfs:subClassOf ?par .
                    FILTER(!isBlank(?par))
                    FILTER(?par != owl:Thing)
                }}
                {root_filter}
            }}
            OPTIONAL {{
                SELECT ?c (GROUP_CONCAT(DISTINCT STR(?chunk); separator="|") AS ?chunks) WHERE {{
                    GRAPH <{graph}> {{ ?c <urn:olaf:extractedFrom> ?chunk }}
                }}
                GROUP BY ?c
            }}
        }}
        GROUP BY ?c
        LIMIT {limit}
        """, ["c", "label", "definition", "parent", "chunksSample"])

        return [
            {
                "uri": row["c"] or "",
                "label": row["label"] or "",
                "definition": row["definition"] or "",
                "parent_uri": row["parent"],
                "source_chunk_ids": _split_chunks(row["chunksSample"]),
            }
            for row in rows
        ]

    # ── Individuals (owl:NamedIndividual) ─────────────────────────────────

    async def individual_create(
        self,
        ontology_id: str,
        label: str,
        class_uri: str,
        definition: str | None = None,
        aliases: list[str] | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        _check_uri(class_uri)
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_class(label), ontology_id)

        # Guard against any pre-existing entity (class or individual) at this URI.
        if await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> ?p ?o }} }}"
        ):
            if source_chunk_id is not None:
                await self._add_source_chunk(graph, uri, source_chunk_id)
            return {"uri": uri, "created": False}
        await self._require_existing(graph, await self._resolve_base(ontology_id), class_uri)

        lines = [
            f"<{uri}> a owl:NamedIndividual",
            f"<{uri}> a <{class_uri}>",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
        ]
        if definition:
            lines.append(f'<{uri}> skos:definition "{_esc(definition)}"@{lang}')
        for alias in (aliases or []):
            lines.append(f'<{uri}> rdfs:altLabel "{_esc(alias)}"@{lang}')
        if source_chunk_id is not None:
            lines.append(f"<{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(source_chunk_id)}>")

        joined_lines = " .\n                ".join(lines)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {joined_lines} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    # ── Properties ────────────────────────────────────────────────────────

    async def property_create(
        self,
        ontology_id: str,
        label: str,
        prop_type: str,
        domain_uri: str | None = None,
        range_uri: str | None = None,
        parent_uri: str | None = None,
        lang: str = "en",
        source_chunk_id: str | None = None,
    ) -> dict:
        graph = self._graph(ontology_id)
        uri = await self._uri(_slugify_prop(label), ontology_id)
        owl_type = "owl:ObjectProperty" if prop_type == "object" else "owl:DatatypeProperty"

        if await self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a {owl_type} }} }}"):
            if source_chunk_id is not None:
                await self._add_source_chunk(graph, uri, source_chunk_id)
            return {"uri": uri, "created": False}

        base = await self._resolve_base(ontology_id)
        # A datatype property's range is a datatype (xsd:string, geo:wktLiteral…), not a term to declare.
        for ref in (domain_uri, None if prop_type == "datatype" else range_uri, parent_uri):
            if ref:
                await self._require_existing(graph, base, _check_uri(ref))
        if range_uri:
            _check_uri(range_uri)
        lines = [
            f"<{uri}> a {owl_type}",
            f'<{uri}> rdfs:label "{_esc(label)}"@{lang}',
        ]
        if domain_uri:
            lines.append(f"<{uri}> rdfs:domain <{domain_uri}>")
        if range_uri:
            lines.append(f"<{uri}> rdfs:range <{range_uri}>")
        if parent_uri:
            lines.append(f"<{uri}> rdfs:subPropertyOf <{parent_uri}>")
        if source_chunk_id is not None:
            lines.append(f"<{uri}> <urn:olaf:extractedFrom> <{_chunk_uri(source_chunk_id)}>")

        joined_lines = " .\n                ".join(lines)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                {joined_lines} .
            }}
        }}
        """)
        return {"uri": uri, "created": True}

    async def property_update(
        self,
        ontology_id: str,
        uri: str,
        domain_uri: str | None = None,
        range_uri: str | None = None,
        parent_uri: str | None = None,
    ) -> None:
        _check_uri(uri)
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        self._require_local(base, uri, "property_update")
        if not await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a ?t FILTER(?t IN (owl:ObjectProperty, owl:DatatypeProperty)) }} }}"
        ):
            raise ValueError(f"No property {uri!r} in this ontology: create it first with property_create, "
                             "or find its exact URI with property_search.")
        is_datatype = await self._ex.execute_ask(f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a owl:DatatypeProperty }} }}")
        for ref in (domain_uri, None if is_datatype else range_uri, parent_uri):
            if ref:
                await self._require_existing(graph, base, _check_uri(ref))
        if domain_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain ?o }} }}
            """)
            if domain_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:domain <{_check_uri(domain_uri)}> }} }}
                """)
        if range_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:range ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:range ?o }} }}
            """)
            if range_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:range <{_check_uri(range_uri)}> }} }}
                """)
        if parent_uri is not None:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf ?o }} }}
            WHERE  {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf ?o }} }}
            """)
            if parent_uri:
                await self._ex.execute_update(f"""
                {_PREFIXES}
                INSERT DATA {{ GRAPH <{graph}> {{ <{uri}> rdfs:subPropertyOf <{_check_uri(parent_uri)}> }} }}
                """)
                await self._unmark_inferred(graph, [(uri, "http://www.w3.org/2000/01/rdf-schema#subPropertyOf", f"<{parent_uri}>")])

    async def property_get(self, ontology_id: str, uri: str) -> dict | None:
        _check_uri(uri)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p ?o WHERE {{ GRAPH <{self._graph(ontology_id)}> {{ <{uri}> ?p ?o }} }}
        """, ["p", "o"])
        if not rows:
            return None
        triples = [{"predicate": row["p"] or "", "object": row["o"] or ""} for row in rows]
        source_chunk_ids = [
            _chunk_id_from_uri(t["object"]) for t in triples if t["predicate"] == "urn:olaf:extractedFrom"
        ]
        return {"uri": uri, "triples": triples, "source_chunk_ids": source_chunk_ids}

    async def property_search(
        self, ontology_id: str, query_str: str, top_k: int = 10, include_seeds: bool = False,
        seed_ids: list[str] | None = None,
    ) -> list[dict]:
        """Properties whose label contains `query_str`. With include_seeds, also the properties of
        the seeds (all, or those of seed_ids) — owl:ObjectProperty, owl:DatatypeProperty or
        rdf:Property, label rdfs:label or skos:prefLabel — each tagged with its seed id."""
        q = _esc(query_str.lower())
        graph = self._graph(ontology_id)
        if include_seeds:
            values = ""
            scope = f"FILTER(?g = <{graph}> || {_seed_graphs('?g', seed_ids)})"
            label = "rdfs:label|skos:prefLabel"
            rdf_property = 'UNION { ?p a rdf:Property . BIND("" AS ?t) }'
        else:
            values = f"VALUES ?g {{ <{graph}> }}"
            scope = rdf_property = ""
            label = "rdfs:label"
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?p (MIN(IF(?g = <{graph}>, "", STR(?g))) AS ?src) (SAMPLE(?l) AS ?label) (MAX(?t) AS ?type)
               (SAMPLE(?ch) AS ?chunks) WHERE {{
            {values}
            GRAPH ?g {{
                ?p {label} ?l .
                {{ ?p a owl:ObjectProperty . BIND("object" AS ?t) }}
                UNION {{ ?p a owl:DatatypeProperty . BIND("datatype" AS ?t) }}
                {rdf_property}
                FILTER(CONTAINS(LCASE(STR(?l)), "{q}"))
            }}
            {scope}
            OPTIONAL {{
                SELECT ?p (GROUP_CONCAT(DISTINCT STR(?chunk); separator="|") AS ?ch) WHERE {{
                    GRAPH <{graph}> {{ ?p <urn:olaf:extractedFrom> ?chunk }}
                }}
                GROUP BY ?p
            }}
        }}
        GROUP BY ?p
        LIMIT {top_k}
        """, ["p", "src", "label", "type", "chunks"])
        results = []
        for row in rows:
            result = {
                "uri": row["p"] or "",
                "label": row["label"] or "",
                "type": row["type"] or "",
                "source_chunk_ids": _split_chunks(row["chunks"]),
            }
            if include_seeds:
                src = row["src"] or ""
                result["seed"] = src[len(_SEED_PREFIX):] if src.startswith(_SEED_PREFIX) else None
            results.append(result)
        return results

    # ── Relations ─────────────────────────────────────────────────────────

    async def _require_existing(self, graph: str, base: str, uri: str) -> None:
        """Reject URIs the ontology doesn't know — catches hand-typed/mis-slugified URIs before
        they create a silent duplicate or a dangling reference. A URI under our own base must
        appear in the ontology; an external one must be declared in it (a seed term, through
        seed_reuse) or belong to a standard vocabulary (rdf, rdfs, owl, skos, xsd)."""
        if uri.startswith(base):
            if await self._ex.execute_ask(f"""
            ASK {{ GRAPH <{graph}> {{ {{ <{uri}> ?p ?o }} UNION {{ ?s ?p2 <{uri}> }} }} }}
            """):
                return
            raise ValueError(
                f"URI not found in this ontology: {uri!r}. If this entity doesn't exist yet, "
                "create it first with concept_create/individual_create/property_create. "
                "If it should already exist, use concept_search/concept_get/property_search "
                "to find its exact URI instead of guessing it."
            )
        if uri.startswith(_STANDARD_NAMESPACES) or await self._ex.execute_ask(
            f"ASK {{ GRAPH <{graph}> {{ <{uri}> a ?t }} }}"
        ):
            return
        if seed_id := await self._seed_of(uri):
            raise ValueError(
                f"{uri} is a term of seed '{seed_id}', not yet used in this ontology: declare it "
                "with seed_reuse first, then use its URI."
            )
        raise ValueError(
            f"Unknown external URI: {uri!r} — it is neither in this ontology, nor in a seed, nor a "
            "standard vocabulary term. Use concept_search/property_search with include_seeds=true "
            "to find an exact URI instead of guessing it."
        )

    @staticmethod
    def _require_local(base: str, uri: str, action: str) -> None:
        """Refuse to change the definition of an external (seed) term: it belongs to its source
        ontology. Its meaning is refined by local entities linked to it instead."""
        if not uri.startswith(base):
            raise ValueError(
                f"{uri} is not in this ontology's namespace ({base}#): {action} only applies to its "
                "own entities. A seed term is reused as defined by its source ontology — to refine it, "
                "create a local subclass or subproperty of it."
            )

    async def _seed_of(self, uri: str, seed_ids: list[str] | None = None) -> str | None:
        """Id of a seed (any, or one of seed_ids) that declares `uri` as a class, individual or
        property, if any."""
        rows = await self._ex.execute_select(f"""
        SELECT ?g WHERE {{
            GRAPH ?g {{ <{uri}> a ?t FILTER(?t IN ({", ".join(f"<{t}>" for t in _INDEXED_TYPES)})) }}
            FILTER({_seed_graphs("?g", seed_ids)})
        }} LIMIT 1
        """, ["g"])
        return rows[0]["g"][len(_SEED_PREFIX):] if rows and rows[0]["g"] else None

    @staticmethod
    def _format_object(object_value: str, is_literal: bool, datatype: str | None) -> str:
        if is_literal:
            return f'"{_esc(object_value)}"^^<{_check_uri(datatype)}>' if datatype else f'"{_esc(object_value)}"'
        return f"<{_check_uri(object_value)}>"

    async def _unmark_inferred(self, graph: str, triples: list[tuple[str, str, str]]) -> None:
        """A triple asserted by a tool call is no longer only inferred: drop its inferred
        mark, so that clearing the inferences does not delete it — and its whole statement
        node when the mark was all it carried (no source chunk)."""
        for subject_uri, property_uri, obj_term in triples:
            statements = _statements_of(subject_uri, property_uri, obj_term)
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }}
            WHERE  {{ GRAPH <{graph}> {{
                {statements} ?st <{INFERRED_BY}> ?by ; ?sp ?so .
                FILTER NOT EXISTS {{ ?st <urn:olaf:extractedFrom> ?chunk }}
            }} }}
            """)
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{graph}> {{ ?st <{INFERRED_BY}> ?by }} }}
            WHERE  {{ GRAPH <{graph}> {{ {statements} ?st <{INFERRED_BY}> ?by }} }}
            """)

    async def _add_relation_source(
        self, graph: str, subject_uri: str, property_uri: str, obj_term: str, chunk_id: str
    ) -> None:
        """Record that the (subject, property, object) triple was (also) extracted from
        `chunk_id`, via a deterministic reification URI. Adding the same chunk twice is a
        no-op — RDF triples are a set — so callers don't need to check for existence first."""
        stmt = _stmt_uri(subject_uri, property_uri, obj_term)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{
                <{stmt}> a rdf:Statement ;
                         rdf:subject <{subject_uri}> ;
                         rdf:predicate <{property_uri}> ;
                         rdf:object {obj_term} ;
                         <urn:olaf:extractedFrom> <{_chunk_uri(chunk_id)}> .
            }}
        }}
        """)

    async def relation_add(
        self,
        ontology_id: str,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
        source_chunk_id: str | None = None,
    ) -> None:
        _check_uri(subject_uri)
        _check_uri(property_uri)
        for term in (property_uri, None if is_literal else object_value):
            if term and (unknown := _unknown_term(term)):
                hint = f" — did you mean {unknown[1]}?" if unknown[1] else ""
                raise ValueError(f"{term} is not a term of its vocabulary{hint}")
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        await self._require_existing(graph, base, subject_uri)
        if not subject_uri.startswith(base) and not await self._ex.execute_ask(
            f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{subject_uri}> a owl:NamedIndividual }} }}"
        ):
            raise ValueError(
                f"{subject_uri} is an external (seed) class or property: its definition belongs to its "
                "source ontology and is not changed here. State the relation from the local entity "
                "instead (e.g. <local> rdfs:subClassOf <seed>), or create a local subclass/subproperty."
            )
        await self._require_existing(graph, base, property_uri)
        if not is_literal:
            _check_uri(object_value)
            if property_uri not in _LINK_PROPERTIES:
                await self._require_existing(graph, base, object_value)
        obj = self._format_object(object_value, is_literal, datatype)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{
            GRAPH <{graph}> {{ <{subject_uri}> <{property_uri}> {obj} }}
        }}
        """)
        await self._unmark_inferred(graph, [(subject_uri, property_uri, obj)])
        if source_chunk_id is not None:
            await self._add_relation_source(graph, subject_uri, property_uri, obj, source_chunk_id)

    async def relation_delete(
        self,
        ontology_id: str,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
    ) -> None:
        _check_uri(subject_uri)
        _check_uri(property_uri)
        graph = self._graph(ontology_id)
        obj = self._format_object(object_value, is_literal, datatype)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE DATA {{
            GRAPH <{graph}> {{ <{subject_uri}> <{property_uri}> {obj} }}
        }}
        """)
        # Drop the reified provenance records of this triple, if any.
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }}
        WHERE  {{ GRAPH <{graph}> {{ {_statements_of(subject_uri, property_uri, obj)} ?st ?sp ?so }} }}
        """)

    async def relation_sources(
        self,
        ontology_id: str,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = False,
        datatype: str | None = None,
    ) -> list[str]:
        """Chunk IDs this (subject, property, object) triple was extracted from."""
        _check_uri(subject_uri)
        _check_uri(property_uri)
        graph = self._graph(ontology_id)
        obj = self._format_object(object_value, is_literal, datatype)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?chunk WHERE {{
            GRAPH <{graph}> {{ {_statements_of(subject_uri, property_uri, obj)} ?st <urn:olaf:extractedFrom> ?chunk }}
        }}
        """, ["chunk"])
        return [_chunk_id_from_uri(row["chunk"]) for row in rows if row["chunk"]]

    async def relation_search(
        self,
        ontology_id: str,
        subject_uri: str | None = None,
        property_uri: str | None = None,
        object_uri: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        filters = []
        if subject_uri:
            filters.append(f"FILTER(?s = <{_check_uri(subject_uri)}>)")
        if property_uri:
            filters.append(f"FILTER(?p = <{_check_uri(property_uri)}>)")
        if object_uri:
            filters.append(f"FILTER(?o = <{_check_uri(object_uri)}>)")
        filter_block = "\n                ".join(filters)

        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o WHERE {{
            GRAPH <{self._graph(ontology_id)}> {{
                ?s ?p ?o .
                {filter_block}
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o"])
        return [
            {"subject": row["s"] or "", "property": row["p"] or "", "object": row["o"] or ""}
            for row in rows
        ]

    # ── OWL Restrictions ──────────────────────────────────────────────────

    async def restriction_add(
        self,
        ontology_id: str,
        class_uri: str,
        property_uri: str,
        restriction_type: str,
        value: str,
        cardinality: int | None = None,
        is_literal_value: bool = False,
    ) -> str:
        if restriction_type not in _RESTRICTION_PROP:
            raise ValueError(f"Unknown restriction_type '{restriction_type}'. Valid: {list(_RESTRICTION_PROP)}")
        _check_uri(class_uri)
        _check_uri(property_uri)
        if value and not is_literal_value:
            _check_uri(value)

        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        self._require_local(base, class_uri, "restriction_add")
        await self._require_existing(graph, base, class_uri)
        await self._require_existing(graph, base, property_uri)
        if value and not is_literal_value:
            await self._require_existing(graph, base, value)
        # INSERT DATA forbids blank nodes (SPARQL 1.1 §3.1.1).
        # INSERT { } WHERE {} creates a fresh blank node per execution.
        bnode = "_:r"

        if restriction_type in ("exactly", "min", "max"):
            if cardinality is None:
                raise ValueError(f"restriction_type='{restriction_type}' requires cardinality")
            if value:
                prop_name = _RESTRICTION_PROP[restriction_type]
                extra = f"{bnode} owl:onClass <{value}> ."
            else:
                prop_name = _UNQUALIFIED_RESTRICTION_PROP[restriction_type]
                extra = ""
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{cardinality}"^^xsd:nonNegativeInteger .
                    {extra}
                }}
            }}
            WHERE {{}}
            """)
        elif is_literal_value:
            prop_name = _RESTRICTION_PROP[restriction_type]
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} "{_esc(value)}" .
                }}
            }}
            WHERE {{}}
            """)
        else:
            prop_name = _RESTRICTION_PROP[restriction_type]
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{
                GRAPH <{graph}> {{
                    <{class_uri}> rdfs:subClassOf {bnode} .
                    {bnode} a owl:Restriction ;
                             owl:onProperty <{property_uri}> ;
                             {prop_name} <{value}> .
                }}
            }}
            WHERE {{}}
            """)
        return bnode

    # ── Disjointness ──────────────────────────────────────────────────────

    async def disjoint_add(self, ontology_id: str, class_uris: list[str]) -> dict:
        """Declare the given classes pairwise disjoint (owl:disjointWith, one direction
        per pair so a single relation_delete undoes it)."""
        uris = list(dict.fromkeys(_check_uri(u) for u in class_uris))
        if len(uris) < 2:
            raise ValueError("disjoint_add needs at least 2 distinct class URIs.")
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        for uri in uris:
            await self._require_existing(graph, base, uri)
            if uri.startswith(base) and not await self._ex.execute_ask(
                f"{_PREFIXES}\nASK {{ GRAPH <{graph}> {{ <{uri}> a owl:Class }} }}"
            ):
                raise ValueError(f"Not an owl:Class: {uri!r}. Disjointness only applies to classes.")

        # The local class of a pair is the subject, so a seed class is never given an axiom
        # of its own (owl:disjointWith is symmetric).
        pairs = [(b, a) if not a.startswith(base) and b.startswith(base) else (a, b)
                 for i, a in enumerate(uris) for b in uris[i + 1:]]
        for a, b in pairs:
            # A class disjoint with one of its own ancestors is unsatisfiable by construction.
            if await self._ex.execute_ask(f"""
            {_PREFIXES}
            ASK {{ GRAPH <{graph}> {{ {{ <{a}> rdfs:subClassOf+ <{b}> }} UNION {{ <{b}> rdfs:subClassOf+ <{a}> }} }} }}
            """):
                raise ValueError(
                    f"{a!r} and {b!r} are in a subclass relation: declaring them disjoint would make "
                    "the subclass empty. Only declare disjointness between classes that cannot share instances."
                )

        added = 0
        for a, b in pairs:
            if await self._ex.execute_ask(f"""
            {_PREFIXES}
            ASK {{ GRAPH <{graph}> {{ {{ <{a}> owl:disjointWith <{b}> }} UNION {{ <{b}> owl:disjointWith <{a}> }} }} }}
            """):
                continue
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{graph}> {{ <{a}> owl:disjointWith <{b}> }} }}
            """)
            added += 1
        return {"pairs_added": added, "pairs_existing": len(pairs) - added}

    # ── Validation ────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def _asserted_view(self, ontology_id: str):
        """Graph to run the checks on: the ontology without its materialized inferences
        (an inferred rdf:type or rdfs:subClassOf would hide the very errors the checks
        look for). The ontology graph itself when it has none, else a temporary copy."""
        graph = self._graph(ontology_id)
        if not await self._ex.execute_ask(
            f"ASK {{ GRAPH <{graph}> {{ ?st <{INFERRED_BY}> ?by }} }}"
        ):
            yield graph
            return
        view = f"urn:olaf:tmp:asserted:{uuid.uuid4().hex}"
        try:
            await self._ex.execute_update(f"COPY <{graph}> TO <{view}>")
            await self._ex.execute_update(f"""
            {_PREFIXES}
            DELETE {{ GRAPH <{view}> {{ ?s ?p ?o }} }}
            WHERE  {{ GRAPH <{view}> {{ {_INFERRED_ONLY} ?s ?p ?o }} }}
            """)
            # …and the inferred marks themselves, whose statement nodes reference entities.
            await self._ex.execute_update(f"""
            DELETE {{ GRAPH <{view}> {{ ?st ?sp ?so }} }}
            WHERE  {{ GRAPH <{view}> {{
                ?st <{INFERRED_BY}> ?by ; ?sp ?so .
                FILTER NOT EXISTS {{ ?st <urn:olaf:extractedFrom> ?chunk }}
            }} }}
            """)
            yield view
        finally:
            await self._ex.execute_update(f"DROP SILENT GRAPH <{view}>")

    async def orphans(self, ontology_id: str, limit: int = 100) -> list[dict]:
        """Classes/individuals with no relation to the rest of the graph beyond their own
        rdf:type/label/altLabel/definition/extractedFrom bookkeeping triples."""
        async with self._asserted_view(ontology_id) as graph:
            return await self._orphans_in(graph, limit)

    async def _orphans_in(self, graph: str, limit: int) -> list[dict]:
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?e (SAMPLE(?lbl) AS ?label) (SAMPLE(?knd) AS ?kind) WHERE {{
            GRAPH <{graph}> {{
                {{ ?e a owl:Class . BIND("class" AS ?knd) }}
                UNION
                {{ ?e a owl:NamedIndividual . BIND("individual" AS ?knd) }}
                OPTIONAL {{ ?e rdfs:label ?lbl }}
                FILTER NOT EXISTS {{
                    ?e ?p ?o .
                    FILTER(?p NOT IN (rdf:type, rdfs:label, rdfs:altLabel, skos:definition, <urn:olaf:extractedFrom>))
                }}
                FILTER NOT EXISTS {{ ?s ?p2 ?e }}
            }}
        }}
        GROUP BY ?e
        LIMIT {limit}
        """, ["e", "label", "kind"])
        return [
            {"uri": row["e"] or "", "label": row["label"] or "", "kind": row["kind"] or ""}
            for row in rows
        ]

    async def integrity_checks(self, ontology_id: str, limit: int = 50) -> dict:
        """Closed-world checks the OWL reasoner does not report: under OWL semantics
        rdfs:domain/range only *infer* types (and a relation between two classes is
        punning, not constrained at all), and subclass cycles silently mean equivalence."""
        async with self._asserted_view(ontology_id) as graph:
            return await self._integrity_checks_in(graph, limit)

    async def _integrity_checks_in(self, graph: str, limit: int) -> dict:
        # Subject/object "conforms" to class ?c if it is an instance of ?c or a subclass of
        # ?c (relations between classes read as "every X relates to some Y").
        conforms = "(rdf:type|rdfs:subClassOf)/rdfs:subClassOf*"

        cycles = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?c WHERE {{
            GRAPH <{graph}> {{ ?c a owl:Class . ?c rdfs:subClassOf+ ?c }}
        }}
        LIMIT {limit}
        """, ["c"])

        untyped = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?i WHERE {{
            GRAPH <{graph}> {{
                ?i a owl:NamedIndividual .
                FILTER NOT EXISTS {{ ?i a ?t . FILTER(?t NOT IN (owl:NamedIndividual, owl:Thing)) }}
            }}
        }}
        LIMIT {limit}
        """, ["i"])

        domain = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o ?expected WHERE {{
            GRAPH <{graph}> {{
                ?p rdfs:domain ?expected .
                FILTER(isIRI(?expected))
                ?s ?p ?o .
                FILTER(?s != ?expected)
                FILTER NOT EXISTS {{ ?s {conforms} ?expected }}
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o", "expected"])

        obj_range = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o ?expected WHERE {{
            GRAPH <{graph}> {{
                ?p a owl:ObjectProperty ; rdfs:range ?expected .
                FILTER(isIRI(?expected))
                ?s ?p ?o .
                FILTER(isIRI(?o) && ?o != ?expected)
                FILTER NOT EXISTS {{ ?o {conforms} ?expected }}
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o", "expected"])

        data_range = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o ?expected WHERE {{
            GRAPH <{graph}> {{
                ?p a owl:DatatypeProperty ; rdfs:range ?expected .
                FILTER(isIRI(?expected) && ?expected != rdfs:Literal)
                ?s ?p ?o .
                FILTER(isLiteral(?o) && DATATYPE(?o) != ?expected)
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o", "expected"])

        kind = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o ?problem WHERE {{
            GRAPH <{graph}> {{
                {{ ?p a owl:ObjectProperty . ?s ?p ?o . FILTER(isLiteral(?o))
                   BIND("object property used with a literal value" AS ?problem) }}
                UNION
                {{ ?p a owl:DatatypeProperty . ?s ?p ?o . FILTER(!isLiteral(?o))
                   BIND("datatype property used with a URI value" AS ?problem) }}
            }}
        }}
        LIMIT {limit}
        """, ["s", "p", "o", "problem"])

        def triples(rows: list[dict], extra: str) -> list[dict]:
            return [
                {"subject": r["s"] or "", "property": r["p"] or "", "object": r["o"] or "", extra: r[extra] or ""}
                for r in rows
            ]

        return {
            "subclass_cycles": [r["c"] or "" for r in cycles],
            "untyped_individuals": [r["i"] or "" for r in untyped],
            "domain_violations": triples(domain, "expected"),
            "range_violations": triples(obj_range, "expected") + triples(data_range, "expected"),
            "property_kind_mismatches": triples(kind, "problem"),
            "unknown_terms": await self._unknown_terms_in(graph, limit),
        }

    async def _unknown_terms_in(self, graph: str, limit: int) -> list[dict]:
        """Triples using, as property or as class, a term of a standard vocabulary's namespace
        that the vocabulary does not define — a typo no tool or reasoner understands."""
        namespaces = " || ".join(f'STRSTARTS(STR(?t), "{ns}")' for ns in _VOCABULARIES)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?t WHERE {{ GRAPH <{graph}> {{
            {{ ?s ?t ?o }} UNION {{ ?s rdf:type ?t }} FILTER({namespaces})
        }} }}
        """, ["t"])
        found = []
        for term, suggestion in filter(None, (_unknown_term(r["t"] or "") for r in rows)):
            uses = await self._ex.execute_select(f"""
            SELECT ?s ?p ?o WHERE {{ GRAPH <{graph}> {{
                {{ ?s <{term}> ?o BIND(<{term}> AS ?p) }}
                UNION {{ ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{term}>
                         BIND(<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> AS ?p) BIND(<{term}> AS ?o) }}
                FILTER(!STRSTARTS(STR(?s), "urn:olaf:stmt:"))
            }} }} LIMIT {limit}
            """, ["s", "p", "o"])
            found += [{"subject": u["s"] or "", "property": u["p"] or "", "object": u["o"] or "",
                       "term": term, "suggestion": suggestion or ""} for u in uses]
        return found[:limit]

    # ── Seeds ─────────────────────────────────────────────────────────────

    async def seed_load(
        self,
        seed_id: str | None = None,
        url: str | None = None,
        content: str | bytes | None = None,
        fmt: str | None = None,
        source_name: str | None = None,
        allow_private_hosts: bool = False,
    ) -> dict:
        """Load a seed into urn:olaf:seed:{seed_id}; an explicit seed_id that already exists is
        replaced. `source_name` (a file name) only serves to guess the format and the default id.
        A URL may not point to a private/loopback address unless allow_private_hosts (the CLI),
        so that an MCP client cannot make the server fetch internal resources."""
        if (url is None) == (content is None):
            raise ValueError("Provide either 'url' or 'content'.")

        http_content_type = None
        if url is not None:
            data, http_content_type = await _fetch_rdf(url, allow_private_hosts)
            source_name = url
        else:
            data = content.encode() if isinstance(content, str) else content
            if len(data) > MAX_SEED_BYTES:
                raise ValueError(f"Seed too large ({len(data)} bytes, max {MAX_SEED_BYTES})")

        if seed_id is None:
            seed_id = default_seed_id(source_name)
            if seed_id is None:
                raise ValueError("No seed id can be derived from the source: pass seed_id.")
            if await self._ex.execute_ask(f"ASK {{ GRAPH <{seed_graph(seed_id)}> {{ ?s ?p ?o }} }}"):
                raise ValueError(f"A seed '{seed_id}' already exists: pass seed_id='{seed_id}' to replace it, "
                                 "or another seed_id to keep both.")
        graph = seed_graph(seed_id)
        content_type = rdf_content_type(fmt, source_name, http_content_type, data)
        await self._ex.load_graph(data, graph, content_type)

        rows = await self._ex.execute_select(
            f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}",
            ["n"],
        )
        count = int(rows[0]["n"]) if rows and rows[0]["n"] is not None else 0
        return {
            "seed_id": seed_id,
            "graph_uri": graph,
            "source": source_name or "(inline content)",
            "format": content_type,
            "triples_loaded": count,
        }

    async def seed_reuse(
        self, ontology_id: str, uri: str, source_chunk_id: str | None = None, seed_ids: list[str] | None = None,
    ) -> dict:
        """Declare a seed term in the ontology under its own IRI, MIREOT-style: its OWL type,
        its labels and definitions, its parents (rdfs:subClassOf / subPropertyOf), its classes
        for an individual, its domain and range for a property, and rdfs:isDefinedBy its source
        ontology. Only the seed's own statements are copied: the term is not redefined."""
        graph = self._graph(ontology_id)
        base = await self._resolve_base(ontology_id)
        if _check_uri(uri).startswith(base):
            raise ValueError(f"{uri} is already in this ontology's namespace: nothing to reuse.")
        seed_id, entry = await self._seed_entry(uri, seed_ids)
        result = {"uri": uri, "seed": seed_id, "kind": entry["kind"], "label": entry["label"]}
        if await self._ex.execute_ask(f"ASK {{ GRAPH <{graph}> {{ <{uri}> a ?t }} }}"):
            if source_chunk_id is not None:
                await self._add_source_chunk(graph, uri, source_chunk_id)
            return {**result, "created": False}

        sg = seed_graph(seed_id, strict=False)
        if entry["kind"] == "class":
            owl_type = "owl:Class"
        elif entry["kind"] == "individual":
            owl_type = "owl:NamedIndividual"
        elif entry["property_type"] == "datatype" or (entry["property_type"] is None and (entry["range"] or "").startswith(
                ("http://www.w3.org/2001/XMLSchema#", "http://www.w3.org/2000/01/rdf-schema#Literal"))):
            owl_type = "owl:DatatypeProperty"  # an rdf:Property gets its OWL type from its range
        else:
            owl_type = "owl:ObjectProperty"
        # Source ontology: the term's own rdfs:isDefinedBy, else the seed's ontology whose IRI the
        # term's IRI starts with (a seed graph may merge several, e.g. FIBO and OMG Commons
        # modules), else the term's namespace.
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?o ?own WHERE {{ {{ GRAPH <{sg}> {{ <{uri}> rdfs:isDefinedBy ?o }} BIND(true AS ?own) }}
                                UNION {{ GRAPH <{sg}> {{ ?o a owl:Ontology }} }} FILTER(isIRI(?o)) }}
        """, ["o", "own"])
        own = [r["o"] for r in rows if r["own"]]
        matching = sorted((r["o"] for r in rows if not r["own"] and uri.startswith(r["o"].rstrip("#/"))), key=len)
        defined_by = (own or matching[-1:] or [re.sub(r"[^#/]*$", "", uri)])[0]

        # Labels and definitions: one per distinct text (a seed may give "X" and "X"@en, or
        # rdfs:label and skos:prefLabel); the IRI's local name when the seed has no label.
        literals = []
        for predicates, target in (("rdfs:label|skos:prefLabel", "rdfs:label"),
                                   ("skos:definition|rdfs:comment", "skos:definition")):
            values = await self._ex.execute_select(f"""
            {_PREFIXES}
            SELECT ?v (LANG(?v) AS ?lang) WHERE {{ GRAPH <{sg}> {{ <{uri}> {predicates} ?v FILTER(isLiteral(?v)) }} }}
            ORDER BY DESC(LANG(?v))
            """, ["v", "lang"])
            texts: dict[str, str] = {}
            for v in values:
                texts.setdefault(v["v"] or "", v["lang"] or "")
            if target == "rdfs:label" and not texts:
                texts[entry["label"]] = ""
            literals += [f'<{uri}> {target} "{_esc(text)}"' + (f"@{lang}" if lang else "")
                         for text, lang in texts.items() if text]

        # The seed's statements about the term to copy, as (seed pattern, ontology template).
        copied = []
        if entry["kind"] == "class":
            copied.append(("rdfs:subClassOf ?v FILTER(isIRI(?v))", "rdfs:subClassOf ?v"))
        elif entry["kind"] == "individual":
            copied.append(("a ?v FILTER(isIRI(?v) && ?v NOT IN (owl:NamedIndividual, owl:Thing))", "a ?v"))
        else:
            copied += [("rdfs:subPropertyOf ?v FILTER(isIRI(?v))", "rdfs:subPropertyOf ?v"),
                       ("rdfs:domain ?v FILTER(isIRI(?v))", "rdfs:domain ?v"),
                       ("rdfs:range ?v FILTER(isIRI(?v))", "rdfs:range ?v")]
        joined = " .\n            ".join(literals)
        await self._ex.execute_update(f"""
        {_PREFIXES}
        INSERT DATA {{ GRAPH <{graph}> {{
            <{uri}> a {owl_type} ; rdfs:isDefinedBy <{defined_by}> .
            {joined} .
        }} }}
        """)
        for pattern, template in copied:
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT {{ GRAPH <{graph}> {{ <{uri}> {template} }} }}
            WHERE  {{ GRAPH <{sg}> {{ <{uri}> {pattern} }} }}
            """)
        if source_chunk_id is not None:
            await self._add_source_chunk(graph, uri, source_chunk_id)
        return {**result, "created": True}

    async def _seed_entry(self, uri: str, seed_ids: list[str] | None) -> tuple[str, dict]:
        """The seed (among seed_ids, or any) declaring `uri`, and the term's index entry there."""
        if _check_uri(uri).startswith(_STANDARD_NAMESPACES):
            raise ValueError(f"{uri} is a standard vocabulary term: use it directly, without seed_reuse.")
        seed_id = await self._seed_of(uri, seed_ids)
        if seed_id is None:
            if seed_ids is not None and (other := await self._seed_of(uri)):
                raise ValueError(f"{uri} is a term of seed '{other}', which is not selected for this "
                                 f"session (selected: {', '.join(seed_ids) or 'none'}).")
            raise ValueError(
                f"{uri} is not declared (as a class, individual or property) in any seed. Find seed "
                "terms with concept_search/property_search and include_seeds=true."
            )
        return seed_id, (await self._index_entries(seed_graph(seed_id, strict=False), [uri]))[0]

    async def seed_term(self, uri: str, seed_ids: list[str] | None = None, limit: int = 20) -> dict:
        """A seed term as its seed defines it — label, definition, parents, subclasses (or
        subproperties), an individual's classes, a property's domain and range — to judge
        whether it fits before seed_reuse."""
        seed_id, entry = await self._seed_entry(uri, seed_ids)
        sg = seed_graph(seed_id, strict=False)
        down = "rdfs:subPropertyOf" if entry["kind"] == "property" else "rdfs:subClassOf"
        up = "a" if entry["kind"] == "individual" else down

        async def linked(pattern: str) -> list[dict]:
            rows = await self._ex.execute_select(f"""
            {_PREFIXES}
            SELECT ?x (SAMPLE(?l) AS ?label) WHERE {{ GRAPH <{sg}> {{
                {pattern} FILTER(isIRI(?x) && ?x NOT IN (owl:NamedIndividual, owl:Thing))
                OPTIONAL {{ ?x rdfs:label|skos:prefLabel ?l }}
            }} }} GROUP BY ?x LIMIT {limit}
            """, ["x", "label"])
            return [{"uri": r["x"], "label": r["label"] or _local_name(r["x"])} for r in rows]

        result = {
            "uri": uri, "seed": seed_id, "kind": entry["kind"], "label": entry["label"],
            "definition": entry["definition"] or None,
            "parents" if entry["kind"] != "individual" else "classes": await linked(f"<{uri}> {up} ?x"),
        }
        if entry["kind"] != "individual":
            result["children"] = await linked(f"?x {down} <{uri}>")
        if entry["kind"] == "property":
            result.update({k: entry[k] for k in ("property_type", "domain", "domain_label", "range", "range_label")})
        return result

    async def seed_entities(self, seed_id: str) -> list[dict]:
        """The seed's entities as indexed for semantic search (see _index_entries)."""
        return await self._index_entries(seed_graph(seed_id, strict=False))

    async def ontology_entities(self, ontology_id: str, uris: list[str] | None = None) -> list[dict]:
        """The ontology's entities — or only those of `uris` — as indexed for semantic search."""
        return await self._index_entries(self._graph(ontology_id), uris)

    async def _index_entries(self, graph: str, uris: list[str] | None = None) -> list[dict]:
        """Classes (owl:Class, rdfs:Class), individuals (owl:NamedIndividual) and properties
        (owl:ObjectProperty, owl:DatatypeProperty, rdf:Property) of a graph, one entry each:
        {uri, kind, label, definition, source_chunk_id}, plus property_type, domain, range and their
        labels for a property. An entity with several of these types gets the first in that order.
        English or untagged literals are preferred; a missing label is the IRI's local name."""
        types = list(_INDEXED_TYPES)
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT DISTINCT ?e ?t WHERE {{
            {_values("e", uris)}
            GRAPH <{graph}> {{ ?e a ?t FILTER(?t IN ({", ".join(f"<{t}>" for t in types)}) && isIRI(?e)) }}
        }}
        """, ["e", "t"])
        entities: dict[str, dict] = {}
        for row in sorted(rows, key=lambda r: types.index(r["t"])):
            uri = row["e"] or ""
            if not uri or uri.startswith(_BUILTIN_NAMESPACES) or uri in entities:
                continue
            kind, prop_type = _INDEXED_TYPES[row["t"]]
            entities[uri] = {"uri": uri, "kind": kind, "label": "", "definition": "", "source_chunk_id": None}
            if kind == "property":
                entities[uri]["property_type"] = prop_type
        if not entities:
            return []

        subjects = list(entities) if uris else None
        labels = await self._best_literals(graph, _LABEL_PREDICATES, subjects)
        definitions = await self._best_literals(graph, _DEFINITION_PREDICATES, subjects)
        chunks = await self._ex.execute_select(f"""
        SELECT ?e (SAMPLE(?c) AS ?chunk) WHERE {{
            {_values("e", subjects)}
            GRAPH <{graph}> {{ ?e <urn:olaf:extractedFrom> ?c }}
        }} GROUP BY ?e
        """, ["e", "chunk"])
        for row in chunks:
            if row["e"] in entities and row["chunk"]:
                entities[row["e"]]["source_chunk_id"] = _chunk_id_from_uri(row["chunk"])

        properties = [u for u, e in entities.items() if e["kind"] == "property"]
        ends: dict[tuple[str, str], str] = {}
        if properties:
            for field in ("domain", "range"):
                for row in await self._ex.execute_select(f"""
                {_PREFIXES}
                SELECT ?e ?o WHERE {{
                    {_values("e", properties if uris else None)}
                    GRAPH <{graph}> {{ ?e rdfs:{field} ?o FILTER(isIRI(?o)) }}
                }}
                """, ["e", "o"]):
                    if row["e"] in entities:
                        ends.setdefault((row["e"], field), row["o"] or "")
            # Domain/range labels: from this graph, else from any graph (e.g. a seed class).
            missing = list({o for o in ends.values() if o not in labels})
            if missing:
                labels = {**await self._best_literals(None, _LABEL_PREDICATES, missing), **labels}

        for uri, entity in entities.items():
            entity["label"] = labels.get(uri) or _local_name(uri)
            entity["definition"] = definitions.get(uri, "")
            if entity["kind"] == "property":
                for field in ("domain", "range"):
                    end = ends.get((uri, field))
                    entity[field] = end
                    entity[f"{field}_label"] = (labels.get(end) or _local_name(end)) if end else None
        return list(entities.values())

    async def _best_literals(self, graph: str | None, predicates: tuple[str, ...],
                             uris: list[str] | None = None) -> dict[str, str]:
        """For each subject (of `uris`, or all), its preferred literal among `predicates` (in order of
        preference): English first, then untagged, then any language. `graph` None = any graph."""
        rows = await self._ex.execute_select(f"""
        SELECT ?e ?p ?v (LANG(?v) AS ?lang) WHERE {{
            {_values("e", uris)}
            GRAPH {f"<{graph}>" if graph else "?g"} {{
                ?e ?p ?v FILTER(?p IN ({", ".join(f"<{p}>" for p in predicates)}) && isLiteral(?v))
            }}
        }}
        """, ["e", "p", "v", "lang"])
        best: dict[str, tuple] = {}
        for row in rows:
            lang = (row["lang"] or "").lower()
            rank = (0 if lang.startswith("en") else 1 if not lang else 2, predicates.index(row["p"]))
            if row["e"] and (row["e"] not in best or rank < best[row["e"]][0]):
                best[row["e"]] = (rank, row["v"] or "")
        return {uri: value for uri, (_, value) in best.items()}

    # ── Free-form SPARQL (read-only) ──────────────────────────────────────

    async def sparql_query(self, query: str, limit: int = 100, max_chars: int = 50_000) -> dict:
        """Run a caller-written SELECT / ASK / CONSTRUCT / DESCRIBE query over the whole
        store (no implicit graph: use GRAPH <urn:olaf:{id}> to target an ontology).
        Updates are rejected here and, independently, by Oxigraph's /query endpoint.
        SELECT rows beyond `limit` and graph results beyond `max_chars` are truncated."""
        form = _query_form(query)
        if form not in _READ_QUERY_FORMS:
            raise ValueError(
                f"Only read-only queries are allowed ({', '.join(sorted(_READ_QUERY_FORMS))}); got {form or 'unknown'}."
            )

        if form in ("CONSTRUCT", "DESCRIBE"):
            text = (await self._ex.execute_raw_query(query, "text/turtle")).text
            return {"form": form, "turtle": text[:max_chars], "truncated": len(text) > max_chars}

        data = (await self._ex.execute_raw_query(query, "application/sparql-results+json")).json()
        if form == "ASK":
            return {"form": form, "boolean": bool(data.get("boolean", False))}

        variables = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])
        rows = [{v: _fmt_http(b.get(v)) for v in variables} for b in bindings[:limit]]
        return {
            "form": form,
            "variables": variables,
            "rows": rows,
            "row_count": len(bindings),
            "truncated": len(bindings) > limit,
        }

    # ── Export ────────────────────────────────────────────────────────────

    async def export_ttl(self, ontology_id: str, include_seeds: bool = False,
                         seed_ids: list[str] | None = None) -> str:
        result = (await self._ex.dump_graph(self._graph(ontology_id))).decode()

        if include_seeds:
            rows = await self._ex.execute_select(
                f"SELECT DISTINCT ?g WHERE {{ GRAPH ?g {{ ?s ?p ?o }} FILTER({_seed_graphs('?g', seed_ids)}) }}", ["g"]
            )
            for row in rows:
                g = row["g"] or ""
                result += f"\n# Seed graph: {g}\n" + (await self._ex.dump_graph(g)).decode()

        return result

    async def reasoner_input(self, ontology_id: str, include_seeds: bool = False,
                             seed_ids: list[str] | None = None) -> bytes:
        """The ontology's logical content as N-Triples, for the reasoner: provenance
        (extractedFrom, rdf:Statement reification nodes) is dropped, and so are
        owl:imports, which would make the reasoner fetch ontologies over the network, and
        materialized inferences, so that explanations only cite asserted axioms."""
        graph = self._graph(ontology_id)
        graph_filter = f"?g = <{graph}>"
        if include_seeds:
            graph_filter += f" || {_seed_graphs('?g', seed_ids)}"
        resp = await self._ex.execute_raw_query(f"""
        {_PREFIXES}
        CONSTRUCT {{ ?s ?p ?o }} WHERE {{
            GRAPH ?g {{ ?s ?p ?o }}
            FILTER({graph_filter})
            FILTER(?p NOT IN (<urn:olaf:extractedFrom>, owl:imports))
            FILTER(!STRSTARTS(STR(?s), "urn:olaf:stmt:"))
            FILTER NOT EXISTS {{ GRAPH ?g {{ {_INFERRED_ONLY} }} }}
        }}
        """, "application/n-triples")
        return resp.content

    # ── Materialized inferences ───────────────────────────────────────────

    async def inferences_list(self, ontology_id: str) -> list[tuple[str, str, str]]:
        """The (subject, predicate, object term) of the inferred-only triples."""
        rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?s ?p ?o WHERE {{ GRAPH <{self._graph(ontology_id)}> {{ {_INFERRED_ONLY} ?s ?p ?o }} }}
        """, ["s", "p", "o"])
        return [(r["s"] or "", r["p"] or "", f"<{r['o']}>") for r in rows]

    async def inferences_clear(self, ontology_id: str) -> int:
        """Delete the inferred-only triples and every inferred mark. Returns how many
        triples were deleted."""
        graph = self._graph(ontology_id)
        count = len(await self.inferences_list(ontology_id))
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?s ?p ?o }} }}
        WHERE  {{ GRAPH <{graph}> {{ {_INFERRED_ONLY} ?s ?p ?o }} }}
        """)
        # Statement nodes that only carried the mark go entirely; on the others (asserted
        # triples whose mark was not dropped), only the mark goes.
        await self._ex.execute_update(f"""
        {_PREFIXES}
        DELETE {{ GRAPH <{graph}> {{ ?st ?sp ?so }} }}
        WHERE  {{ GRAPH <{graph}> {{
            ?st <{INFERRED_BY}> ?by ; ?sp ?so .
            FILTER NOT EXISTS {{ ?st <urn:olaf:extractedFrom> ?chunk }}
        }} }}
        """)
        await self._ex.execute_update(f"""
        DELETE {{ GRAPH <{graph}> {{ ?st <{INFERRED_BY}> ?by }} }}
        WHERE  {{ GRAPH <{graph}> {{ ?st <{INFERRED_BY}> ?by }} }}
        """)
        return count

    async def inferences_store(self, ontology_id: str, triples: list[tuple[str, str, str]], by: str) -> int:
        """Replace the materialized inferences with `triples` (subject, predicate, object
        term), each marked as inferred by `by`. Returns how many were stored."""
        graph = self._graph(ontology_id)
        await self.inferences_clear(ontology_id)
        for start in range(0, len(triples), 200):
            lines = []
            for s, p, o in triples[start : start + 200]:
                stmt = _stmt_uri(s, p, o)
                lines.append(f"<{s}> <{p}> {o} .")
                lines.append(
                    f"<{stmt}> a rdf:Statement ; rdf:subject <{s}> ; rdf:predicate <{p}> ; "
                    f'rdf:object {o} ; <{INFERRED_BY}> "{_esc(by)}" .'
                )
            body = "\n                ".join(lines)
            await self._ex.execute_update(f"""
            {_PREFIXES}
            INSERT DATA {{ GRAPH <{graph}> {{
                {body}
            }} }}
            """)
        return len(triples)

    # ── Summary ───────────────────────────────────────────────────────────

    async def summary(self, ontology_id: str) -> dict:
        graph = self._graph(ontology_id)
        root_rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?c ?label WHERE {{
            GRAPH <{graph}> {{
                ?c a owl:Class .
                OPTIONAL {{ ?c rdfs:label ?label }}
                FILTER NOT EXISTS {{
                    ?c rdfs:subClassOf ?parent .
                    FILTER(!isBlank(?parent))
                    FILTER(?parent != owl:Thing)
                }}
            }}
        }}
        LIMIT 20
        """, ["c", "label"])
        root_classes = [
            {"uri": row["c"] or "", "label": row["label"] or row["c"] or ""}
            for row in root_rows
        ]

        counts_rows = await self._ex.execute_select(f"""
        {_PREFIXES}
        SELECT ?classes ?object_properties ?datatype_properties ?individuals ?restrictions WHERE {{
            {{ SELECT (COUNT(DISTINCT ?x) AS ?classes)            WHERE {{ GRAPH <{graph}> {{ ?x a owl:Class }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?object_properties)  WHERE {{ GRAPH <{graph}> {{ ?x a owl:ObjectProperty }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?datatype_properties) WHERE {{ GRAPH <{graph}> {{ ?x a owl:DatatypeProperty }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?individuals)         WHERE {{ GRAPH <{graph}> {{ ?x a owl:NamedIndividual }} }} }}
            {{ SELECT (COUNT(DISTINCT ?x) AS ?restrictions)        WHERE {{ GRAPH <{graph}> {{ ?x a owl:Restriction }} }} }}
        }}
        """, ["classes", "object_properties", "datatype_properties", "individuals", "restrictions"])
        counts = {k: int(v or 0) for k, v in (counts_rows[0] if counts_rows else {}).items()}

        return {
            "active_ontology": ontology_id,
            **counts,
            "inferred_triples": len(await self.inferences_list(ontology_id)),
            # Since the server started; later changes are not reflected until the next check.
            "last_check": self._last_check.get(ontology_id),
            "root_classes": root_classes,
        }

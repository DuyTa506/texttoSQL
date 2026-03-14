"""
Node Enricher — offline LLM-based enrichment of KGNode descriptions and synonyms.

Calls an LLM (OpenAI or Anthropic) once per TABLE / COLUMN node and stores:
  - ``description``: 1–2 sentence natural language explanation of what the
    schema element stores, its business meaning, and typical query patterns.
  - ``synonyms``: 4–8 natural language phrases / words that a user might
    use in a question to refer to this schema element.

After enrichment, call ``embed_nodes()`` to compute description embeddings
(via SentenceTransformer) and store them in ``KGNode.embedding``.
These embeddings are then used by:
  - ``build_semantic_edges()``  → EMBEDDING_SIMILAR + SYNONYM_MATCH edges
  - ``SchemaGraph.retrieve()``  → cosine entry-point scoring in PPR

This module is intentionally separate from the graph builder — enrichment is
a one-time offline cost (~$3–15 for all Spider + BIRD schemas with gpt-4o-mini)
and the results are persisted in the serialised graph JSON.

Usage (offline build script)
-----------------------------
  builder  = SchemaGraphBuilder()
  graph    = builder.build_many(databases)
  enricher = NodeEnricher(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
  enricher.enrich(graph, batch_size=20, sleep_between_batches=1.0)
  enricher.embed_nodes(graph)   # requires sentence-transformers
  graph.save("data/schema_graphs/spider_enriched.json")

Batching
--------
  Nodes are grouped into batches and each batch is sent as a single LLM request
  using a structured JSON response format.  This reduces API calls from O(N)
  to O(N / batch_size) and cuts cost significantly.

  The LLM is asked to return a JSON object:
    {
      "node_id_1": {"description": "...", "synonyms": ["...", ...]},
      "node_id_2": ...
    }

  If parsing fails for any node, it is re-tried individually (fallback).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.schema_graph.graph_types import KGNode, NodeType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a database documentation assistant.
Given schema elements (tables and columns) from a relational database, you write:
1. A concise 1-2 sentence description of what each element stores or represents.
2. A list of 4-8 natural language synonyms or phrases a non-technical user might
   use when asking about this element in plain English.

Always respond with valid JSON only — no markdown, no explanation outside the JSON.
"""

_BATCH_PROMPT_TEMPLATE = """\
Database: {db_id}
Context: {db_context}

For each schema element below, provide:
  - "description": 1-2 sentences explaining what it stores, its business meaning,
    and when it would appear in a query.
  - "synonyms": list of 4-8 NL words/phrases a user might say (e.g. "hired" for hire_date).

Schema elements to document:
{elements_json}

Respond with a JSON object mapping each node_id to its description and synonyms:
{{
  "<node_id>": {{"description": "...", "synonyms": ["...", "..."]}},
  ...
}}
"""

_SINGLE_PROMPT_TEMPLATE = """\
Database: {db_id}
Context: {db_context}

Schema element: {node_label}

Provide:
  - "description": 1-2 sentences explaining what it stores, its business meaning,
    and when it would appear in a query.
  - "synonyms": list of 4-8 NL words/phrases a user might say.

Respond with valid JSON only:
{{"description": "...", "synonyms": ["...", "..."]}}
"""


# ---------------------------------------------------------------------------
# NodeEnricher
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Summary returned by ``NodeEnricher.enrich()``."""
    total_nodes: int = 0
    enriched: int = 0
    skipped: int = 0          # already had description
    failed: int = 0
    api_calls: int = 0


class NodeEnricher:
    """
    Enriches TABLE and COLUMN nodes in a ``SchemaGraph`` with LLM-generated
    descriptions and synonyms.

    Parameters
    ----------
    model:
        LLM model identifier.
        OpenAI: "gpt-4o-mini", "gpt-4o"
        Anthropic: "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"
    api_key:
        API key for the chosen provider.  If None, reads from environment
        (OPENAI_API_KEY or ANTHROPIC_API_KEY).
    provider:
        "openai" (default) or "anthropic".  Auto-detected from model name
        if not set.
    max_retries:
        Number of retry attempts per API call on transient errors.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.provider = provider or _detect_provider(model)
        self.max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        graph,       # SchemaGraph — avoid circular import at runtime
        *,
        db_id: Optional[str] = None,
        batch_size: int = 15,
        skip_existing: bool = True,
        sleep_between_batches: float = 0.5,
        force_regen: bool = False,
    ) -> EnrichmentResult:
        """
        Enrich all TABLE and COLUMN nodes with descriptions and synonyms.

        Parameters
        ----------
        graph:
            The ``SchemaGraph`` to enrich in-place.
        db_id:
            Restrict to one database.
        batch_size:
            Number of nodes per LLM API call.  Smaller = more reliable parsing.
            Larger = fewer API calls.  Default 15 is a good balance.
        skip_existing:
            If True (default), skip nodes that already have a non-empty description.
            Set False to re-generate all descriptions.
        sleep_between_batches:
            Seconds to sleep between API calls to avoid rate limits.
        force_regen:
            Alias for ``skip_existing=False``.
        """
        if force_regen:
            skip_existing = False

        # Select nodes to enrich
        target_types = (NodeType.TABLE, NodeType.COLUMN)
        candidates = [
            n for n in graph.nodes.values()
            if n.node_type in target_types
            and (db_id is None or n.db_id == db_id)
            and (not skip_existing or not n.description)
        ]

        result = EnrichmentResult(total_nodes=len(candidates))

        if not candidates:
            logger.info("NodeEnricher: no nodes to enrich.")
            return result

        # Group by db_id for context-aware batching
        by_db: dict[str, list[KGNode]] = {}
        for node in candidates:
            by_db.setdefault(node.db_id, []).append(node)

        for current_db, nodes in sorted(by_db.items()):
            db_context = _build_db_context(graph, current_db)
            logger.info(
                "Enriching %d nodes for db='%s' in batches of %d ...",
                len(nodes), current_db, batch_size,
            )

            for batch_start in range(0, len(nodes), batch_size):
                batch = nodes[batch_start: batch_start + batch_size]

                try:
                    enriched_count = self._enrich_batch(
                        batch, current_db, db_context
                    )
                    result.enriched += enriched_count
                    result.api_calls += 1
                except Exception as exc:
                    logger.warning(
                        "Batch enrichment failed for db=%s batch %d: %s. "
                        "Falling back to individual node enrichment.",
                        current_db, batch_start // batch_size, exc,
                    )
                    for node in batch:
                        try:
                            self._enrich_single(node, current_db, db_context)
                            result.enriched += 1
                            result.api_calls += 1
                        except Exception as exc2:
                            logger.warning("Failed to enrich %s: %s", node.node_id, exc2)
                            result.failed += 1

                if sleep_between_batches > 0 and batch_start + batch_size < len(nodes):
                    time.sleep(sleep_between_batches)

        logger.info(
            "NodeEnricher.enrich() complete: "
            "%d enriched, %d failed, %d api_calls",
            result.enriched, result.failed, result.api_calls,
        )
        return result

    def embed_nodes(
        self,
        graph,
        *,
        db_id: Optional[str] = None,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        batch_size: int = 128,
        skip_existing: bool = True,
        # ── ChromaDB storage (recommended) ────────────────────────────────────
        chroma_persist_dir: Optional[str] = None,
        chroma_collection_name: str = "schema_graph_nodes",
    ) -> int:
        """
        Compute embeddings for all enriched TABLE and COLUMN nodes and persist them.

        Storage modes
        -------------
        ChromaDB (recommended, default when ``chroma_persist_dir`` is set):
            Embeddings are upserted into a dedicated ChromaDB collection
            (``chroma_collection_name``).  The graph JSON stays lean — no large
            float arrays.  At inference time ``SchemaGraph.attach_chroma()``
            binds the collection so ``retrieve()`` uses fast HNSW ANN search
            instead of a linear numpy scan.

        In-node fallback (when ``chroma_persist_dir`` is None):
            Embeddings are stored as ``list[float]`` on each ``KGNode``
            and serialised into the graph JSON.  Simple but larger files
            and slower O(N) scan at query time.

        The embedding text for each node is:
          "{raw_name}. {description}. Synonyms: {synonym_1}, ..."
        Falls back gracefully when description / synonyms are missing.

        Parameters
        ----------
        graph:
            The ``SchemaGraph`` to update in-place.
        db_id:
            Restrict to one database.
        embedding_model:
            SentenceTransformer model name.  **Must match** the model used
            for question embedding at inference time.
        batch_size:
            Encode batch size.  Larger = faster but more VRAM.
        skip_existing:
            Skip nodes that already have a non-empty ``node.embedding``
            *or* are already present in the ChromaDB collection.
        chroma_persist_dir:
            Path to the ChromaDB persistent directory.  When set, embeddings
            go into ChromaDB instead of (or in addition to) ``node.embedding``.
            Use the *same* directory as ``indexing.chroma_persist_dir`` in
            ``configs/default.yaml`` to share one ChromaDB instance.
        chroma_collection_name:
            Name of the ChromaDB collection for graph node embeddings.
            Keep separate from ``"schema_chunks"`` (Phase 2 chunk collection)
            so the two indices don't interfere.

        Returns
        -------
        int
            Number of nodes whose embeddings were (re-)computed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embed_nodes(). "
                "Install with: pip install sentence-transformers"
            )

        target_types = (NodeType.TABLE, NodeType.COLUMN)

        # ── Determine which nodes need embedding ──────────────────────────────
        # When using ChromaDB: "existing" means already in the collection.
        chroma_collection = None
        existing_chroma_ids: set[str] = set()

        if chroma_persist_dir:
            chroma_collection = _open_chroma_collection(
                chroma_persist_dir, chroma_collection_name
            )
            if skip_existing:
                # Fetch all IDs already stored — avoid redundant API calls
                try:
                    existing_chroma_ids = set(
                        chroma_collection.get(include=[])["ids"]
                    )
                except Exception:
                    existing_chroma_ids = set()

        candidates = [
            n for n in graph.nodes.values()
            if n.node_type in target_types
            and (db_id is None or n.db_id == db_id)
            and (
                not skip_existing
                or (
                    # In-node fallback: skip if already embedded in node
                    not n.embedding
                    # ChromaDB path: skip if already in collection
                    and n.node_id not in existing_chroma_ids
                )
            )
        ]

        if not candidates:
            logger.info("embed_nodes: no nodes to embed.")
            return 0

        model = SentenceTransformer(embedding_model)
        texts = [_node_to_embed_text(n) for n in candidates]

        logger.info(
            "Embedding %d nodes with '%s' (storage=%s) ...",
            len(candidates),
            embedding_model,
            f"ChromaDB:{chroma_collection_name}" if chroma_collection else "in-node",
        )
        all_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(candidates) > 50,
            convert_to_numpy=True,
        )

        if chroma_collection is not None:
            # ── ChromaDB path: batch upsert, keep node.embedding empty ────────
            for start in range(0, len(candidates), batch_size):
                end = start + batch_size
                batch_nodes = candidates[start:end]
                batch_embs  = all_embeddings[start:end]

                chroma_collection.upsert(
                    ids=[n.node_id for n in batch_nodes],
                    embeddings=[e.tolist() for e in batch_embs],
                    documents=[_node_to_embed_text(n) for n in batch_nodes],
                    metadatas=[
                        {
                            "node_type":   n.node_type.value,
                            "db_id":       n.db_id,
                            "table_name":  n.table_name,
                            "column_name": n.column_name,
                            "is_pk":       str(n.is_pk),
                            "is_fk":       str(n.is_fk),
                        }
                        for n in batch_nodes
                    ],
                )
                # Clear in-node embedding to keep the graph JSON lean
                for node in batch_nodes:
                    node.embedding = []

            # Attach the collection to the graph so retrieve() uses it
            graph.attach_chroma(chroma_collection)
            logger.info(
                "embed_nodes: upserted %d node embeddings into ChromaDB "
                "collection '%s'.",
                len(candidates),
                chroma_collection_name,
            )
        else:
            # ── In-node fallback ──────────────────────────────────────────────
            for node, emb in zip(candidates, all_embeddings):
                node.embedding = emb.tolist()
            logger.info(
                "embed_nodes: stored embeddings in %d nodes (in-node mode).",
                len(candidates),
            )

        return len(candidates)

    # ── Private: LLM calls ────────────────────────────────────────────────────

    def _enrich_batch(
        self,
        nodes: list[KGNode],
        db_id: str,
        db_context: str,
    ) -> int:
        """
        Enrich a batch of nodes with a single LLM call.

        Returns the count of successfully enriched nodes.
        Mutates each node's ``description`` and ``synonyms`` in-place.
        """
        # Build elements JSON for the prompt
        elements = {
            n.node_id: _node_label(n)
            for n in nodes
        }
        elements_json = json.dumps(elements, indent=2, ensure_ascii=False)

        prompt = _BATCH_PROMPT_TEMPLATE.format(
            db_id=db_id,
            db_context=db_context,
            elements_json=elements_json,
        )

        raw = self._call_llm(prompt)
        parsed = _parse_json_response(raw)

        enriched_count = 0
        for node in nodes:
            entry = parsed.get(node.node_id)
            if not entry:
                # Try matching by short id (table.col) in case the LLM dropped db prefix
                short_id = node.node_id[len(db_id) + 1:]
                entry = parsed.get(short_id)

            if entry and isinstance(entry, dict):
                node.description = str(entry.get("description", "")).strip()
                syns = entry.get("synonyms", [])
                node.synonyms = [str(s).strip() for s in syns if s][:10]
                enriched_count += 1
            else:
                logger.debug("No entry for %s in batch response — will retry individually", node.node_id)

        return enriched_count

    def _enrich_single(
        self,
        node: KGNode,
        db_id: str,
        db_context: str,
    ) -> None:
        """
        Enrich a single node with its own dedicated LLM call.
        Mutates ``node.description`` and ``node.synonyms`` in-place.
        """
        prompt = _SINGLE_PROMPT_TEMPLATE.format(
            db_id=db_id,
            db_context=db_context,
            node_label=_node_label(node),
        )
        raw = self._call_llm(prompt)
        parsed = _parse_json_response(raw)

        if isinstance(parsed, dict) and "description" in parsed:
            node.description = str(parsed.get("description", "")).strip()
            syns = parsed.get("synonyms", [])
            node.synonyms = [str(s).strip() for s in syns if s][:10]
        else:
            logger.warning("Failed to parse single enrichment for %s", node.node_id)

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM provider and return the raw text response.
        Retries up to ``self.max_retries`` on transient errors.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return _call_openai(prompt, self.model, self.api_key)
                elif self.provider == "anthropic":
                    return _call_anthropic(prompt, self.model, self.api_key)
                else:
                    raise ValueError(f"Unknown provider: {self.provider!r}")
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt   # exponential backoff: 1s, 2s, 4s
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_openai(prompt: str, model: str, api_key: Optional[str]) -> str:
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai to use OpenAI enrichment models")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_anthropic(prompt: str, model: str, api_key: Optional[str]) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic to use Anthropic enrichment models")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.content[0].text if response.content else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_provider(model: str) -> str:
    """Infer provider from model name prefix."""
    m = model.lower()
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    logger.warning(
        "Cannot auto-detect provider from model name '%s' — defaulting to 'openai'", model
    )
    return "openai"


def _node_label(node: KGNode) -> str:
    """Human-readable label for a node, used in LLM prompts."""
    if node.node_type == NodeType.TABLE:
        return f"TABLE {node.table_name}"
    elif node.node_type == NodeType.COLUMN:
        pk = " (PRIMARY KEY)" if node.is_pk else ""
        fk = " (FOREIGN KEY)" if node.is_fk else ""
        dtype = f" [{node.dtype}]" if node.dtype else ""
        sv = ""
        if node.sample_values:
            samples = ", ".join(f"'{v}'" for v in node.sample_values[:4])
            sv = f" — sample values: {samples}"
        return f"COLUMN {node.table_name}.{node.column_name}{dtype}{pk}{fk}{sv}"
    return node.node_id


def _build_db_context(graph, db_id: str) -> str:
    """
    Build a brief context string summarising the database schema.
    Used as background for the LLM enrichment prompt.
    """
    table_nodes = [
        n for n in graph.nodes.values()
        if n.db_id == db_id and n.node_type == NodeType.TABLE
    ]
    if not table_nodes:
        return f"Database: {db_id}"

    parts = [f"Database '{db_id}' has {len(table_nodes)} tables:"]
    for t in sorted(table_nodes, key=lambda n: n.table_name):
        col_nodes = [
            n for n in graph.nodes.values()
            if n.db_id == db_id
            and n.node_type == NodeType.COLUMN
            and n.table_name == t.table_name
        ]
        col_names = ", ".join(c.column_name for c in sorted(col_nodes, key=lambda c: c.column_name))
        parts.append(f"  - {t.table_name} ({col_names})")
    return "\n".join(parts)


def _node_to_embed_text(node: KGNode) -> str:
    """
    Compose the text string to embed for a node.

    Format: "{raw_name}. {description}. Synonyms: {synonym_1}, {synonym_2}, ..."
    Falls back gracefully when description / synonyms are missing.
    """
    raw = node.column_name or node.table_name
    parts = [raw]
    if node.description:
        parts.append(node.description)
    if node.synonyms:
        parts.append("Synonyms: " + ", ".join(node.synonyms))
    return ". ".join(parts)


def _open_chroma_collection(persist_dir: str, collection_name: str):
    """
    Open (or create) a ChromaDB persistent collection for graph node embeddings.

    Uses cosine distance space — same as the schema_chunks collection in
    SchemaIndexer — so similarity = 1 - distance.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError(
            "chromadb is required for ChromaDB-backed node embeddings. "
            "Install with: pip install chromadb"
        )

    from pathlib import Path
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_json_response(raw: str) -> dict:    """
    Parse a JSON response from the LLM.

    Handles:
    - Clean JSON: direct json.loads
    - JSON wrapped in markdown code fences: ```json ... ```
    - Partial JSON (truncated response): best-effort extraction
    """
    if not raw:
        return {}

    # Strip markdown code fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines)

    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        match_start = text.find("{")
        match_end = text.rfind("}")
        if match_start != -1 and match_end != -1:
            try:
                return json.loads(text[match_start: match_end + 1])
            except json.JSONDecodeError:
                pass
        logger.debug("Failed to parse JSON response: %s", text[:200])
        return {}

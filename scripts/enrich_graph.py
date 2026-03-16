#!/usr/bin/env python3
"""
Enrich schema graph nodes with LLM descriptions + synonyms, then re-embed.
Uses parallel API calls for speed; small batch_size to avoid JSON parse errors.

Usage:
    python scripts/enrich_graph.py \
        --input  data/schema_graphs/bird_dev.json \
        --output data/schema_graphs/bird_dev_enriched_v2.json \
        --model  gpt-5-nano \
        --batch_size 5 \
        --max_workers 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.schema_graph import SchemaGraph
from src.schema_graph.graph_types import NodeType, KGNode
from src.embeddings import OpenAIEmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("enrich_graph")


# ── LLM call (same prompt as NodeEnricher) ────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a database schema analyst. "
    "Given a list of database schema nodes (tables and columns), "
    "return a JSON object where each key is the node_id and the value is an object with:\n"
    '  "description": a 1-2 sentence plain-English description of what this node stores\n'
    '  "synonyms": a list of 4-8 alternative names or phrases a business user might use\n'
    "Be concise, accurate, and domain-aware. Output valid JSON only."
)


def _call_llm(prompt: str, model: str, api_key: str, max_retries: int = 3) -> str:
    import openai
    client = openai.OpenAI(api_key=api_key)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            logger.warning("LLM attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_err}")


def _build_batch_prompt(nodes: list[KGNode], db_id: str) -> str:
    lines = [f"Database: {db_id}\n\nNodes to enrich:"]
    for n in nodes:
        kind = "TABLE" if n.node_type == NodeType.TABLE else f"COLUMN ({n.dtype or 'unknown'})"
        pk_fk = ""
        if n.is_pk: pk_fk += " [PK]"
        if n.is_fk: pk_fk += " [FK]"
        samples = ""
        if n.sample_values:
            samples = f" | sample values: {', '.join(repr(v) for v in n.sample_values[:3])}"
        lines.append(f'- node_id: "{n.node_id}" | {kind}{pk_fk}{samples}')
    lines.append('\nReturn JSON: { "node_id": { "description": "...", "synonyms": ["..."] }, ... }')
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    """Extract the JSON object from LLM response (handles markdown fences)."""
    import json, re
    # strip ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()
    return json.loads(raw)


def enrich_batch(
    nodes: list[KGNode],
    db_id: str,
    model: str,
    api_key: str,
    counter: dict,
    lock: Lock,
    total: int,
) -> int:
    """Enrich one batch of nodes. Returns number successfully enriched."""
    prompt = _build_batch_prompt(nodes, db_id)
    try:
        raw = _call_llm(prompt, model, api_key)
        parsed = _parse_response(raw)
    except Exception as e:
        logger.warning("[%s] Batch failed (%d nodes): %s", db_id, len(nodes), e)
        return 0

    enriched = 0
    for node in nodes:
        entry = parsed.get(node.node_id) or parsed.get(node.node_id.split(".")[-1]) or {}
        if entry:
            node.description = entry.get("description", "")
            raw_syns = entry.get("synonyms", [])
            node.synonyms = [str(s) for s in raw_syns] if isinstance(raw_syns, list) else []
            enriched += 1

    with lock:
        counter["done"] += len(nodes)
        logger.info("[%s] batch done — %d enriched | progress %d/%d",
                    db_id, enriched, counter["done"], total)
    return enriched


def main():
    p = argparse.ArgumentParser(description="Enrich schema graph + re-embed nodes")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model",  default=os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"))
    p.add_argument("--embed_model",
                   default=os.environ.get("OPENAI_API_EMBEDDING_MODEL", "text-embedding-3-large"))
    p.add_argument("--batch_size",  type=int, default=5,
                   help="Nodes per LLM API call (default 5 — small to avoid JSON errors)")
    p.add_argument("--max_workers", type=int, default=8,
                   help="Parallel API calls (default 8)")
    p.add_argument("--force_regen", action="store_true",
                   help="Re-enrich nodes that already have descriptions")
    p.add_argument("--skip_embed", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading: %s", args.input)
    graph = SchemaGraph.load(args.input)
    all_nodes = [n for n in graph.nodes.values()
                 if n.node_type in (NodeType.TABLE, NodeType.COLUMN)]
    logger.info("Graph: %d total nodes, %d TABLE+COLUMN", len(graph.nodes), len(all_nodes))

    # Per-DB status
    db_stats: dict[str, dict] = {}
    for n in all_nodes:
        s = db_stats.setdefault(n.db_id, {"enriched": 0, "total": 0})
        s["total"] += 1
        if n.description:
            s["enriched"] += 1
    for db_id, s in sorted(db_stats.items()):
        status = "DONE" if s["enriched"] == s["total"] else "NEED"
        logger.info("  %-35s  %3d/%-3d  [%s]", db_id, s["enriched"], s["total"], status)

    # ── Enrich (parallel) ─────────────────────────────────────────────────────
    candidates = [n for n in all_nodes if not n.description or args.force_regen]
    logger.info("Nodes to enrich: %d  (batch=%d, workers=%d, model=%s)",
                len(candidates), args.batch_size, args.max_workers, args.model)

    if candidates:
        # Group by DB then split into batches
        by_db: dict[str, list[KGNode]] = {}
        for n in candidates:
            by_db.setdefault(n.db_id, []).append(n)

        batches: list[tuple[list[KGNode], str]] = []
        for db_id, nodes in sorted(by_db.items()):
            for i in range(0, len(nodes), args.batch_size):
                batches.append((nodes[i:i + args.batch_size], db_id))

        logger.info("Total batches: %d  launching with %d workers...",
                    len(batches), args.max_workers)

        counter = {"done": 0}
        lock = Lock()
        total_enriched = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    enrich_batch, batch_nodes, db_id,
                    args.model, api_key, counter, lock, len(candidates)
                ): (batch_nodes, db_id)
                for batch_nodes, db_id in batches
            }
            for fut in as_completed(futures):
                try:
                    total_enriched += fut.result()
                except Exception as e:
                    logger.error("Future error: %s", e)

        logger.info("Enrichment complete: %d/%d nodes enriched", total_enriched, len(candidates))
    else:
        logger.info("All nodes already enriched, skipping.")

    # ── Re-embed ──────────────────────────────────────────────────────────────
    if not args.skip_embed:
        logger.info("Re-embedding %d nodes with %s...", len(all_nodes), args.embed_model)
        emb_model = OpenAIEmbeddingModel(model=args.embed_model, api_key=api_key)

        texts = []
        for n in all_nodes:
            syns = ", ".join(n.synonyms[:5]) if n.synonyms else ""
            text = n.node_id
            if n.description: text += ". " + n.description
            if syns: text += ". Synonyms: " + syns
            texts.append(text)

        all_vecs: list = []
        BATCH = 100
        for i in range(0, len(texts), BATCH):
            vecs = emb_model.embed(texts[i:i + BATCH])
            all_vecs.extend(vecs)
            logger.info("  Embedded %d/%d", min(i + BATCH, len(texts)), len(texts))

        for node, vec in zip(all_nodes, all_vecs):
            node.embedding = np.array(vec, dtype=np.float32)
        logger.info("Re-embedding done.")

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("Saving to: %s", args.output)
    graph.save(args.output)

    enriched_after = sum(1 for n in all_nodes if n.description)
    has_emb = sum(1 for n in all_nodes if n.embedding is not None)
    logger.info("DONE — enriched=%d/%d  with_embedding=%d/%d",
                enriched_after, len(all_nodes), has_emb, len(all_nodes))


if __name__ == "__main__":
    main()

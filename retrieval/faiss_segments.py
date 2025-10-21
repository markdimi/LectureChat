import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from utils.logging import logger

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    import faiss_cpu as faiss  # type: ignore

import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

TOP_K = 5
MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_TEMPLATE = lambda q: f"query: {q}"

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "retrieval" / "faiss_indices.json"


@lru_cache(maxsize=1)
def _load_index_config() -> List[Dict[str, Any]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfgs: List[Dict[str, Any]] = []
    for item in raw:
        name = item["name"]
        idx_path = Path(item["index_path"]) if isinstance(item["index_path"], str) else item["index_path"]
        meta_path = Path(item["metadata_path"]) if isinstance(item["metadata_path"], str) else item["metadata_path"]
        if not idx_path.is_absolute():
            idx_path = REPO_ROOT / idx_path
        if not meta_path.is_absolute():
            meta_path = REPO_ROOT / meta_path
        cfgs.append(
            {
                "name": name,
                "index_path": idx_path,
                "metadata_path": meta_path,
                "description": item.get("description", ""),
            }
        )
    return cfgs


@lru_cache(maxsize=1)
def _load_all_resources() -> List[Dict[str, Any]]:
    resources: List[Dict[str, Any]] = []
    cfgs = _load_index_config()
    for cfg in cfgs:
        name = cfg["name"]
        index_path: Path = cfg["index_path"]
        meta_path: Path = cfg["metadata_path"]
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Parquet metadata not found: {meta_path}")
        index = faiss.read_index(str(index_path))
        table = pq.read_table(meta_path)
        rows = table.to_pylist()
        resources.append({
            "name": name,
            "description": cfg.get("description", ""),
            "index": index,
            "rows": rows,
        })
    return resources


@lru_cache(maxsize=1)
def _load_embedder() -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def _embed_query(text: str) -> np.ndarray:
    tokenizer, model = _load_embedder()
    templated = QUERY_TEMPLATE(text)
    encoded = tokenizer([templated], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**encoded)
        emb = out[0][:, 0]
    emb_np = emb.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_np = emb_np / norms
    emb_np = emb_np[:, :256]
    return emb_np[0].astype(np.float32)


def retrieve_all_indices(user_query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    resources = _load_all_resources()
    q = _embed_query(user_query).astype(np.float32)
    q = np.ascontiguousarray(q.reshape(1, -1))
    all_hits: List[Dict[str, Any]] = []
    overall_rank = 1
    for res in resources:
        name = res["name"]
        index = res["index"]
        rows = res["rows"]
        D, I = index.search(q, top_k)
        for j in range(min(top_k, I.shape[1])):
            idx = int(I[0, j])
            score = float(D[0, j])
            if idx < 0 or idx >= len(rows):
                continue
            meta = rows[idx]
            if not isinstance(meta, dict):
                meta = json.loads(json.dumps(meta))
            hit = {"rank": overall_rank, "score": score, "id": idx, "index_name": name}
            hit.update(meta)
            all_hits.append(hit)
            overall_rank += 1
    return all_hits


def render_human_answers_from_hits(hits: List[Dict[str, Any]]) -> str:
    # Group by index_name to render a line per configured index
    cfgs = _load_index_config()
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        name = h.get("index_name", "unknown")
        by_name.setdefault(name, []).append(h)

    def first_content(arr: List[Dict[str, Any]]) -> str:
        if not arr:
            return "No relevant results found."
        content = arr[0].get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip().replace("\n", " ")
        if len(content) > 500:
            content = content[:500] + "…"
        return content if content else "No relevant results found."

    lines = ["FAISS answers:"]
    for cfg in cfgs:
        name = cfg["name"]
        lines.append(f"- {name}: {first_content(by_name.get(name, []))}")
    return "\n".join(lines)


def dedupe_and_aggregate_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resources = _load_all_resources()
    rows_by_index: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for res in resources:
        name = res["name"]
        rows = res["rows"]
        rows_by_index[name] = list(enumerate(rows))

    seen: set[Tuple[Any, Any, Any]] = set()
    deduped: List[Dict[str, Any]] = []
    for h in hits:
        bm = h.get("block_metadata", {}) or {}
        video_id = h.get("video_id") or bm.get("video_id")
        segment_index = h.get("segment_index") or bm.get("segment_index")
        key = (h.get("index_name"), video_id, segment_index)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)

    for h in deduped:
        name = h.get("index_name")
        bm = h.get("block_metadata", {}) or {}
        video_id = h.get("video_id") or bm.get("video_id")
        segment_index = h.get("segment_index") or bm.get("segment_index")
        if name not in rows_by_index or video_id is None or segment_index is None:
            continue
        rows_enum = rows_by_index[name]
        matches: List[Tuple[int, Dict[str, Any]]] = []
        for rid, row in rows_enum:
            rbm = row.get("block_metadata", {}) or {}
            rvid = row.get("video_id") or rbm.get("video_id")
            rseg = row.get("segment_index") or rbm.get("segment_index")
            if rvid == video_id and rseg == segment_index:
                matches.append((rid, row))
        if len(matches) <= 1:
            continue
        matches.sort(key=lambda t: t[0])
        parts: List[str] = []
        for _, row in matches:
            c = row.get("content", "")
            if not isinstance(c, str):
                c = str(c)
            parts.append(c)
        h["content"] = " ".join(parts)    
    return deduped


def _get_video_and_spans(h: Dict[str, Any]) -> tuple[Any, Any, Any]:
    bm = h.get("block_metadata", {}) or {}
    video_id = h.get("video_id") or bm.get("video_id")
    start_ms = h.get("start_ms") or bm.get("start_ms")
    end_ms = h.get("end_ms") or bm.get("end_ms")
    return video_id, start_ms, end_ms


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0.0
    try:
        a_start = float(a_start)
        a_end = float(a_end)
        b_start = float(b_start)
        b_end = float(b_end)
    except Exception:
        return 0.0
    if a_end < a_start or b_end < b_start:
        return 0.0
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def group_hits_by_video_iou(
    hits: List[Dict[str, Any]], iou_threshold: float = 0.5
) -> List[List[Dict[str, Any]]]:
    by_video: Dict[Any, List[Dict[str, Any]]] = {}
    logger.info(f"IOU START")
    for h in hits:
        vid, s, e = _get_video_and_spans(h)
        # Keep hits without proper spans as their own singleton groups later
        key = vid if vid is not None else (h.get("index_name"), h.get("id"))
        by_video.setdefault(key, []).append(h)

    groups: List[List[Dict[str, Any]]] = []
    for vid, arr in by_video.items():
        # Extract only items with valid spans; others will be singleton groups
        indices_with_spans: List[int] = []
        for i, h in enumerate(arr):
            _, s, e = _get_video_and_spans(h)
            if s is not None and e is not None:
                indices_with_spans.append(i)

        if len(indices_with_spans) <= 1:
            # Either zero or one item has spans: every item becomes its own group
            for h in arr:
                groups.append([h])
            continue

        n = len(arr)
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            vi, si, ei = _get_video_and_spans(arr[i])
            if si is None or ei is None:
                continue
            for j in range(i + 1, n):
                vj, sj, ej = _get_video_and_spans(arr[j])
                if sj is None or ej is None:
                    continue
                if vi != vj:
                    continue
                iou = interval_iou(si, ei, sj, ej)
                logger.info(f"Interval IoU: {iou}")
                if iou >= iou_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        for i in range(n):
            if visited[i]:
                continue
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(arr[u])
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(comp)

    return groups

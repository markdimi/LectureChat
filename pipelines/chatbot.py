import json
import re

from chainlite import chain, llm_generation_chain, register_prompt_constants
from chainlite.llm_output import extract_tag_from_llm_output, lines_to_list
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph

from corpora import corpus_id_to_corpus_object
from utils.logging import logger
from pipelines.dialogue_state import ChatbotConfig, DialogueState
from retrieval.retriever_api import retrieve_via_api
from retrieval.faiss_segments import (
    retrieve_all_indices,
    render_human_answers_from_hits,
    dedupe_and_aggregate_hits,
    group_hits_by_video_iou,
)
from retrieval.search_result_block import SearchResultBlock

register_prompt_constants({"chatbot_name": "WikiChat"})


@chain
async def query_stage(state):
    query_chain = (
        llm_generation_chain("query.prompt", engine=state.config.engine, max_tokens=200)
        | extract_tag_from_llm_output.bind(tags="search_query")  # type: ignore
        | lines_to_list
    )
    search_prompt_output = await query_chain.ainvoke(
        {"dlg": state, "llm_corpus_description": state.config.llm_corpus_description}
    )
    search_prompt_output = [q for q in search_prompt_output if q and q != "None"]

    if not search_prompt_output:
        logger.info("No search needed.")
        state.current_turn.search_query = []
        return
    logger.info(
        f"Search queries: {json.dumps(search_prompt_output, ensure_ascii=False, indent=2)}"
    )

    state.current_turn.search_query = search_prompt_output


@chain
async def search_stage(state):
    query = state.current_turn.search_query
    config = state.config
    if query:
        try:
            search_result = retrieve_via_api(
                state.current_turn.search_query,
                retriever_endpoint=config.retriever_endpoint,
                do_reranking=config.do_reranking,
                pre_reranking_num=config.query_pre_reranking_num,
                post_reranking_num=config.query_post_reranking_num,
            )
            assert len(search_result) == len(query)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            search_result = []
        state.current_turn.search_results = search_result


@chain
async def generate_stage(state):
    llm_claims_chain = (
        llm_generation_chain(
            "generate_split_claims.prompt", engine=state.config.engine, max_tokens=2000
        )
        | lines_to_list
    )

    llm_claims = await llm_claims_chain.ainvoke({"dlg": state})
    if len(llm_claims) == 0 or (len(llm_claims) == 1 and llm_claims[0] == "None"):
        return

    state.current_turn.llm_claims = llm_claims


@chain
async def llm_claim_search_stage(state):
    queries = state.current_turn.llm_claims
    config = state.config
    if queries:
        search_result = retrieve_via_api(
            queries=queries,
            retriever_endpoint=config.retriever_endpoint,
            do_reranking=config.do_reranking,
            pre_reranking_num=config.claim_pre_reranking_num,
            post_reranking_num=config.claim_post_reranking_num,
        )
        assert len(search_result) == len(queries)
        state.current_turn.llm_claim_search_results = search_result


@chain
async def filter_information_stage(state):
    filter_chain = (
        llm_generation_chain(
            "filter_irrelevant_info.prompt", engine=state.config.engine, max_tokens=2000
        )
        | extract_tag_from_llm_output.bind(tags="relevant_content")  # type: ignore
        | lines_to_list
    )
    filtered_info = await filter_chain.abatch(
        [
            {"dlg": state, "result": result}
            for result in state.current_turn.all_single_search_results
        ]
    )
    for result, filtered_result in zip(
        state.current_turn.all_single_search_results, filtered_info
    ):
        # make a deepcopy of result
        if len(filtered_result) == 0 or (
            len(filtered_result) == 1 and filtered_result[0] == "None"
        ):
            continue
        result = result.copy()
        result.summary = filtered_result
        state.current_turn.filtered_search_results.append(result)


@chain
async def draft_stage(state):
    draft_chain = llm_generation_chain(
        "draft_w_citation.prompt", engine=state.config.engine, max_tokens=2000
    ) | extract_tag_from_llm_output.bind(tags="response")  # type: ignore
    draft_output = await draft_chain.ainvoke({"dlg": state})
    state.current_turn.draft_stage_output = draft_output


@chain
async def shift_references(state):
    agent_utterance = state.current_turn.agent_utterance
    references = state.current_turn.filtered_search_results
    # extract all citations in [1], [2], [3] format
    citations = re.findall(r"\[\d+\]", agent_utterance)
    if len(citations) == 0:
        return
    cited_reference_indices = []
    for citation in citations:
        citation_index = int(citation[1:-1]) - 1
        if 0 <= citation_index < len(references):
            cited_reference_indices.append(citation_index)
    cited_reference_indices = list(set(cited_reference_indices))
    cited_reference_indices.sort()

    reference_map = {}
    for i, index in enumerate(cited_reference_indices):
        reference_map[index] = i

    for i, index in enumerate(cited_reference_indices):
        agent_utterance = agent_utterance.replace(f"[{index + 1}]", f"[{i + 1}]")
    references = [references[i] for i in cited_reference_indices]

    state.current_turn.agent_utterance = agent_utterance
    state.current_turn.filtered_search_results = references


@chain
async def refine_stage(state):
    if not state.config.do_refine:
        state.current_turn.agent_utterance = state.current_turn.draft_stage_output
        return
    refine_chain = llm_generation_chain(
        "refine.prompt", engine=state.config.engine, max_tokens=2000
    ) | extract_tag_from_llm_output.bind(tags="revised_response")  # type: ignore
    refine_output = await refine_chain.ainvoke(
        {"dlg": state, "utterance_to_refine": state.current_turn.draft_stage_output}
    )
    state.current_turn.agent_utterance = refine_output


@chain
async def append_faiss_answers(state):
    # Check if search is needed - respect the same decision as Wikipedia
    # If no search_query AND no llm_claims, then search was deemed unnecessary
    if not state.current_turn.search_query and not state.current_turn.llm_claims:
        logger.info("No FAISS search needed (following Wikipedia search decision)")
        state.current_turn.faiss_answer = None
        state.current_turn.faiss_references = []
        state.current_turn.faiss_json_results = []
        return
    
    # Prefer querying FAISS with LLM-generated claims; fallback to user query if no claims
    claims = state.current_turn.llm_claims or []
    queries = claims if len(claims) > 0 else [state.current_turn.user_utterance]

    def _extract_fields(hit: dict) -> dict:
        bm = hit.get("block_metadata", {}) or {}
        language = hit.get("language")
        if language is None:
            language = bm.get("language")
        ret = {
            "index_name": hit.get("index_name"),
            "id": hit.get("id"),
            "document_title": hit.get("document_title"),
            "section_title": hit.get("section_title"),
            "content": hit.get("content"),
            "language": language,
            "course_name": bm.get("course_name"),
            "course_term": bm.get("course_term"),
            "start_ms": bm.get("start_ms"),
            "start_sec": bm.get("start_sec"),
            "end_ms": bm.get("end_ms"),
            "end_sec": bm.get("end_sec"),
            "video_id": bm.get("video_id"),
            "segment_index": bm.get("segment_index"),
        }
        # Attach custom URL
        ret["custom_url"] = _build_custom_url(hit)
        return ret

    def _title_from_hit(h: dict) -> str | None:
        dt = h.get("document_title")
        st = h.get("section_title")
        if dt and st:
            return f"{dt} > {st}"
        return dt or st

    def _build_custom_url(h: dict) -> str:
        bm = h.get("block_metadata", {}) or {}
        base_url = getattr(state.config, "faiss_citation_url_base", "http://3.69.7.143:9000/videos/")
        idx = h.get("index_name") or "index"
        vid = h.get("video_id") or bm.get("video_id")
        ssec = h.get("start_sec") or bm.get("start_sec")
        esec = h.get("end_sec") or bm.get("end_sec")
        try:
            if ssec is None and bm.get("start_ms") is not None:
                ssec = float(bm.get("start_ms")) / 1000.0
        except Exception:
            pass
        try:
            if esec is None and bm.get("end_ms") is not None:
                esec = float(bm.get("end_ms")) / 1000.0
        except Exception:
            pass
        t = ""
        if ssec is not None:
            try:
                st = f"{int(float(ssec))}"
            except Exception:
                st = ""
        if esec is not None:
            try:
                et = f"{int(float(esec))}"
            except Exception:
                et = ""
        return f"{base_url}/{vid}/dash.mpd?start={st}&end={et}"

    all_faiss_blocks: list[SearchResultBlock] = []
    json_results: list[dict] = []

    # We'll temporarily swap out filtered_search_results to reuse the drafting prompt
    original_refs = state.current_turn.filtered_search_results

    for idx, q in enumerate(queries):
        try:
            hits = retrieve_all_indices(q, top_k=5)
        except Exception:
            hits = []

        # De-duplicate by (index_name, video_id, segment_index) and aggregate content per segment
        if hits:
            try:
                hits = dedupe_and_aggregate_hits(hits)
            except Exception:
                pass

            # Exclude known-bad FAISS hits by video_id (temporary safeguard)
            _banned_video_ids = {"231", "232"}

            def _get_video_id_for_filter(h):
                bm2 = h.get("block_metadata", {}) or {}
                vid = h.get("video_id") or bm2.get("video_id")
                try:
                    return str(int(vid))
                except Exception:
                    return str(vid) if vid is not None else ""

            hits = [h for h in hits if _get_video_id_for_filter(h) not in _banned_video_ids]

        # LLM reranking (pointwise Yes/No) against the claim/query
        if hits:
            try:
                rerank_chain = llm_generation_chain(
                    "rerank_pointwise.prompt", engine=state.config.engine, max_tokens=20
                )
                rr_inputs = [
                    {
                        "query": q,
                        "search_result": {
                            "full_title": _title_from_hit(h) or "",
                            "date_human_readable": "",
                            "content": str(h.get("content", "")),
                        },
                    }
                    for h in hits
                ]
                rr_outputs = await rerank_chain.abatch(rr_inputs)
                def _is_yes(x: str) -> bool:
                    s = (x or "").strip().lower()
                    return s.startswith("yes") or s == "y"
                mask = [_is_yes(o) for o in rr_outputs]
                filtered_hits = [h for h, keep in zip(hits, mask) if keep]
                if filtered_hits:
                    hits = filtered_hits
            except Exception:
                pass

        # IoU grouping across the same video and selecting best
        if hits:
            try:
                select_chain = (
                    llm_generation_chain(
                        "faiss_select_best.prompt", engine=state.config.engine, max_tokens=50
                    )
                    | extract_tag_from_llm_output.bind(tags="best_index")  # type: ignore
                )
                groups = group_hits_by_video_iou(hits, iou_threshold=state.config.faiss_iou_threshold)
                selected_hits = []
                for grp in groups:
                    if len(grp) <= 1:
                        selected_hits.extend(grp)
                        continue
                    cands = []
                    for h in grp:
                        bm = h.get("block_metadata", {}) or {}
                        cands.append({
                            "index_name": h.get("index_name"),
                            "id": h.get("id"),
                            "video_id": h.get("video_id") or bm.get("video_id"),
                            "start_ms": h.get("start_ms") or bm.get("start_ms"),
                            "end_ms": h.get("end_ms") or bm.get("end_ms"),
                            "content": str(h.get("content", "")),
                        })
                    try:
                        best_str = await select_chain.ainvoke({"claim": q, "candidates": cands})
                        best_raw = str(best_str).strip()
                        best_idx = int(best_raw) if best_raw.isdigit() else int(best_raw.split("</best_index>")[-2].split(">")[-1])
                    except Exception:
                        best_idx = 1
                    if best_idx < 1 or best_idx > len(grp):
                        best_idx = 1
                    selected_hits.append(grp[best_idx - 1])
                hits = selected_hits
            except Exception:
                pass

        # LLM filtering to keep only claim-relevant parts; also capture summaries
        summaries: list[list[str]] = []
        if hits:
            try:
                filter_chain = (
                    llm_generation_chain(
                        "filter_irrelevant_info.prompt", engine=state.config.engine, max_tokens=2000
                    )
                    | extract_tag_from_llm_output.bind(tags="relevant_content")  # type: ignore
                    | lines_to_list
                )
                f_inputs = [
                    {
                        "dlg": state,
                        "result": {
                            "full_title": _title_from_hit(h) or "",
                            "content": str(h.get("content", "")),
                        },
                    }
                    for h in hits
                ]
                summaries = await filter_chain.abatch(f_inputs)
                # Keep only hits that have meaningful summaries
                filtered_hits = []
                filtered_summaries = []
                for h, lst in zip(hits, summaries):
                    if lst and not (len(lst) == 1 and lst[0] == "None"):
                        h = dict(h)
                        # Store original content before filtering
                        h["original_content"] = h.get("content", "")
                        h["content"] = "\n".join(lst)
                        filtered_hits.append(h)
                        filtered_summaries.append(lst)
                
                # Limit to 2 per claim to match Wikipedia behavior
                hits = filtered_hits[:2] if filtered_hits else []
                summaries = filtered_summaries[:2] if filtered_summaries else []
            except Exception:
                summaries = [[] for _ in hits]

        # Convert to SearchResultBlock objects with custom URL and metadata for FAISS
        faiss_blocks: list[SearchResultBlock] = []
        for i, h in enumerate(hits or []):
            bm = h.get("block_metadata", {}) or {}
            # Fill required fields; fallbacks for None
            doc_title = h.get("document_title") or ""
            sec_title = h.get("section_title") or ""
            content = str(h.get("content", ""))
            original_content = str(h.get("original_content", content))  # Preserve original
            lang = h.get("language") or bm.get("language")
            url = _build_custom_url(h)
            custom_meta = {
                "language": lang,
                "index_name": h.get("index_name"),
                "original_content": original_content,  # Store original content in metadata
                "course_name": bm.get("course_name"),
                "course_term": bm.get("course_term"),
                "start_ms": h.get("start_ms") or bm.get("start_ms"),
                "start_sec": h.get("start_sec") or bm.get("start_sec"),
                "end_ms": h.get("end_ms") or bm.get("end_ms"),
                "end_sec": h.get("end_sec") or bm.get("end_sec"),
                "video_id": h.get("video_id") or bm.get("video_id"),
                "segment_index": h.get("segment_index") or bm.get("segment_index"),
            }
            try:
                sim = float(h.get("score", 0.0))
            except Exception:
                sim = 0.0
            prob = 0.0
            block = SearchResultBlock(
                document_title=doc_title,
                section_title=sec_title,
                content=content,
                url=url,
                block_metadata=custom_meta,
                similarity_score=sim,
                probability_score=prob,
            )
            # Attach summary if available
            if summaries and i < len(summaries) and summaries[i] and not (len(summaries[i]) == 1 and summaries[i][0] == "None"):
                block.summary = summaries[i]
            faiss_blocks.append(block)

        # If we have FAISS blocks, generate a FAISS-only drafted answer with citations
        if faiss_blocks:
            # Collect all FAISS blocks for aggregated answer
            all_faiss_blocks.extend(faiss_blocks)

        # Collect JSON for this query/claim
        json_results.append({
            ("claim" if claims else "query"): q,
            "results": [_extract_fields(h) for h in hits] if faiss_blocks else [],
        })

    # Generate unified FAISS answer if we have any blocks
    faiss_answer = ""
    faiss_references = []
    
    if all_faiss_blocks:
        # Limit to 10 most relevant blocks across all claims
        all_faiss_blocks = all_faiss_blocks[:10]
        
        try:
            state.current_turn.filtered_search_results = all_faiss_blocks
            draft_chain = llm_generation_chain(
                "draft_w_citation.prompt", engine=state.config.engine, max_tokens=2000
            ) | extract_tag_from_llm_output.bind(tags="response")  # type: ignore
            faiss_draft = await draft_chain.ainvoke({"dlg": state})
        finally:
            # Always restore original references
            state.current_turn.filtered_search_results = original_refs
        
        # Convert numeric citations to letter citations [1] -> [a], [2] -> [b], etc.
        citations = re.findall(r"\[\d+\]", faiss_draft)
        cited_indices = []
        for c in citations:
            try:
                idx = int(c[1:-1]) - 1
                if 0 <= idx < len(all_faiss_blocks):
                    cited_indices.append(idx)
            except Exception:
                pass
        
        cited_indices = sorted(set(cited_indices))
        
        # Map numeric to letter citations
        for i, idx in enumerate(cited_indices):
            letter = chr(ord('a') + i)  # a, b, c, ...
            faiss_draft = faiss_draft.replace(f"[{idx+1}]", f"[{letter}]")
            
            # Build reference entry with metadata
            ref = all_faiss_blocks[idx]
            m = ref.block_metadata or {}
            
            # Create reference object for frontend
            faiss_references.append({
                "id": letter,
                "index_name": m.get("index_name", "unknown"),
                "url": ref.url,
                "content": ref.content,  # This is the filtered/summarized content
                "original_content": m.get("original_content", ref.content),  # Original before filtering
                "summary": ref.summary if hasattr(ref, 'summary') else [],  # Summary bullets
                "video_id": m.get("video_id"),
                "segment_index": m.get("segment_index"),
                "start_sec": m.get("start_sec"),
                "end_sec": m.get("end_sec"),
                "course_name": m.get("course_name"),
                "course_term": m.get("course_term"),
                "language": m.get("language", "en"),  # Add language field
            })
        
        faiss_answer = faiss_draft
    
    # Store FAISS data separately for frontend
    state.current_turn.faiss_answer = faiss_answer
    state.current_turn.faiss_references = faiss_references
    state.current_turn.faiss_json_results = json_results
    
    # Note: Don't append to agent_utterance - keep FAISS separate for frontend


def create_chain(args) -> tuple[CompiledStateGraph, DialogueState]:
    llm_corpus_description = corpus_id_to_corpus_object(
        args.corpus_id
    ).llm_corpus_description
    initial_state = DialogueState(
        turns=[],
        config=ChatbotConfig(
            engine=args.engine,
            do_refine=args.do_refine,
            llm_corpus_description=llm_corpus_description,
            retriever_endpoint=args.retriever_endpoint,
            do_reranking=args.do_reranking,
            query_pre_reranking_num=args.query_pre_reranking_num,
            query_post_reranking_num=args.query_post_reranking_num,
            claim_pre_reranking_num=args.claim_pre_reranking_num,
            claim_post_reranking_num=args.claim_post_reranking_num,
        ),
    )

    graph = StateGraph(DialogueState)

    # nodes
    graph.add_node("query_stage", query_stage)
    graph.add_node("search_stage", search_stage)
    graph.add_node("generate_stage", generate_stage)
    graph.add_node("llm_claim_search_stage", llm_claim_search_stage)
    graph.add_node("filter_information_stage", filter_information_stage)
    graph.add_node("draft_stage", draft_stage)
    graph.add_node("refine_stage", refine_stage)
    graph.add_node("shift_references", shift_references)
    graph.add_node("append_faiss_answers", append_faiss_answers)

    # edges
    graph.add_edge(START, "query_stage")
    graph.add_edge("query_stage", "search_stage")

    graph.add_edge(START, "generate_stage")
    graph.add_edge("generate_stage", "llm_claim_search_stage")

    graph.add_edge("search_stage", "filter_information_stage")
    graph.add_edge("llm_claim_search_stage", "filter_information_stage")

    graph.add_edge("filter_information_stage", "draft_stage")

    graph.add_edge("draft_stage", "refine_stage")
    graph.add_edge("refine_stage", "shift_references")
    graph.add_edge("shift_references", "append_faiss_answers")
    graph.add_edge("append_faiss_answers", END)

    runnable = graph.compile()

    # runnable.get_graph().print_ascii()

    return runnable, initial_state

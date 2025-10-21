from chainlite import chain, llm_generation_chain
from chainlite.llm_output import extract_tag_from_llm_output
from pipelines.dialogue_state import DialogueState

@chain
async def faiss_comparison_stage(state: DialogueState):
    """
    Compares FAISS results with Wikipedia results using LLM.
    """
    comparison_chain = (
        llm_generation_chain(
            "faiss_comparison.prompt",
            engine=state.config.engine,
            max_tokens=1000
        )
        | extract_tag_from_llm_output.bind(tags="comparison_results")  # type: ignore
    )

    # Combine Wikipedia and FAISS results
    all_results = {
        "wikipedia_results": state.current_turn.filtered_search_results,
        "faiss_results": state.current_turn.faiss_results
    }
    
    comparison = await comparison_chain.ainvoke({
        "dlg": state,
        "results": all_results
    })
    
    state.current_turn.comparison_results = comparison

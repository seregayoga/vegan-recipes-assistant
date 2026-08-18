import time

import streamlit as st
from openai import OpenAI

from vegan_recipes.assistant import estimate_cost, parse_query_with_usage, recommend_with_usage
from vegan_recipes.config import get_settings
from vegan_recipes.db import log_interaction, migrate, save_feedback
from vegan_recipes.embeddings import get_embedder
from vegan_recipes.store import PostgresRecipeStore

st.set_page_config(page_title="Vegan Recipe Match", page_icon="🌿", layout="wide")
st.title("🌿 Vegan Recipe Match")
st.caption("Turn the ingredients you already have into a practical dinner plan.")
settings = get_settings()


@st.cache_resource
def recipe_store() -> PostgresRecipeStore:
    """Initialize the schema once per Streamlit process and reuse the embedder."""
    migrate()
    return PostgresRecipeStore(get_embedder())


def log_failure(raw_query: str, error: Exception) -> None:
    try:
        log_interaction(
            raw_query=raw_query,
            status="error",
            error=f"{type(error).__name__}: {error}",
            model=settings.openai_model if settings.openai_api_key else "deterministic",
        )
    except Exception:
        # Do not hide the original user-facing failure when monitoring is unavailable.
        pass


if "interaction_id" not in st.session_state:
    st.session_state.interaction_id = None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

raw = st.text_area(
    "What ingredients do you have?",
    placeholder="chickpeas, spinach, tomato — no peanuts",
    max_chars=settings.input_max_chars,
)
st.caption("Try: “aubergine, chickpeas and tomatoes, no peanuts”")

if st.button("Find recipes", type="primary", use_container_width=True):
    st.session_state.feedback_submitted = False
    st.session_state.interaction_id = None
    st.session_state.pop("result", None)
    progress = st.status("Preparing your recipe search…", expanded=True)
    started = time.perf_counter()
    try:
        client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        progress.update(label="Understanding your ingredients…", state="running")
        rewrite_started = time.perf_counter()
        parsed, rewrite_usage = parse_query_with_usage(raw, client)
        rewrite_ms = int((time.perf_counter() - rewrite_started) * 1000)
        progress.update(label="Searching 1,390 vegan recipes…", state="running")
        retrieval_started = time.perf_counter()
        candidates = recipe_store().search(parsed)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        progress.update(label="Preparing grounded recommendations…", state="running")
        llm_started = time.perf_counter()
        answer, recommendation_usage = recommend_with_usage(parsed, candidates, client)
        llm_ms = int((time.perf_counter() - llm_started) * 1000)
        input_tokens = rewrite_usage.input_tokens + recommendation_usage.input_tokens
        output_tokens = rewrite_usage.output_tokens + recommendation_usage.output_tokens
        fallbacks = [usage.error for usage in (rewrite_usage, recommendation_usage) if usage.error]
        interaction_id = log_interaction(
            raw_query=raw,
            parsed_query=parsed.model_dump(),
            candidates=[candidate.model_dump(mode="json") for candidate in candidates[:20]],
            selected_recipe_ids=[item.recipe_id for item in answer.recommendations],
            status=answer.status,
            error="; ".join(fallbacks) or None,
            model=settings.openai_model if client else "deterministic",
            rewrite_ms=rewrite_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
        )
        st.session_state.result = (answer, {candidate.recipe_id: candidate for candidate in candidates})
        st.session_state.interaction_id = interaction_id
        progress.update(label="Recommendations ready", state="complete", expanded=False)
    except ValueError as error:
        log_failure(raw, error)
        progress.update(label="Please adjust your ingredients", state="error")
        st.warning(str(error))
    except Exception as error:
        log_failure(raw, error)
        progress.update(label="Recipe search failed", state="error")
        st.error("The recipe service is temporarily unavailable. Check the database and model configuration.")
        st.caption(str(error))

if "result" in st.session_state:
    answer, by_id = st.session_state.result
    if answer.status != "ok" or not answer.recommendations:
        st.info(answer.message)
    else:
        st.subheader(answer.message)
        columns = st.columns(len(answer.recommendations))
        for column, item in zip(columns, answer.recommendations, strict=False):
            candidate = by_id[item.recipe_id]
            with column:
                st.markdown(f"### {candidate.title}")
                st.write(item.explanation)
                st.success("Matched: " + ", ".join(item.matched_ingredients))
                st.info("Still needed: " + (", ".join(item.missing_ingredients) or "pantry staples only"))
                st.write(item.preparation_summary)
                with st.expander("Full recipe"):
                    st.markdown("**Ingredients**")
                    st.markdown("\n".join(f"- {ingredient}" for ingredient in candidate.ingredients))
                    st.markdown("**Preparation**")
                    st.write(candidate.preparation)
                st.link_button("Open original recipe", str(candidate.href))
        st.write("Was this useful?")
        interaction_id = st.session_state.interaction_id
        submitted = st.session_state.feedback_submitted
        left, right = st.columns(2)
        if left.button("👍 Yes", disabled=submitted or not interaction_id):
            save_feedback(interaction_id, 1)
            st.session_state.feedback_submitted = True
            st.rerun()
        if right.button("👎 No", disabled=submitted or not interaction_id):
            save_feedback(interaction_id, -1)
            st.session_state.feedback_submitted = True
            st.rerun()
        if submitted:
            st.success("Thanks — your feedback was saved.")
        elif not interaction_id:
            st.caption("Feedback is unavailable because this interaction was not saved.")

st.divider()
st.caption("Recipes come from their linked sources. Verify ingredients and allergens yourself; this is not medical advice.")

"""
Phase 2: Autonomous Content Engine — LangGraph Orchestrator
-------------------------------------------------------------
A LangGraph state machine that autonomously produces an opinionated social post.

Node flow:
    decide_search  →  web_search  →  draft_post

Design decisions:
- TypedDict state: strict, inspectable, avoids dict key typos.
- Groq default (free tier, fast); swap to OpenAI by setting LLM_PROVIDER=openai in .env.
- mock_searxng_search is a real @tool so LangGraph/LangChain tool-calling wires correctly.
- Structured output via JSON mode + Pydantic v2 model ensures exact schema on output.
- Each node logs its input/output so execution is fully traceable.
"""

import json
import logging
import os
import sys
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase2_graph")

# ---------------------------------------------------------------------------
# LLM Factory — swap provider via env var
# ---------------------------------------------------------------------------
def _build_llm(json_mode: bool = False) -> Any:
    """
    Returns a LangChain chat model.
    Set LLM_PROVIDER=openai in .env to use OpenAI; defaults to Groq.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0.8,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        logger.info("LLM: OpenAI / %s", kwargs["model"])
        return ChatOpenAI(**kwargs)

    else:  # groq (default)
        from langchain_groq import ChatGroq
        kwargs = {
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "temperature": 0.8,
            "api_key": os.getenv("GROQ_API_KEY"),
        }
        if json_mode:
            # Groq supports json_mode via model_kwargs
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        logger.info("LLM: Groq / %s", kwargs["model"])
        return ChatGroq(**kwargs)


# ---------------------------------------------------------------------------
# Mock Search Tool
# ---------------------------------------------------------------------------
MOCK_NEWS_DB: dict[str, str] = {
    "crypto":    "Bitcoin hits new all-time high amid regulatory ETF approvals.",
    "bitcoin":   "Bitcoin hits new all-time high amid regulatory ETF approvals.",
    "ai":        "OpenAI launches o3 reasoning model, beating human benchmarks on PhD-level science.",
    "openai":    "OpenAI launches o3 reasoning model, beating human benchmarks on PhD-level science.",
    "markets":   "Fed signals rate cuts as inflation cools to 2.1%; S&P 500 rallies 3%.",
    "rates":     "Fed signals rate cuts as inflation cools to 2.1%; S&P 500 rallies 3%.",
    "tech":      "Microsoft and Google race to embed AI agents into enterprise software stacks.",
    "elon":      "Elon Musk's xAI raises $6B Series B to accelerate Grok development.",
    "privacy":   "EU fines Meta €1.3B for GDPR violations; activists call for stricter enforcement.",
    "climate":   "UN report: 2024 was the hottest year on record; tipping points accelerating.",
    "stocks":    "Nvidia stock surges 8% after record data-centre revenue tops $30B quarterly.",
    "trading":   "Quant hedge funds outperform traditional managers for third consecutive year.",
}

FALLBACK_HEADLINE = "Tech sector uncertainty grows as global macro headwinds persist."


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG search engine.
    Returns a hardcoded recent headline based on keywords in the query.
    In production this would call a real SearXNG or Brave Search API.
    """
    query_lower = query.lower()
    for keyword, headline in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            logger.info("mock_searxng_search(%r) → %r", query, headline)
            return headline
    logger.info("mock_searxng_search(%r) → fallback headline", query)
    return FALLBACK_HEADLINE


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------
class PostOutput(BaseModel):
    bot_id: str = Field(description="ID of the bot authoring the post.")
    topic: str = Field(description="One-line topic the post is about.")
    post_content: str = Field(
        description="The social media post, max 280 characters, highly opinionated."
    )


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_result: str
    output: dict  # final PostOutput as dict


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
def node_decide_search(state: GraphState) -> GraphState:
    """
    Node 1 — Decide Search
    The LLM reads the bot persona and decides what topic to post about,
    then formulates a concise search query.
    """
    logger.info("[Node 1: decide_search] Starting for bot_id=%s", state["bot_id"])

    llm = _build_llm(json_mode=False)

    system_prompt = (
        "You are the following social media bot persona:\n\n"
        f"{state['persona']}\n\n"
        "Decide ONE trending topic you want to post about today that perfectly aligns with "
        "your worldview. Respond with ONLY a short search engine query (5 words max). "
        "No explanation. No quotes."
    )

    response = llm.invoke([SystemMessage(content=system_prompt)])
    search_query = response.content.strip().strip('"').strip("'")

    logger.info("[Node 1: decide_search] Query decided: %r", search_query)
    return {**state, "search_query": search_query}


def node_web_search(state: GraphState) -> GraphState:
    """
    Node 2 — Web Search
    Executes mock_searxng_search with the query from Node 1.
    """
    logger.info("[Node 2: web_search] Searching for: %r", state["search_query"])
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    logger.info("[Node 2: web_search] Result: %r", result)
    return {**state, "search_result": result}


def node_draft_post(state: GraphState) -> GraphState:
    """
    Node 3 — Draft Post
    Uses persona + search context to generate a ≤280-char opinionated post.
    Returns strict JSON matching PostOutput schema.
    """
    logger.info("[Node 3: draft_post] Drafting post for bot_id=%s", state["bot_id"])

    llm = _build_llm(json_mode=True)

    system_prompt = (
        "You are the following social media bot persona:\n\n"
        f"{state['persona']}\n\n"
        "RULES:\n"
        "1. Write a single opinionated social media post about the context provided.\n"
        "2. The post MUST be ≤280 characters.\n"
        "3. Use the persona's authentic voice — aggressive, jargon-heavy, or cynical as appropriate.\n"
        "4. Respond ONLY with a valid JSON object. No markdown. No explanation.\n"
        "5. JSON schema:\n"
        '   {"bot_id": "<string>", "topic": "<string>", "post_content": "<string>"}\n'
    )

    user_prompt = (
        f"Bot ID: {state['bot_id']}\n"
        f"Search query you chose: {state['search_query']}\n"
        f"Search result / headline: {state['search_result']}\n\n"
        "Now write the post."
    )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    raw = response.content.strip()
    logger.info("[Node 3: draft_post] Raw LLM response: %s", raw)

    try:
        parsed = json.loads(raw)
        # Enforce 280-char limit silently
        if len(parsed.get("post_content", "")) > 280:
            parsed["post_content"] = parsed["post_content"][:277] + "…"
        # Validate with Pydantic
        validated = PostOutput(**parsed)
        output = validated.model_dump()
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("[Node 3: draft_post] JSON parse failed: %s", exc)
        # Graceful fallback — surface the error without crashing the graph
        output = {
            "bot_id": state["bot_id"],
            "topic": state["search_query"],
            "post_content": f"[PARSE ERROR] {raw[:240]}",
        }

    logger.info("[Node 3: draft_post] Final output: %s", json.dumps(output, indent=2))
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_content_graph() -> Any:
    """Compile and return the LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("decide_search", node_decide_search)
    builder.add_node("web_search",    node_web_search)
    builder.add_node("draft_post",    node_draft_post)

    builder.add_edge(START,           "decide_search")
    builder.add_edge("decide_search", "web_search")
    builder.add_edge("web_search",    "draft_post")
    builder.add_edge("draft_post",    END)

    graph = builder.compile()
    logger.info("LangGraph compiled successfully.")
    return graph


# ---------------------------------------------------------------------------
# Console Test Harness
# ---------------------------------------------------------------------------
BOT_PERSONAS_P2: dict[str, str] = {
    "bot_a_tech_maximalist": (
        "I believe AI and crypto will solve all human problems. I am highly optimistic "
        "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
    ),
    "bot_b_doomer_skeptic": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
    ),
    "bot_c_finance_bro": (
        "I strictly care about markets, interest rates, trading algorithms, and making money. "
        "I speak in finance jargon and view everything through the lens of ROI."
    ),
}


def run_graph_for_bot(bot_id: str, persona: str) -> dict:
    graph = build_content_graph()
    initial_state: GraphState = {
        "bot_id":        bot_id,
        "persona":       persona,
        "search_query":  "",
        "search_result": "",
        "output":        {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["output"]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 2 — AUTONOMOUS CONTENT ENGINE TEST")
    print("=" * 70)

    for bot_id, persona in BOT_PERSONAS_P2.items():
        print(f"\n▶ Running graph for: {bot_id}")
        result = run_graph_for_bot(bot_id, persona)
        print("\nFINAL JSON OUTPUT:")
        print(json.dumps(result, indent=2))
        print("-" * 70)

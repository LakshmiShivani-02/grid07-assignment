"""
Phase 1: Vector-Based Persona Routing
--------------------------------------
Routes incoming posts to semantically relevant bots using ChromaDB + sentence-transformers.
Each bot persona is embedded and stored; incoming posts are matched via cosine similarity.

Design decisions:
- ChromaDB in-memory mode: no server setup needed, portable, deterministic for tests.
- sentence-transformers (all-MiniLM-L6-v2): fast, lightweight, good cosine alignment.
- Threshold default 0.85 is intentionally strict; tweak down to 0.30-0.40 for MiniLM.
  (MiniLM cosine scores rarely exceed 0.7 for non-identical text — see README note.)
"""

import logging
import sys
from typing import Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase1_router")

# ---------------------------------------------------------------------------
# Bot Persona Definitions
# ---------------------------------------------------------------------------
BOT_PERSONAS: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Router Class
# ---------------------------------------------------------------------------
class PersonaRouter:
    """
    Embeds bot personas into ChromaDB and routes incoming posts
    to the most semantically relevant bots via cosine similarity.
    """

    COLLECTION_NAME = "bot_personas"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        logger.info("Initializing PersonaRouter…")
        self._model = SentenceTransformer(self.EMBEDDING_MODEL)
        # In-memory ChromaDB — no persistence needed for this assignment
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection: Optional[chromadb.Collection] = None
        self._build_persona_store()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        """Return a normalised embedding vector for a single text string."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _build_persona_store(self) -> None:
        """
        Embed every persona and upsert into ChromaDB.
        Idempotent: safe to call multiple times.
        """
        logger.info("Building persona vector store…")

        # Drop and recreate for clean slate on each run
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet

        self._collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # tell ChromaDB to use cosine distance
        )

        for bot_id, persona_text in BOT_PERSONAS.items():
            embedding = self._embed(persona_text)
            self._collection.add(
                ids=[bot_id],
                embeddings=[embedding],
                documents=[persona_text],
                metadatas=[{"bot_id": bot_id}],
            )
            logger.info("Stored persona for %s", bot_id)

        logger.info("Persona store ready with %d entries.", len(BOT_PERSONAS))

    @staticmethod
    def _chroma_distance_to_cosine_similarity(distance: float) -> float:
        """
        ChromaDB cosine *distance* = 1 − cosine_similarity.
        Convert back so callers reason in similarity (higher = more similar).
        """
        return round(1.0 - distance, 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route_post_to_bots(
        self,
        post_content: str,
        threshold: float = 0.85,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Embed `post_content` and return bots whose cosine similarity
        to the post exceeds `threshold`.

        Args:
            post_content: The raw post text to route.
            threshold:    Minimum cosine similarity to include a bot (0–1).
                          NOTE: all-MiniLM-L6-v2 rarely produces scores > 0.6
                          for non-identical sentences; try 0.30 for realistic matching.
            top_k:        Max candidates to retrieve from ChromaDB before filtering.

        Returns:
            List of dicts: [{"bot_id": str, "similarity": float, "persona": str}]
            Sorted descending by similarity. Empty list if no match.
        """
        if not post_content or not post_content.strip():
            logger.warning("route_post_to_bots called with empty post_content.")
            return []

        logger.info("Routing post: %r  (threshold=%.2f)", post_content[:80], threshold)

        post_embedding = self._embed(post_content)

        results = self._collection.query(
            query_embeddings=[post_embedding],
            n_results=min(top_k, len(BOT_PERSONAS)),
            include=["documents", "metadatas", "distances"],
        )

        matched_bots: list[dict] = []

        distances: list[float] = results["distances"][0]
        documents: list[str]   = results["documents"][0]
        metadatas: list[dict]  = results["metadatas"][0]

        for distance, persona_text, meta in zip(distances, documents, metadatas):
            similarity = self._chroma_distance_to_cosine_similarity(distance)
            bot_id = meta["bot_id"]

            logger.info(
                "  %-40s  similarity=%.4f  (distance=%.4f)  %s",
                bot_id,
                similarity,
                distance,
                "✓ MATCHED" if similarity >= threshold else "✗ below threshold",
            )

            if similarity >= threshold:
                matched_bots.append(
                    {
                        "bot_id": bot_id,
                        "similarity": similarity,
                        "persona": persona_text,
                    }
                )

        matched_bots.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info("Routing complete. %d bot(s) matched.", len(matched_bots))
        return matched_bots


# ---------------------------------------------------------------------------
# Console Test Harness
# ---------------------------------------------------------------------------
def _run_tests(router: PersonaRouter) -> None:
    test_posts = [
        # Should hit bot_a
        (
            "OpenAI just released a new model that might replace junior developers.",
            0.30,
        ),
        # Should hit bot_b
        (
            "Big Tech is collecting your data and selling your privacy to the highest bidder.",
            0.30,
        ),
        # Should hit bot_c
        (
            "The Fed just raised interest rates again. Bond yields are surging.",
            0.30,
        ),
        # Ambiguous — might hit multiple
        (
            "AI-powered trading algorithms are dominating Wall Street returns this quarter.",
            0.30,
        ),
        # Edge case: very short / low-signal text
        (
            "Hello world.",
            0.30,
        ),
    ]

    print("\n" + "=" * 70)
    print("PHASE 1 — ROUTING TEST RESULTS")
    print("=" * 70)

    for post, threshold in test_posts:
        print(f"\nPOST    : {post}")
        print(f"THRESHOLD: {threshold}")
        matches = router.route_post_to_bots(post, threshold=threshold)
        if matches:
            for m in matches:
                print(f"  → {m['bot_id']}  (similarity={m['similarity']:.4f})")
        else:
            print("  → No bots matched above threshold.")
        print("-" * 70)


if __name__ == "__main__":
    router = PersonaRouter()
    _run_tests(router)

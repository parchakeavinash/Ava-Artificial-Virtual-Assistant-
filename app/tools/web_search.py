import re
import datetime as dt
from firecrawl import Firecrawl
from app.config.settings import settings


def web_search(query: str, limit: int = 3) -> str:
    """
    Search the live web using Firecrawl for current news, facts, documentation, or real-time information.

    Args:
        query: Search keywords or question.
        limit: Number of search results to return (default: 3 for speed).

    Returns:
        Clean summarized search results.
    """
    api_key = settings.FIRECRAWL_API_KEY

    if not api_key:
        return "Error: FIRECRAWL_API_KEY is not configured."

    # Guard: If model asks for "latest" but appends an old training cutoff year (e.g. 2023 or 2024),
    # remove the stale year so Firecrawl returns true fresh news.
    cleaned_query = query
    temporal_words = ["latest", "recent", "new", "news", "today", "current"]
    if any(w in query.lower() for w in temporal_words):
        cleaned_query = re.sub(r"\b(202[0-4])\b", "", query).strip()
        cleaned_query = re.sub(r"\s+", " ", cleaned_query)

    try:
        print(f"[TOOL: Firecrawl] Searching web for: {cleaned_query!r} (raw: {query!r})")
        firecrawl = Firecrawl(api_key=api_key)
        results = firecrawl.search(cleaned_query, limit=limit)

        if not results or not results.web:
            return f"No search results found for: {query}"

        output = []
        for i, item in enumerate(results.web[:limit], 1):
            title = getattr(item, "title", "No Title")
            description = getattr(item, "description", "") or getattr(item, "markdown", "")
            # Truncate each snippet to ~300 chars for fast voice reasoning
            clean_desc = (description[:300] + "...") if len(description) > 300 else description
            output.append(f"[{i}] {title}\nSummary: {clean_desc}\n")

        formatted_result = "\n".join(output)
        print(f"[TOOL: Firecrawl] Retrieved {len(results.web[:limit])} search results.")
        return formatted_result

    except Exception as e:
        print(f"[TOOL: Firecrawl] Error: {e}")
        return f"Error performing web search: {str(e)}"

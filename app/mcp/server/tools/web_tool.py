from firecrawl import Firecrawl
from app.config.setting import settings


def web_search(query: str, limit: int = 5) -> str:
    """
    Search the live web using Firecrawl.

    Args:
        query: Search query string.
        limit: Number of search results to return (default: 5).

    Returns:
        Formatted text containing search results with titles, URLs, and summaries.
    """
    api_key = settings.FIRECRAWL_API_KEY

    if not api_key:
        return "Error: FIRECRAWL_API_KEY is not configured."

    try:
        firecrawl = Firecrawl(api_key=api_key)
        results = firecrawl.search(query, limit=limit)

        if not results or not results.web:
            return f"No search results found for: {query}"

        output = []
        for i, item in enumerate(results.web[:limit], 1):
            title = getattr(item, "title", "No Title")
            url = getattr(item, "url", "")
            description = getattr(item, "description", "") or getattr(item, "markdown", "")
            output.append(f"[{i}] {title}\nURL: {url}\nSummary: {description}\n")

        return "\n".join(output)

    except Exception as e:
        return f"Error performing web search: {str(e)}"
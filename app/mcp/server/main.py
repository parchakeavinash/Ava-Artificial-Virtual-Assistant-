from mcp.server import MCPServer
from app.mcp.server.tools.web_tool import web_search

mcp = MCPServer(
    name="AVA Voice Agent MCP Server"
)


@mcp.tool()
def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the live web for up-to-date information, news, documentation, or facts.

    Use this tool whenever the user asks about recent events, facts, news,
    weather, stock prices, or any topic that requires fresh internet search.

    Args:
        query: The search query string.
        limit: Number of results to return (default is 5).

    Returns:
        Structured search results with titles, links, and content summaries.
    """
    return web_search(query, limit=limit)


if __name__ == "__main__":
    mcp.run()

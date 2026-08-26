from mcp.server import MCPServer

from app.tools.email_tool import (
    delete_email_confirmed,
    find_email_by_subject,
    read_inbox,
    send_email,
)
from app.tools.notion_tool import (
    append_blocks,
    create_page,
    delete_page,
    get_page,
    list_databases,
    query_database,
    search_pages,
    update_page,
)
from app.tools.web_search import web_search

mcp = MCPServer(name="AVA Unified Tools MCP Server")


# =========================================================
# WEB SEARCH TOOLS
# =========================================================
@mcp.tool()
def web_search_tool(query: str, limit: int = 3) -> str:
    """
    Search the live web using Firecrawl for real-time information, facts, or news.
    """
    return web_search(query=query, limit=limit)


# =========================================================
# EMAIL TOOLS
# =========================================================
@mcp.tool()
def send_email_tool(to: str, subject: str, body: str, is_html: bool = False) -> str:
    """
    Send an email via SMTP.
    """
    return send_email(to=to, subject=subject, body=body, is_html=is_html)


@mcp.tool()
def read_inbox_tool(limit: int = 5, unread_only: bool = False) -> str:
    """
    Read latest emails from inbox.
    """
    return read_inbox(limit=limit, unread_only=unread_only)


@mcp.tool()
def find_email_by_subject_tool(subject_keyword: str) -> str:
    """
    Search the inbox for emails matching a subject keyword and return
    a confirmation prompt before deleting.
    """
    return find_email_by_subject(subject_keyword=subject_keyword)


@mcp.tool()
def delete_email_confirmed_tool(subject_keyword: str) -> str:
    """
    Permanently delete emails matching the subject keyword after confirmation.
    """
    return delete_email_confirmed(subject_keyword=subject_keyword)


# =========================================================
# NOTION TOOLS
# =========================================================
@mcp.tool()
def search_notion_pages_tool(query: str = "", limit: int = 10) -> str:
    """
    Search Notion pages. Query filters by title; empty query lists recent pages.
    """
    return search_pages(query=query, limit=limit)


@mcp.tool()
def get_notion_page_tool(page_id: str) -> str:
    """
    Get a Notion page by its ID (UUID or 32-char ID). Returns properties and content blocks.
    """
    return get_page(page_id=page_id)


@mcp.tool()
def create_notion_page_tool(parent_id: str, title: str, content: str = "", parent_type: str = "page") -> str:
    """
    Create a new Notion page under a parent page or database.
    """
    return create_page(parent_id=parent_id, title=title, content=content, parent_type=parent_type)


@mcp.tool()
def update_notion_page_tool(page_id: str, title: str = "") -> str:
    """
    Update an existing Notion page's title.
    """
    return update_page(page_id=page_id, title=title)


@mcp.tool()
def append_notion_blocks_tool(page_id: str, content: str) -> str:
    """
    Append paragraphs / text to a Notion page.
    """
    return append_blocks(page_id=page_id, content=content)


@mcp.tool()
def delete_notion_page_tool(page_id: str) -> str:
    """
    Archive (soft-delete) a Notion page by its ID.
    """
    return delete_page(page_id=page_id)


@mcp.tool()
def list_notion_databases_tool(limit: int = 20) -> str:
    """
    List Notion databases the integration has access to.
    """
    return list_databases(limit=limit)


@mcp.tool()
def query_notion_database_tool(database_id: str, limit: int = 20) -> str:
    """
    Query a Notion database and return its rows as a summary list.
    """
    return query_database(database_id=database_id, limit=limit)


if __name__ == "__main__":
    mcp.run()

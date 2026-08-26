from mcp.server import MCPServer

from app.tools.email_tool import (
    delete_email_confirmed,
    find_email_by_subject,
    read_inbox,
    send_email,
)
from app.tools.web_search import web_search

mcp = MCPServer(name="AVA Unified Tools MCP Server")



@mcp.tool()
def web_search_tool(query: str, limit: int = 3) -> str:
    """
    Search the live web using Firecrawl for real-time information, facts, or news.
    """
    return web_search(query=query, limit=limit)


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


if __name__ == "__main__":
    mcp.run()

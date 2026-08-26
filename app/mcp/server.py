from mcp.server import MCPServer

from app.tools.calculator_tool import calculate
from app.tools.diary_tool import (
    add_diary_entry,
    delete_diary_entry,
    read_recent_diary,
    search_diary,
)
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
from app.tools.task_tool import (
    complete_task,
    create_task,
    delete_task,
    get_task,
    list_pending_tasks,
    update_task,
)
from app.tools.web_search import web_search

mcp = MCPServer(name="AVA Unified Tools MCP Server")

CORE_TOOLS = [
    # Math
    calculate,
    # Web
    web_search,
    # Email
    send_email,
    read_inbox,
    find_email_by_subject,
    delete_email_confirmed,
    # Notion
    search_pages,
    get_page,
    create_page,
    update_page,
    append_blocks,
    delete_page,
    list_databases,
    query_database,
    # Task Manager
    create_task,
    update_task,
    get_task,
    list_pending_tasks,
    complete_task,
    delete_task,
    # Diary / Ideas
    add_diary_entry,
    read_recent_diary,
    search_diary,
    delete_diary_entry,
]

for tool_fn in CORE_TOOLS:
    mcp.add_tool(tool_fn)


if __name__ == "__main__":
    mcp.run()

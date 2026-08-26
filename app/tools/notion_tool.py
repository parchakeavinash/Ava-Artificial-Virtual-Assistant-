import json
import os
import httpx
from app.config.settings import settings

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _get_api_key() -> str:
    return settings.NOTION_API_KEY or os.getenv("NOTION_API_KEY", "")


def _headers() -> dict:
    key = _get_api_key()
    if not key:
        raise ValueError("NOTION_API_KEY is not configured in .env")
    return {
        "Authorization": f"Bearer {key}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _extract_title(properties: dict) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in prop.get("title", []))
    return ""


def _summarize_page(page: dict) -> str:
    props = page.get("properties", {})
    title = _extract_title(props)
    return f"[{page.get('id')}] {title} (last edited: {page.get('last_edited_time', '?')})"


def _page_to_text(page: dict, include_children: bool = True) -> str:
    lines = [
        f"ID: {page.get('id')}",
        f"URL: {page.get('url', '')}",
        f"Created: {page.get('created_time', '')}",
        f"Last edited: {page.get('last_edited_time', '')}",
        f"Archived: {page.get('archived', False)}",
    ]
    props = page.get("properties", {})
    if props:
        lines.append("\nProperties:")
        for name, value in props.items():
            vtype = value.get("type")
            display = ""
            if vtype == "title":
                display = "".join(t.get("plain_text", "") for t in value.get("title", []))
            elif vtype == "rich_text":
                display = "".join(t.get("plain_text", "") for t in value.get("rich_text", []))
            elif vtype == "select":
                sel = value.get("select") or {}
                display = sel.get("name", "")
            elif vtype == "multi_select":
                display = ", ".join(s.get("name", "") for s in value.get("multi_select", []))
            elif vtype == "status":
                st = value.get("status") or {}
                display = st.get("name", "")
            elif vtype == "date":
                d = value.get("date") or {}
                display = d.get("start", "")
            elif vtype == "checkbox":
                display = "Yes" if value.get("checkbox") else "No"
            elif vtype == "number":
                display = str(value.get("number"))
            elif vtype == "url":
                display = value.get("url", "")
            elif vtype == "email":
                display = value.get("email", "")
            elif vtype == "phone_number":
                display = value.get("phone_number", "")
            else:
                display = json.dumps(value.get(vtype)) if vtype else ""
            lines.append(f"  {name} ({vtype}): {display}")

    if include_children:
        children = page.get("_children_blocks", [])
        if children:
            lines.append("\nContent:")
            for block in children:
                btype = block.get("type")
                bdata = block.get(btype, {})
                rich = bdata.get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich)
                if btype == "heading_1":
                    lines.append(f"# {text}")
                elif btype == "heading_2":
                    lines.append(f"## {text}")
                elif btype == "heading_3":
                    lines.append(f"### {text}")
                elif btype == "bulleted_list_item":
                    lines.append(f"- {text}")
                elif btype == "numbered_list_item":
                    lines.append(f"1. {text}")
                elif btype == "to_do":
                    lines.append(f"[{'x' if bdata.get('checked') else ' '}] {text}")
                elif btype == "code":
                    lang = bdata.get("language", "")
                    lines.append(f"```{lang}\n{text}\n```")
                elif btype == "quote":
                    lines.append(f"> {text}")
                else:
                    if text:
                        lines.append(text)
    return "\n".join(lines)


def search_pages(query: str = "", limit: int = 10) -> str:
    """
    Search Notion pages. Query filters by title; empty query lists recent pages.
    """
    try:
        headers = _headers()
        payload = {"page_size": min(max(limit, 1), 100)}
        if query:
            payload["query"] = query
        resp = httpx.post(f"{NOTION_API_BASE}/search", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        results = resp.json().get("results", [])
        if not results:
            return f"No pages found for query={query!r}."
        return "\n".join(_summarize_page(p) for p in results)
    except Exception as e:
        return f"Error searching Notion pages: {e}"


def get_page(page_id: str) -> str:
    """
    Get a Notion page by its ID (UUID or 32-char ID). Returns properties and content blocks.
    """
    try:
        headers = _headers()
        clean_id = page_id.strip()
        if "-" not in clean_id and len(clean_id) == 32:
            clean_id = f"{clean_id[:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:]}"

        page_resp = httpx.get(f"{NOTION_API_BASE}/pages/{clean_id}", headers=headers, timeout=30)
        if page_resp.status_code != 200:
            return f"Notion API error {page_resp.status_code}: {page_resp.text}"
        page = page_resp.json()

        children_lines = []
        cursor = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            c_resp = httpx.get(
                f"{NOTION_API_BASE}/blocks/{clean_id}/children",
                headers=headers,
                params=params,
                timeout=30,
            )
            if c_resp.status_code != 200:
                break
            c_data = c_resp.json()
            children_lines.extend(c_data.get("results", []))
            if not c_data.get("has_more"):
                break
            cursor = c_data.get("next_cursor")
            if not cursor:
                break
        page["_children_blocks"] = children_lines
        return _page_to_text(page, include_children=True)
    except Exception as e:
        return f"Error getting Notion page: {e}"


def _resolve_parent_id(parent_id_or_name: str = "") -> tuple[str, str]:
    """
    Resolve a parent ID or page name to a valid Notion UUID.
    If empty, returns the first accessible page ID from search.
    Returns (parent_id, parent_type).
    """
    clean = parent_id_or_name.strip() if parent_id_or_name else ""

    # If already a valid UUID (32 or 36 chars)
    if clean:
        if len(clean) == 36 and "-" in clean:
            return clean, "page"
        if len(clean) == 32 and "-" not in clean:
            return f"{clean[:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:]}", "page"

    headers = _headers()

    # If parent name was given, search by that name
    if clean and clean.lower() not in ("default", "none", "root", "workspace"):
        resp = httpx.post(f"{NOTION_API_BASE}/search", headers=headers, json={"query": clean, "page_size": 5}, timeout=30)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for item in results:
                item_id = item.get("id")
                obj_type = item.get("object", "page")
                if item_id:
                    return item_id, ("database" if obj_type == "database" else "page")

    # Otherwise fallback to the first accessible page
    resp = httpx.post(f"{NOTION_API_BASE}/search", headers=headers, json={"page_size": 1}, timeout=30)
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        if results:
            item = results[0]
            item_id = item.get("id")
            obj_type = item.get("object", "page")
            return item_id, ("database" if obj_type == "database" else "page")

    raise ValueError("No accessible parent page found in Notion workspace. Please invite the integration to a page.")


def create_page(title: str, content: str = "", parent_id: str = "", parent_type: str = "page") -> str:
    """
    Create a new Notion page with a title and content.

    Args:
        title: Title of the new page.
        content: Body text/markdown content for the page.
        parent_id: (Optional) ID or name of parent page/database. If empty, automatically adds to your Notion workspace.
        parent_type: 'page' or 'database' (default: 'page').

    Returns:
        Confirmation message with page ID, title, and Notion URL.
    """
    try:
        headers = _headers()
        resolved_parent, resolved_type = _resolve_parent_id(parent_id)

        parent = {"page_id": resolved_parent} if resolved_type == "page" else {"database_id": resolved_parent}
        properties = {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}],
            }
        }

        children = []
        if content:
            for para in content.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if para.startswith("# "):
                    children.append({
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": [{"type": "text", "text": {"content": para[2:]}}]},
                    })
                elif para.startswith("## "):
                    children.append({
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": para[3:]}}]},
                    })
                elif para.startswith("### "):
                    children.append({
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {"rich_text": [{"type": "text", "text": {"content": para[4:]}}]},
                    })
                elif para.startswith("- ") or para.startswith("* "):
                    children.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": para[2:]}}]},
                    })
                else:
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": para}}]},
                    })

        payload = {"parent": parent, "properties": properties}
        if children:
            payload["children"] = children

        resp = httpx.post(f"{NOTION_API_BASE}/pages", headers=headers, json=payload, timeout=30)
        if resp.status_code not in (200, 201):
            return f"Notion API error {resp.status_code}: {resp.text}"
        page = resp.json()
        return f"Successfully created Notion page: '{title}' (ID: {page.get('id')})\nURL: {page.get('url', '')}"
    except Exception as e:
        return f"Error creating Notion page: {e}"



def update_page(page_id: str, title: str = "") -> str:
    """
    Update an existing Notion page's title.
    """
    if not title:
        return "No updates supplied. Provide a new title."
    try:
        headers = _headers()
        clean_id = page_id.strip()
        payload = {
            "properties": {
                "title": {"title": [{"type": "text", "text": {"content": title}}]},
            }
        }
        resp = httpx.patch(f"{NOTION_API_BASE}/pages/{clean_id}", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        return f"Page {clean_id} updated. New title: {title}"
    except Exception as e:
        return f"Error updating Notion page: {e}"


def append_blocks(page_id: str, content: str) -> str:
    """
    Append paragraphs / text to a Notion page.
    """
    try:
        headers = _headers()
        clean_id = page_id.strip()
        children = []
        for para in content.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("# "):
                children.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": para[2:]}}]},
                })
            elif para.startswith("## "):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": para[3:]}}]},
                })
            elif para.startswith("- "):
                children.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": para[2:]}}]},
                })
            else:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": para}}]},
                })

        if not children:
            return "No content to append."

        resp = httpx.patch(f"{NOTION_API_BASE}/blocks/{clean_id}/children", headers=headers, json={"children": children}, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        return f"Appended {len(children)} block(s) to page {clean_id}."
    except Exception as e:
        return f"Error appending blocks to Notion page: {e}"


def delete_page(page_id: str) -> str:
    """
    Archive (soft-delete) a Notion page by its ID.
    """
    try:
        headers = _headers()
        clean_id = page_id.strip()
        resp = httpx.patch(f"{NOTION_API_BASE}/pages/{clean_id}", headers=headers, json={"archived": True}, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        return f"Page {clean_id} archived (moved to trash)."
    except Exception as e:
        return f"Error archiving Notion page: {e}"


def list_databases(limit: int = 20) -> str:
    """
    List Notion databases the integration has access to.
    """
    try:
        headers = _headers()
        payload = {"filter": {"value": "database", "property": "object"}, "page_size": min(max(limit, 1), 100)}
        resp = httpx.post(f"{NOTION_API_BASE}/search", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        results = resp.json().get("results", [])
        if not results:
            return "No databases found. Make sure the integration is connected to your page/database in Notion."
        lines = []
        for db in results:
            title_parts = db.get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_parts)
            if not title:
                title = _extract_title(db.get("properties", {}))
            lines.append(f"[{db.get('id')}] {title}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing Notion databases: {e}"


def query_database(database_id: str, limit: int = 20) -> str:
    """
    Query a Notion database and return its rows as a summary list.
    """
    try:
        headers = _headers()
        clean_id = database_id.strip()
        payload = {"page_size": min(max(limit, 1), 100)}
        resp = httpx.post(f"{NOTION_API_BASE}/databases/{clean_id}/query", headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return f"Notion API error {resp.status_code}: {resp.text}"
        results = resp.json().get("results", [])
        if not results:
            return "Database is empty (or no rows match)."
        return "\n".join(_summarize_page(p) for p in results)
    except Exception as e:
        return f"Error querying Notion database: {e}"

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.config.settings import settings
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
from app.tools.user_context import set_current_user
from app.tools.web_search import web_search


# =========================================================
# LANGCHAIN TOOL ADAPTERS (Zero-boilerplate directly from app.tools)
# =========================================================
ALL_LANGCHAIN_TOOLS = [
    # Math
    tool(calculate),
    # Web
    tool(web_search),
    # Email
    tool(send_email),
    tool(read_inbox),
    tool(find_email_by_subject),
    tool(delete_email_confirmed),
    # Notion
    tool(search_pages),
    tool(get_page),
    tool(create_page),
    tool(update_page),
    tool(append_blocks),
    tool(delete_page),
    tool(list_databases),
    tool(query_database),
    # Task Manager
    tool(create_task),
    tool(update_task),
    tool(get_task),
    tool(list_pending_tasks),
    tool(complete_task),
    tool(delete_task),
    # Diary / Ideas
    tool(add_diary_entry),
    tool(read_recent_diary),
    tool(search_diary),
    tool(delete_diary_entry),
]

TOOL_MAP = {t.name: t for t in ALL_LANGCHAIN_TOOLS}

_SYSTEM_PROMPT = """You are Ava, an intelligent, helpful, and friendly voice AI assistant.

Your Capabilities & Tools:
1. `calculate`: Use for any math — salary hikes, percentages, arithmetic, compound interest.
2. `web_search`: Use for real-time news, weather, stock updates, or live facts.
3. `send_email` / `read_inbox` / `find_email_by_subject` / `delete_email_confirmed`: Full Gmail management.
   - ALWAYS call `find_email_by_subject` first when user asks to delete an email; only call `delete_email_confirmed` after explicit confirmation.
4. Notion Tools (`create_page`, `search_pages`, `get_page`, `update_page`, `append_blocks`, `delete_page`, `list_databases`, `query_database`):
   - When writing an article/note for Notion, generate full content then call `create_page`.
   - IMPORTANT FOR VOICE: Reply with a SHORT confirmation only — do NOT read the whole article aloud.
5. Task Manager (`create_task`, `update_task`, `get_task`, `list_pending_tasks`, `complete_task`, `delete_task`):
   - Use to manage to-dos and tasks for the user.
   - When creating: call `create_task(title=..., description=...)`.
   - When listing: call `list_pending_tasks()` and summarise count + first few titles.
   - When completing: call `complete_task(task_id=...)`.
   - When deleting: ALWAYS confirm with the user first before calling `delete_task`.
6. Diary / Ideas (`add_diary_entry`, `read_recent_diary`, `search_diary`, `delete_diary_entry`):
   - Use when the user wants to note down ideas, thoughts, or personal reflections.
   - When saving: call `add_diary_entry(content=..., title=...)`.
   - When reading: call `read_recent_diary()` and summarise the key ideas aloud.
   - When searching: call `search_diary(query=...)`.

Voice Response Guidelines:
- Keep answers SHORT, conversational, and natural — they will be spoken aloud by TTS.
- Never read raw IDs, UUIDs, URLs, markdown symbols, or code aloud.
- Speak warmly, clearly, and concisely.
"""


class AgentMetrics:
    """Tracks which LLM provider answered and fallback occurrences."""

    def __init__(self):
        self.last_provider: str = "Groq (Llama 3.3 70B)"
        self.groq_calls: int = 0
        self.gemini_fallback_calls: int = 0

    def record_groq(self):
        self.last_provider = "Groq (Llama 3.3 70B)"
        self.groq_calls += 1

    def record_gemini(self):
        self.last_provider = "Google Gemini (Fallback)"
        self.gemini_fallback_calls += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "last_provider": self.last_provider,
            "groq_calls": self.groq_calls,
            "gemini_fallback_calls": self.gemini_fallback_calls,
        }


agent_metrics = AgentMetrics()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()


from app.memory.manager import MemoryManager


class LangChainResilientAgent:
    """
    Multi-provider conversational agent with automatic failover and 3-layer cognitive memory:
    Primary: Groq (openai/gpt-oss-120b)
    Secondary: Google Gemini (gemini-3.5-flash-lite) via LangChain with_fallbacks.
    Memory: Short-Term (sliding window + summary) + Episodic (Gemini vectors) + Semantic (user facts).
    """

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id or settings.MEMORY_USER_ID

        self.primary_llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.7,
        )

        self.backup_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7,
        )

        # Grounded low-temp LLM for deterministic fact extraction & summarization
        self.extraction_llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

        # Bind tools to both models
        self.primary_with_tools = self.primary_llm.bind_tools(ALL_LANGCHAIN_TOOLS)
        self.backup_with_tools = self.backup_llm.bind_tools(ALL_LANGCHAIN_TOOLS)

        # Resilient fallback chain
        self.model_chain = self.primary_with_tools.with_fallbacks([self.backup_with_tools])

        # Cognitive Memory Manager
        self.memory = MemoryManager(user_id=self.user_id)
        self.messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]

    def respond(self, user_text: str, session_id: str = "default", user_id: str | None = None) -> str:
        """
        Send user input, execute any required tool calls in an agentic loop,
        and return the synthesized voice-friendly response.
        Loads short-term history, running summary, episodic context, and semantic facts.
        """
        uid = user_id or self.user_id
        set_current_user(uid)
        try:
            print(f"[AGENT: LangChain] User ({uid}@{session_id}) said: {user_text!r}")
        except Exception:
            pass

        # 1. Retrieve multi-layered memory context
        try:
            injected_prompts, history = self.memory.build_memory_context(
                user_text=user_text,
                session_id=session_id,
                user_id=uid,
            )
        except Exception as mem_err:
            print(f"[AGENT: Memory Warning] Failed to build context: {mem_err}")
            injected_prompts, history = [], []

        # 2. Build turn prompt payload
        turn_messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
        turn_messages.extend(injected_prompts)
        turn_messages.extend(history)
        turn_messages.append(HumanMessage(content=user_text))

        max_tool_iterations = 5
        iteration = 0

        while iteration < max_tool_iterations:
            iteration += 1

            # Step 1: Call LLM (Groq with Gemini Fallback)
            try:
                ai_response: AIMessage = self.model_chain.invoke(turn_messages)
                agent_metrics.record_groq()
            except Exception as primary_err:
                print(f"[AGENT: Fallback Triggered] Primary Groq error: {primary_err}")
                print(f"[AGENT: Fallback Triggered] Retrying on Google Gemini...")
                ai_response: AIMessage = self.backup_with_tools.invoke(turn_messages)
                agent_metrics.record_gemini()

            turn_messages.append(ai_response)

            # Step 2: Check for tool calls
            tool_calls = getattr(ai_response, "tool_calls", [])
            if not tool_calls:
                # Final response reached
                raw_text = _extract_text(ai_response.content) or "I completed your request."
                import re
                clean_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL).strip()
                final_text = clean_text or raw_text

                try:
                    print(f"[AGENT: LangChain] Final Voice Reply ({agent_metrics.last_provider}): {final_text!r}")
                except Exception:
                    pass

                # 3. Record turn in memory (persists to DB, auto-extracts facts, updates summary)
                try:
                    self.memory.record_turn(
                        user_text=user_text,
                        ai_text=final_text,
                        session_id=session_id,
                        user_id=uid,
                        extraction_llm=self.extraction_llm,
                    )
                except Exception as rec_err:
                    print(f"[AGENT: Memory Record Warning] {rec_err}")

                self.messages = turn_messages
                return final_text

            # Step 3: Execute tool calls
            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("args", {})
                tool_call_id = call.get("id", str(time.time()))

                try:
                    print(f"[AGENT: Tool Execution] Calling {tool_name} with args: {tool_args}")
                except Exception:
                    pass

                tool_fn = TOOL_MAP.get(tool_name)

                if tool_fn:
                    try:
                        tool_result = tool_fn.invoke(tool_args)
                    except Exception as err:
                        tool_result = f"Error executing {tool_name}: {err}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."

                # Append tool response to messages
                turn_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        return "I processed your request, but hit the tool execution limit."

    def create_episode(self, session_id: str = "default", user_id: str | None = None) -> dict | None:
        """Distills active conversation into an episodic memory."""
        return self.memory.create_episode(
            session_id=session_id,
            extraction_llm=self.extraction_llm,
            user_id=user_id or self.user_id,
        )

    def auto_distill_if_needed(self, session_id: str = "default", user_id: str | None = None) -> dict | None:
        """Automatically distills conversation if eligible."""
        return self.memory.auto_distill_if_needed(
            session_id=session_id,
            extraction_llm=self.extraction_llm,
            user_id=user_id or self.user_id,
        )

    def list_known_facts(self, user_id: str | None = None) -> list[dict]:
        """Lists persistent facts known about the user."""
        return self.memory.list_known_facts(user_id=user_id or self.user_id)

    def list_sessions(self, user_id: str | None = None) -> list[dict]:
        """Lists active conversation sessions."""
        return self.memory.list_sessions(user_id=user_id or self.user_id)

    def clear_history(self, session_id: str = "default", user_id: str | None = None):
        """Reset conversation session."""
        self.memory.clear_session(session_id=session_id, user_id=user_id or self.user_id)
        self.messages = [SystemMessage(content=_SYSTEM_PROMPT)]

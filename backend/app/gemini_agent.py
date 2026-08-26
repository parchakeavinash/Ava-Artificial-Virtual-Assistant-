from google import genai
from google.genai import types

from app.config import settings
from app.web_search import web_search


_SYSTEM_PROMPT = """You are Ava, an intelligent, helpful voice AI assistant.

Capabilities:
- You have access to the `web_search` tool powered by Firecrawl.
- Use `web_search` whenever the user asks about real-time news, current events, recent developments, live facts, weather, stock prices, or specific topics that require fresh web information.
- Synthesize search results into a clean, natural, human summary.

Voice Guidelines:
- Keep your answers CONCISE and conversational — they will be spoken aloud.
- Never read raw URLs, markdown asterisks, bullet points, or code syntax aloud.
- Speak naturally, warmly, and confidently.
"""


class GeminiAgent:
    """
    Wraps Google Gemini with multi-turn conversation history and
    automatic tool calling (MCP / Firecrawl Web Search).
    """

    def __init__(self):

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        # Multi-turn chat session with tools
        self.chat = self.client.chats.create(
            model=settings.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.7,
                tools=[web_search],
            ),
        )

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------

    def respond(self, user_text: str) -> str:
        """
        Send user_text to Gemini. Gemini can automatically invoke
        tools (like Firecrawl web search) if needed, then return
        the final synthesized voice answer.
        """

        print(f"[GEMINI] Sending to LLM: {user_text!r}")

        response = self.chat.send_message(user_text)

        reply_text = response.text.strip() if response.text else "I could not find an answer to that."

        print(f"[GEMINI] Final Response: {reply_text!r}")

        return reply_text

    def clear_history(self):
        """Reset conversation session."""
        self.chat = self.client.chats.create(
            model=settings.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.7,
                tools=[web_search],
            ),
        )


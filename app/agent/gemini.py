from google import genai
from google.genai import types

from app.config.settings import settings
from app.tools.email_tool import (
    delete_email_confirmed,
    find_email_by_subject,
    read_inbox,
    send_email,
)
from app.tools.web_search import web_search

_SYSTEM_PROMPT = """You are Ava, an intelligent, helpful, and friendly voice AI assistant.

Your Capabilities & Tools:
1. `web_search`: Use whenever the user asks for real-time news, live facts, weather, stock market updates, or current events.
2. `send_email`: Send an email with recipient, subject, and body.
3. `read_inbox`: Read recent unread or all emails from Gmail.
4. `find_email_by_subject`: When the user asks to delete an email by subject/topic, ALWAYS use this tool first to locate it and present the subject to the user for confirmation.
5. `delete_email_confirmed`: ONLY call this after the user has explicitly confirmed with 'yes', 'confirm', or 'delete it'.

Voice Response Guidelines:
- Keep answers SHORT, conversational, and natural — they will be spoken aloud by TTS.
- Never read raw URLs, markdown asterisks, bullet points, or code syntax aloud.
- Speak warmly, clearly, and concisely.
"""


class GeminiAgent:
    """
    Wraps Google Gemini with multi-turn conversation history and
    automatic tool calling (Firecrawl search, Email tools).
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.tools = [
            web_search,
            send_email,
            read_inbox,
            find_email_by_subject,
            delete_email_confirmed,
        ]
        self._init_chat()

    def _init_chat(self):
        self.chat = self.client.chats.create(
            model=settings.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.7,
                tools=self.tools,
            ),
        )

    def respond(self, user_text: str) -> str:
        """
        Send user_text to Gemini, automatically execute tool calls,
        and return the synthesized voice-friendly reply.
        """
        print(f"[GEMINI] Sending to LLM: {user_text!r}")
        try:
            response = self.chat.send_message(user_text)
            reply_text = response.text.strip() if response.text else "I completed your request."
        except Exception as e:
            print(f"[GEMINI] Error during generation: {e}")
            reply_text = f"Sorry, I encountered an issue: {e}"

        print(f"[GEMINI] Response: {reply_text!r}")
        return reply_text

    def clear_history(self):
        """Reset conversation session."""
        self._init_chat()

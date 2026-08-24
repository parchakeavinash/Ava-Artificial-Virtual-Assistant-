from google import genai
from google.genai import types

from app.config import settings


_SYSTEM_PROMPT = """You are Ava, a helpful and friendly voice AI assistant.

Important rules:
- Keep your answers SHORT and conversational — they will be spoken aloud.
- Never use markdown, bullet points, numbered lists, or code blocks.
- Never say things like "As an AI" or "I cannot".
- Speak naturally, like a helpful human colleague.
- If you don't know something, say so simply and offer to help another way.
"""


class GeminiAgent:
    """
    Wraps Google Gemini with multi-turn conversation history.

    Each call to respond() appends the user message and assistant reply
    to self.history so Gemini remembers what was said earlier in the
    same session.

    Phase 2: plain text only.
    Phase 3+: add tool_config for MCP function calling.
    """

    def __init__(self):

        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        # Conversation history — list of Content objects.
        # Grows with every user + assistant turn.
        self.history: list[types.Content] = []

    # ---------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------

    def respond(self, user_text: str) -> str:
        """
        Send user_text to Gemini, return the assistant text response.
        Updates internal history so the next call has full context.
        """

        print(f"[GEMINI] Sending to LLM: {user_text!r}")

        # Append the new user turn
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_text)],
            )
        )

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        reply_text = response.text.strip()

        print(f"[GEMINI] Response: {reply_text!r}")

        # Append the assistant reply to history for next turn
        self.history.append(
            types.Content(
                role="model",
                parts=[types.Part(text=reply_text)],
            )
        )

        return reply_text

    def clear_history(self):
        """Reset conversation — call when starting a new session."""
        self.history = []

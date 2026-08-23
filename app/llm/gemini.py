from google import genai

from app.config.setting import settings

class GeminiClient:
    """
    Client responsible for communicating with Google's Gemini API.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key = settings.GEMINI_API_KEY
        )


    def generate(self,prompt: str)->str:
        """
        Send a prompt to Gemini and return the generated text.
        """
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents = prompt,
        )

        return response.text
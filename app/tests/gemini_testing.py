from app.llm.gemini import GeminiClient


client = GeminiClient()

response = client.generate(
    "Explain what RAG is in simple terms."
)

print(response)
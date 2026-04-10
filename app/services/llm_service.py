import openai

class LLMService:
    def __init__(self):
        self.client = openai.OpenAI()

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
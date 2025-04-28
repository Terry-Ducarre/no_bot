from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class OpenRouterClient:
    def __init__(self, api_key=None, model="mistralai/mistral-small-3.1-24b-instruct:free", max_retries=3):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

    def get_chat_response(self, user_message, system_prompt=None, history=None):
        """Gets a chat response from OpenRouter AI"""
        retries = 0
        messages = []

        # Include system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Include chat history if provided
        if history:
            messages.extend(history)

        # Append user message
        messages.append({"role": "user", "content": user_message})

        while retries < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional for rankings
                        "X-Title": "<YOUR_SITE_NAME>",  # Optional for rankings
                    },
                    model=self.model,
                    messages=messages
                )
                return completion.choices[0].message.content
            except Exception as e:
                retries += 1
                print(f"Error: {e}. Retrying {retries}/{self.max_retries}...")
                if completion:
                    print(f"Response: {completion}")   
                time.sleep(2**retries)  # Exponential backoff

        return "Error: Unable to get a response."

# Usage Example
client = OpenRouterClient()
response = client.get_chat_response(
    user_message="What is the meaning of life?",
    system_prompt="You are a wise AI that provides insightful answers.",
    history=[{"role": "assistant", "content": "Hello! How can I help you today?"}]
)
print(response)

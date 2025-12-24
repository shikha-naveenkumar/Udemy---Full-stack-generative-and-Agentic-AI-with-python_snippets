import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "I need you to solve only math related qs"},
        {"role": "user", "content": "Hey There, Can you help me solve a + b whole square?"}
    ]
)

print(response.choices[0].message.content)

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key="Google api key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        { "role": "system", "content": "I need you to solve only math related qs"},
        { "role": "user", "content": "Hey There, Can you help me solve a + b whole square?"}
    ]

)
print(response.choices[0].message.content)

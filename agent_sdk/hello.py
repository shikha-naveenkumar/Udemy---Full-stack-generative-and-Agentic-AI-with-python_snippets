import os
from pathlib import Path
from dotenv import load_dotenv


script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env") 
load_dotenv(script_dir.parent / "prompting&api_setup" / ".env") 


api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")


os.environ["OPENAI_API_KEY"] = api_key

from openai import OpenAI
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled


set_tracing_disabled(True)


gemini_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions") 

hello_agent = Agent(
    name="Hello World Agent",
    model="gemini-2.0-flash", 
    instructions="You're an agent which greets the user and helps them and answers using emojis and in a funny way"
)

result = Runner.run_sync(hello_agent, "Hey There, My Name is Piyush Garg")

print(result.final_output)

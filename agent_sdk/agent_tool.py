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

from openai import OpenAI, RateLimitError
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled
import time

set_tracing_disabled(True)

gemini_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")

spanish_agent = Agent(
    name="Spanish agent",
    model="gemini-1.5-flash",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    model="gemini-1.5-flash",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="gemini-1.5-flash",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ]
)

# Run with retry logic for rate limits
max_retries = 5
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}...")
        result = Runner.run_sync(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
        print(result.final_output)
        break
    except RateLimitError as e:
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 15
            print(f"Rate limited! Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            print(f"Failed after {max_retries} attempts. Please wait a few minutes and try again.")
            raise
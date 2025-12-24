import os
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import json

script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env") 
load_dotenv(script_dir.parent / "prompting&api_setup" / ".env") 

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")

os.environ["OPENAI_API_KEY"] = api_key

from openai import OpenAI
from agents import Agent, Runner, set_default_openai_client, set_default_openai_api, set_tracing_disabled, function_tool

set_tracing_disabled(True)

gemini_client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")

# Custom tool function - example: get current time
@function_tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Custom tool function - example: simple calculator
@function_tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression. Only numbers and +, -, *, /, (, ) are allowed."
    except Exception as e:
        return f"Error calculating: {str(e)}"

hello_agent = Agent(
    name="Hello World Agent",
    model="gemini-1.5-flash",
    instructions="You're an agent which greets the user and helps them and answers using emojis and in a funny way. Use the available tools when appropriate.",
    tools=[
        get_current_time,
        calculate
    ]
)

# Run with retry logic for rate limits
import time
from openai import RateLimitError

max_retries = 5
for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1}...")
        result = Runner.run_sync(hello_agent, "Hey! What time is it right now? Also, what's 25 * 4?")
        print(result.final_output)
        break
    except RateLimitError as e:
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 15  # Wait 15, 30, 45, 60 seconds
            print(f"Rate limited! Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            print(f"Failed after {max_retries} attempts. Please wait a few minutes and try again.")
            raise
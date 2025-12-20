from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel
from typing import Literal, Optional, Union
import requests
import json
import time

load_dotenv()

# ========================
# Pydantic Models for Chain of Thought
# ========================

class PlanStep(BaseModel):
    """Step 1: The agent plans what to do"""
    step: Literal["plan"]
    thought: str  # Agent's reasoning about what needs to be done


class ToolStep(BaseModel):
    """Step 2: The agent decides which tool to use"""
    step: Literal["tool"]
    tool_name: str  # Name of the tool to use
    tool_input: dict  # Input parameters for the tool


class ObserveStep(BaseModel):
    """Step 3: The agent observes the tool's output"""
    step: Literal["observe"]
    observation: str  # Result from the tool execution


class OutputStep(BaseModel):
    """Step 4: The agent provides the final answer"""
    step: Literal["output"]
    final_answer: str  # The final response to the user


# Union type for all possible steps
AgentStep = Union[PlanStep, ToolStep, ObserveStep, OutputStep]


class AgentResponse(BaseModel):
    """Complete agent response with chain of thought"""
    current_step: AgentStep


# ========================
# Available Tools
# ========================

def get_weather(city: str) -> str:
    """Fetches current weather for a given city using wttr.in API"""
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    
    if response.status_code == 200:
        return f"The weather in {city} is {response.text.strip()}"
    return "Something went wrong fetching weather data"


def get_forecast(city: str, days: int = 3) -> str:
    """Fetches weather forecast for a given city"""
    url = f"https://wttr.in/{city.lower()}?format=3"
    response = requests.get(url)
    
    if response.status_code == 200:
        return f"Weather forecast for {city}: {response.text.strip()}"
    return "Something went wrong fetching forecast data"


# Tool registry - maps tool names to functions
AVAILABLE_TOOLS = {
    "get_weather": {
        "function": get_weather,
        "description": "Get current weather for a city",
        "parameters": {"city": "string - name of the city"}
    },
    "get_forecast": {
        "function": get_forecast,
        "description": "Get weather forecast for a city",
        "parameters": {"city": "string - name of the city", "days": "integer - number of days (optional)"}
    }
}


# ========================
# AI Agent with Chain of Thought
# ========================

SYSTEM_PROMPT = """
You are a helpful AI Weather Agent that uses Chain of Thought reasoning.
You must respond in JSON format following these steps:

STEP 1 - PLAN: Think about what the user needs and plan your approach.
Response format: {"current_step": {"step": "plan", "thought": "your reasoning here"}}

STEP 2 - TOOL: Decide which tool to use and with what parameters.
Response format: {"current_step": {"step": "tool", "tool_name": "tool_name", "tool_input": {"param": "value"}}}

STEP 3 - OBSERVE: After I provide the tool result, analyze it.
Response format: {"current_step": {"step": "observe", "observation": "what you learned from the tool"}}

STEP 4 - OUTPUT: Provide your final answer to the user.
Response format: {"current_step": {"step": "output", "final_answer": "your complete answer"}}

Available Tools:
- get_weather: Get current weather for a city. Parameters: {"city": "city name"}
- get_forecast: Get weather forecast for a city. Parameters: {"city": "city name", "days": number}

Always respond with valid JSON. Think step by step.
"""


client = OpenAI(
    api_key="GOOGLE_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def parse_agent_response(response_text: str) -> AgentResponse:
    """Parse the LLM response into a structured AgentResponse"""
    try:
        # Clean up the response if it has markdown code blocks
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        data = json.loads(cleaned)
        return AgentResponse(**data)
    except Exception as e:
        print(f"⚠️ Error parsing response: {e}")
        print(f"Raw response: {response_text}")
        raise


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result"""
    if tool_name not in AVAILABLE_TOOLS:
        return f"Error: Tool '{tool_name}' not found"
    
    tool_func = AVAILABLE_TOOLS[tool_name]["function"]
    try:
        result = tool_func(**tool_input)
        return result
    except Exception as e:
        return f"Error executing tool: {str(e)}"


def run_agent(user_query: str):
    """Run the agent with Chain of Thought reasoning"""
    print(f"\n{'='*60}")
    print(f"🧑 User: {user_query}")
    print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    max_iterations = 10 
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Retry logic for rate limit errors
        max_retries = 3
        retry_delay = 5  # Start with 5 seconds
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=messages
                )
                break  # Success, exit retry loop
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = min(retry_delay * (2 ** attempt), 60)  # Exponential backoff, max 60s
                    print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} retries. Please wait a minute and try again.")
                    return None
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
    
        try:
            agent_response = parse_agent_response(assistant_message)
            current_step = agent_response.current_step
        except Exception:
            print("❌ Failed to parse response, retrying...")
            messages.append({"role": "user", "content": "Please respond with valid JSON format."})
            continue
        
        if isinstance(current_step, PlanStep):
            print(f"📋 PLAN: {current_step.thought}")
            messages.append({"role": "user", "content": "Good plan! Now proceed to select a tool."})
        
        elif isinstance(current_step, ToolStep):
            print(f"🔧 TOOL: {current_step.tool_name}")
            print(f"   Input: {current_step.tool_input}")
            
            # Execute the tool
            tool_result = execute_tool(current_step.tool_name, current_step.tool_input)
            print(f"   Result: {tool_result}")
            
            # Send the tool result back to the agent
            messages.append({
                "role": "user", 
                "content": f"Tool Result: {tool_result}\n\nNow analyze this result in an 'observe' step."
            })
        
        elif isinstance(current_step, ObserveStep):
            print(f"👁️ OBSERVE: {current_step.observation}")
            messages.append({"role": "user", "content": "Good observation! Now provide your final answer in an 'output' step."})
        
        elif isinstance(current_step, OutputStep):
            print(f"\n{'='*60}")
            print(f"🤖 FINAL ANSWER: {current_step.final_answer}")
            print(f"{'='*60}\n")
            return current_step.final_answer
    
    print("⚠️ Max iterations reached without a final answer")
    return None



if __name__ == "__main__":
    print("\n" + "🌤️ "*20)
    print("      WEATHER AGENT - Chain of Thought Demo")
    print("🌤️ "*20 + "\n")
    
    # Skip examples to avoid rate limiting, go straight to interactive mode
    
    # Example 3: Interactive mode
    print("\n📌 INTERACTIVE MODE")
    print("-" * 40)
    print("Type your weather questions (or 'quit' to exit):\n")
    
    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if user_input:
            run_agent(user_input)
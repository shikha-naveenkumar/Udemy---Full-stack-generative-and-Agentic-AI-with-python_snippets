# Full Stack Generative & Agentic AI with Python

Learning materials and code examples from the Udemy course on Generative and Agentic AI.

## 📁 Project Structure

| Folder | Description |
|--------|-------------|
| `prompting&api_setup/` | Basic Gemini API setup and chat completions |
| `agent_sdk/` | OpenAI Agents SDK examples with Gemini integration |

## 🛠️ Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with your API key:
   ```env
   GEMINI_API_KEY=your_google_api_key_here
   ```

4. **Get your free API key** from [Google AI Studio](https://aistudio.google.com/)

## 🚀 Examples

### Basic API Usage
```bash
python prompting&api_setup/main.py
```

### Agent SDK Examples
```bash
python agent_sdk/hello.py           # Basic agent
python agent_sdk/agent_with_tool.py # Agent with custom tools
python agent_sdk/agent_tool.py      # Multi-agent orchestration
```

## 📚 Technologies Used

- Python 3.8+
- Google Gemini API
- OpenAI Agents SDK
- python-dotenv

## ⚠️ Note

API keys are stored in `.env` files which are git-ignored for security.
# 🧪 Chatbot Lab

A tiny, **web-based** chatbot whose real purpose is to *teach AI concepts through its own setup*.
The chatbot itself just chats — the learning happens as your team edits one file,
`config.yaml`, and watches each change take effect. Perfect for a lunch-and-learn.

- **Web UI**, not a CLI — a chat window plus a live sidebar showing exactly what the
  config is doing right now.
- **One config file** controls everything: which provider (Anthropic, OpenAI,
  Google Gemini, a LiteLLM proxy, or local Ollama), the model, the system
  prompt, and tool/data connections (MCP).
- **Python 3.11** (tested on 3.11.15).

---

## Quick start

```bash
cd "Chatbot Lab"
./run.sh
```

`run.sh` creates a virtual environment, installs dependencies, copies
`config.example.yaml` to `config.yaml` if needed, and starts the server at
**http://127.0.0.1:8000**.

Prefer to do it by hand?

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # then edit config.yaml
python app.py
```

---

## The lunch-and-learn walkthrough

Open `config.yaml` on the projector. Each edit below teaches one idea — restart
the server after each change and refresh the browser to see it land.

### 1. Where does the model run? (`provider`)
Start with a **local** model so there's no key and no internet dependency:

```bash
# install once from https://ollama.com, then:
ollama pull llama3.2
```

```yaml
provider: ollama
```

**Talking point:** the same chat app can point at a model on this laptop or one in
the cloud. Flip `provider` to `anthropic` (with a key) and compare the answers,
speed, and quality. Concept: *models are interchangeable back-ends.*

### 2. How do we authenticate to a hosted model? (`api_key`)
Switch to a cloud provider:

```yaml
provider: anthropic
anthropic:
  api_key: "${ANTHROPIC_API_KEY}"   # or paste a key directly
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Talking point:** cloud models require a secret key that identifies (and bills)
you. Notice we keep it in an environment variable, not in the file — concept:
*don't commit secrets.* (`config.yaml` is git-ignored for the same reason.)

### 3. How do we steer behavior without training? (`system_prompt`)
Rewrite the prompt live — make the bot a pirate, a terse expert, a support agent:

```yaml
system_prompt: |
  You are a grumpy but helpful pirate. Answer in one sentence.
```

**Talking point:** we didn't retrain anything. The system prompt is instructions
prepended to every conversation. Concept: *prompting is the cheapest way to change
behavior.*

### 4. Bigger vs. smaller models (`model`)
```yaml
anthropic:
  model: claude-opus-5     # vs. claude-sonnet-5
```

**Talking point:** ask the same hard question of a small and a large model.
Concept: *there's a speed/cost/quality trade-off; pick the right size for the job.*

### 5. How does a model reach tools and live data? (`mcp_servers`)
Flip one server to `enabled: true`:

```yaml
mcp_servers:
  - name: filesystem
    enabled: true
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

Refresh the browser — the new tools appear in the sidebar. Now ask the bot to
"list the files in /tmp."

**Talking point:** a plain language model can't touch files or the internet by
itself. **MCP (Model Context Protocol)** is a standard way to give it tools. The
model *decides* to call a tool, we run it, and hand back the result. Concept:
*this is how chatbots become agents that can actually do things.*
(The `npx` examples need [Node.js](https://nodejs.org) installed.)

---

## How it fits together

```
Browser (static/index.html)
   │  POST /api/chat
   ▼
FastAPI (app.py)
   ├── reads config.yaml
   ├── talks to the model     (anthropic / openai / ollama)
   └── runs MCP tools on demand (mcp_servers)
```

- **`app.py`** — the whole backend, ~300 readable lines, commented for teaching.
- **`config.example.yaml`** — copy to `config.yaml`; this is the star of the show.
- **`static/index.html`** — the chat UI and the live config sidebar.

## Troubleshooting
- **"Config file not found"** — run `cp config.example.yaml config.yaml`.
- **"No API key set"** — set the key for your chosen provider (env var or file).
- **Ollama errors** — make sure `ollama` is running and you've `ollama pull`ed the model.
- **MCP tool didn't appear** — check the sidebar for a ⚠ error; `npx` MCP servers
  need Node.js. Tool-calling with Ollama needs a tool-capable model (e.g. `llama3.1`).

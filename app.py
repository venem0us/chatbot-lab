"""
Chatbot Lab — a tiny, config-driven web chatbot for teaching AI concepts.

The *point* of this app is the configuration file. During a lunch-and-learn you
walk your team through `config.yaml` and each edit teaches one idea:

  * provider        -> "where does the model run?" (cloud API vs. local Ollama)
  * api_key         -> "how do we authenticate to a hosted model?"
  * model           -> "models are swappable; bigger != always better"
  * system_prompt   -> "how we steer a model's behavior without training it"
  * mcp_servers     -> "how a model safely reaches tools and live data (MCP)"

The code is intentionally small and linear so it can be read top-to-bottom.

Runs on Python 3.11 (tested against 3.11.15).
"""

from __future__ import annotations

import os
import sys
import traceback
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# MCP (Model Context Protocol) client — lets the chatbot use external tools.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


BASE_DIR = Path(__file__).parent
CONFIG_PATH = Path(os.environ.get("CHATBOT_CONFIG", BASE_DIR / "config.yaml"))


# --------------------------------------------------------------------------- #
# 1. Configuration                                                            #
# --------------------------------------------------------------------------- #
def load_config() -> dict[str, Any]:
    """Read config.yaml. Everything the chatbot does is driven from here."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}\n"
            "Copy config.example.yaml to config.yaml and edit it."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Allow the API key to come from an env var so we never have to commit it.
    # e.g. api_key: "${ANTHROPIC_API_KEY}"
    def resolve_env(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value

    for section in ("anthropic", "openai", "gemini", "litellm", "ollama"):
        if isinstance(cfg.get(section), dict) and "api_key" in cfg[section]:
            cfg[section]["api_key"] = resolve_env(cfg[section]["api_key"])
    return cfg


# --------------------------------------------------------------------------- #
# 2. MCP tool manager                                                         #
# --------------------------------------------------------------------------- #
class MCPManager:
    """Connects to the MCP servers listed in config and exposes their tools.

    MCP is how a model reaches the outside world (files, APIs, databases) in a
    standard way. Each configured server is a small program we launch; it tells
    us which tools it offers, and we let the model call them.
    """

    def __init__(self, servers: list[dict[str, Any]]):
        self.servers = servers
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}  # tool name -> session
        self.tools: list[dict[str, Any]] = []          # normalized tool defs
        self.errors: list[str] = []                    # surfaced in the UI

    async def connect(self) -> None:
        for srv in self.servers:
            name = srv.get("name", "unnamed")
            if not srv.get("enabled", True):
                continue
            try:
                params = StdioServerParameters(
                    command=srv["command"],
                    args=srv.get("args", []),
                    env={**os.environ, **srv.get("env", {})},
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                listed = await session.list_tools()
                for tool in listed.tools:
                    self._sessions[tool.name] = session
                    self.tools.append(
                        {
                            "server": name,
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                        }
                    )
            except Exception as exc:  # keep the app usable even if a server fails
                self.errors.append(f"MCP server '{name}' failed to start: {exc}")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        session = self._sessions.get(name)
        if session is None:
            return f"Error: no MCP tool named '{name}' is connected."
        result = await session.call_tool(name, arguments)
        # MCP returns a list of content blocks; stitch the text together.
        parts = []
        for block in result.content:
            parts.append(getattr(block, "text", None) or str(block))
        return "\n".join(parts) if parts else "(tool returned no output)"

    async def close(self) -> None:
        await self._stack.aclose()


# --------------------------------------------------------------------------- #
# 3. Provider clients (cloud API or local Ollama)                             #
# --------------------------------------------------------------------------- #
# We support two shapes of API:
#   * Anthropic (Claude)     -> the `anthropic` SDK
#   * OpenAI-compatible       -> the `openai` SDK, also used for Ollama since
#                                Ollama exposes an OpenAI-compatible endpoint.
# The chat loops below handle "tool use": the model asks to run an MCP tool,
# we run it, hand back the result, and let the model continue.

async def chat_anthropic(cfg: dict, mcp: MCPManager, messages: list[dict]) -> str:
    from anthropic import AsyncAnthropic

    section = cfg.get("anthropic", {})
    api_key = section.get("api_key") or ""
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key set. Add it under `anthropic.api_key` in config.yaml."
        )
    client = AsyncAnthropic(api_key=api_key)
    model = section.get("model", "claude-sonnet-5")
    system_prompt = cfg.get("system_prompt", "")

    tools = [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in mcp.tools
    ]

    convo = [{"role": m["role"], "content": m["content"]} for m in messages]

    while True:
        resp = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=convo,
            tools=tools or None,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # Model wants to use one or more tools.
        convo.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                output = await mcp.call_tool(block.name, block.input or {})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        convo.append({"role": "user", "content": tool_results})


async def chat_openai_compatible(cfg: dict, mcp: MCPManager, messages: list[dict], provider: str) -> str:
    from openai import AsyncOpenAI

    section = cfg.get(provider, {})
    if provider == "ollama":
        base_url = section.get("base_url", "http://localhost:11434/v1")
        api_key = section.get("api_key") or "ollama"  # Ollama ignores the key
        model = section.get("model", "llama3.2")
    elif provider == "litellm":
        # A LiteLLM proxy is a central OpenAI-compatible gateway (many teams run
        # one for key management and governance). We just point at its URL; the
        # proxy decides which real model each `model` name maps to.
        base_url = section.get("base_url", "http://localhost:4000/v1")
        # Proxies may require a virtual/master key or none at all; the OpenAI SDK
        # needs *some* string, so fall back to a harmless placeholder.
        api_key = section.get("api_key") or "sk-litellm"
        model = section.get("model", "gpt-4o-mini")
    elif provider == "gemini":
        # Google exposes an OpenAI-compatible endpoint for Gemini, so we reuse
        # this same code path — only the base URL and defaults differ.
        base_url = section.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
        api_key = section.get("api_key") or ""
        model = section.get("model", "gemini-2.5-flash")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key set. Add it under `gemini.api_key` in config.yaml."
            )
    else:  # openai (or any OpenAI-compatible gateway)
        base_url = section.get("base_url", "https://api.openai.com/v1")
        api_key = section.get("api_key") or ""
        model = section.get("model", "gpt-4o-mini")
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key set. Add it under `openai.api_key` in config.yaml."
            )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    system_prompt = cfg.get("system_prompt", "")

    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in mcp.tools
    ]

    convo = [{"role": "system", "content": system_prompt}] if system_prompt else []
    convo += [{"role": m["role"], "content": m["content"]} for m in messages]

    while True:
        resp = await client.chat.completions.create(
            model=model,
            messages=convo,
            tools=tools or None,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""

        # Record the assistant's tool request, then answer each call.
        convo.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        import json

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            output = await mcp.call_tool(tc.function.name, args)
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": output})


async def run_chat(app_state: "AppState", messages: list[dict]) -> str:
    provider = app_state.config.get("provider", "anthropic").lower()
    if provider == "anthropic":
        return await chat_anthropic(app_state.config, app_state.mcp, messages)
    if provider in ("openai", "ollama", "gemini", "litellm"):
        return await chat_openai_compatible(app_state.config, app_state.mcp, messages, provider)
    raise RuntimeError(
        f"Unknown provider '{provider}'. Use one of: anthropic, openai, gemini, litellm, ollama."
    )


# --------------------------------------------------------------------------- #
# 4. Web app                                                                  #
# --------------------------------------------------------------------------- #
class AppState:
    config: dict[str, Any]
    mcp: MCPManager


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load config and connect MCP servers once, reuse for all requests.
    state.config = load_config()
    state.mcp = MCPManager(state.config.get("mcp_servers", []) or [])
    await state.mcp.connect()
    yield
    # Shutdown: cleanly close MCP subprocesses.
    await state.mcp.close()


app = FastAPI(title="Chatbot Lab", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    """Non-secret view of the active config, shown in the UI as a teaching aid."""
    cfg = state.config
    provider = cfg.get("provider", "anthropic").lower()
    model = (cfg.get(provider, {}) or {}).get("model", "(default)")
    return {
        "provider": provider,
        "model": model,
        "system_prompt": cfg.get("system_prompt", ""),
        "tools": [{"server": t["server"], "name": t["name"], "description": t["description"]} for t in state.mcp.tools],
        "mcp_errors": state.mcp.errors,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        reply = await run_chat(state, req.messages)
        return {"reply": reply}
    except Exception as exc:
        # Surface errors to the UI — debugging config mistakes is part of the lesson.
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(exc)})


# Serve any other static assets (kept last so it doesn't shadow the API routes).
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"Loading config from: {CONFIG_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

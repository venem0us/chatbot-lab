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

import json
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
from mcp.types import ListRootsResult, Root


BASE_DIR = Path(__file__).parent
CONFIG_PATH = Path(os.environ.get("CHATBOT_CONFIG", BASE_DIR / "config.yaml"))

# Every backend we know how to talk to, and which ones can't work without a key.
PROVIDERS = ("anthropic", "openai", "gemini", "litellm", "ollama")
KEY_REQUIRED = ("anthropic", "openai", "gemini")


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

    for section in PROVIDERS:
        if isinstance(cfg.get(section), dict) and "api_key" in cfg[section]:
            cfg[section]["api_key"] = resolve_env(cfg[section]["api_key"])
    return cfg


def model_catalog(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Every provider/model pair the config offers, for the UI's model dropdown.

    A provider section names its default with `model:` and may list extra
    choices with `models:`. Both are optional; a section with neither simply
    doesn't appear in the dropdown.
    """
    catalog: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        section = cfg.get(provider)
        if not isinstance(section, dict):
            continue
        names = [section["model"]] if section.get("model") else []
        names += [n for n in (section.get("models") or [])]
        for name in dict.fromkeys(str(n) for n in names):  # de-dupe, keep order
            catalog.append(
                {
                    "id": f"{provider}:{name}",
                    "provider": provider,
                    "model": name,
                    # Flagged in the dropdown so a missing key is obvious *before*
                    # you send a message and get an error.
                    "needs_key": provider in KEY_REQUIRED and not section.get("api_key"),
                }
            )
    return catalog


def active_model(cfg: dict[str, Any]) -> tuple[str, str]:
    """The provider/model the config starts on — the dropdown's initial value."""
    provider = str(cfg.get("provider", "anthropic")).lower()
    section = cfg.get(provider) if isinstance(cfg.get(provider), dict) else {}
    model = section.get("model") or next(iter(section.get("models") or []), "")
    return provider, str(model)


def resolve_model(cfg: dict[str, Any], model_id: str | None) -> tuple[str, str]:
    """Turn a dropdown selection ("provider:model") into a provider and model.

    Only pairs present in the config are accepted, so the browser can switch
    between configured models but can't ask the server for an arbitrary one.
    """
    if not model_id:
        return active_model(cfg)
    for entry in model_catalog(cfg):
        if entry["id"] == model_id:
            return entry["provider"], entry["model"]
    raise RuntimeError(
        f"'{model_id}' isn't a model configured in config.yaml. "
        "Add it under the provider's `models:` list."
    )


# --------------------------------------------------------------------------- #
# 2. MCP tool manager                                                         #
# --------------------------------------------------------------------------- #
def _server_roots(srv: dict[str, Any]) -> list[str]:
    """The directories to expose to a server as MCP "roots" (file:// URIs).

    "Roots" are how an MCP client tells a server which folders it may work with.
    We use the ones set explicitly in config (`roots:`), or, failing that, infer
    them from any absolute path arguments (e.g. the "/tmp" that the filesystem
    server is launched with). Answering the server's roots request is required —
    the filesystem server asks for roots on startup and hangs if nobody replies.
    """
    raw = srv.get("roots")
    if not raw:
        raw = [
            a for a in srv.get("args", [])
            if isinstance(a, str) and not a.startswith("-")
            and (p := Path(os.path.expanduser(a))).is_absolute() and p.exists()
        ]
    uris: list[str] = []
    for r in raw:
        s = str(r)
        if s.startswith("file://"):
            uris.append(s)
        else:
            p = Path(os.path.expanduser(s))
            uris.append((p if p.is_absolute() else p.resolve()).as_uri())
    return uris


def _make_list_roots_callback(uris: list[str]):
    """Build the handler the client uses to answer the server's roots/list."""
    async def _list_roots(_context) -> ListRootsResult:
        return ListRootsResult(roots=[Root(uri=u) for u in uris])
    return _list_roots


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
                # Register a roots handler so we answer the server's `roots/list`
                # request. Without it, servers that use roots (e.g. filesystem)
                # hang on startup waiting for a reply that never comes.
                session = await self._stack.enter_async_context(
                    ClientSession(read, write, list_roots_callback=_make_list_roots_callback(_server_roots(srv)))
                )
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
#
# Each loop iteration also records a "trace" step — the exact JSON we send and
# get back — so the web UI can show it in the "Under the hood" panel.

# Keys we redact from the trace as a safety net. (Credentials aren't in the
# request *body* anyway — the SDK puts the API key in an HTTP header — but this
# guards against anything credential-shaped slipping into what we display.)
_CRED_KEYS = {"api_key", "apikey", "authorization", "x-api-key", "access_token", "secret", "bearer"}


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("***redacted***" if k.lower() in _CRED_KEYS else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def snapshot(obj: Any) -> Any:
    """A frozen, JSON-safe, credential-free copy for the trace panel."""
    return _scrub(json.loads(json.dumps(obj, default=str)))


async def chat_anthropic(cfg: dict, mcp: MCPManager, messages: list[dict], trace: list[dict], model: str = "") -> str:
    from anthropic import AsyncAnthropic

    section = cfg.get("anthropic", {})
    api_key = section.get("api_key") or ""
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key set. Add it under `anthropic.api_key` in config.yaml."
        )
    client = AsyncAnthropic(api_key=api_key)
    model = model or section.get("model", "claude-sonnet-5")
    system_prompt = cfg.get("system_prompt", "")

    tools = [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in mcp.tools
    ]

    convo = [{"role": m["role"], "content": m["content"]} for m in messages]

    while True:
        request_body = {
            "model": model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": convo,
            "tools": tools or None,
        }
        step = {
            "step": len(trace) + 1,
            "kind": "initial request" if not trace else "follow-up request (after tool use)",
            "request": snapshot(request_body),
        }
        trace.append(step)

        resp = await client.messages.create(**request_body)
        step["response"] = snapshot(resp.model_dump())

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        # Model wants to use one or more tools.
        convo.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        tool_results = []
        calls = []
        for block in resp.content:
            if block.type == "tool_use":
                output = await mcp.call_tool(block.name, block.input or {})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
                calls.append({"tool": block.name, "input": block.input or {}, "result": output})
        step["tool_calls"] = snapshot(calls)
        convo.append({"role": "user", "content": tool_results})


async def chat_openai_compatible(cfg: dict, mcp: MCPManager, messages: list[dict], provider: str, trace: list[dict], model: str = "") -> str:
    from openai import AsyncOpenAI

    # `model` is the dropdown's choice; each branch below falls back to the
    # provider's own default when nothing was picked.
    chosen = model
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

    model = chosen or model
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
        request_body = {"model": model, "messages": convo, "tools": tools or None}
        step = {
            "step": len(trace) + 1,
            "kind": "initial request" if not trace else "follow-up request (after tool use)",
            "request": snapshot(request_body),
        }
        trace.append(step)

        resp = await client.chat.completions.create(**request_body)
        step["response"] = snapshot(resp.model_dump())

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
        calls = []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            output = await mcp.call_tool(tc.function.name, args)
            convo.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            calls.append({"tool": tc.function.name, "input": args, "result": output})
        step["tool_calls"] = snapshot(calls)


async def run_chat(app_state: "AppState", messages: list[dict], trace: list[dict], model_id: str | None = None) -> tuple[str, str]:
    """Run one turn and return (reply, "provider:model" actually used).

    `model_id` is the UI dropdown's selection; when it's absent we use the
    provider and model the config file starts on.
    """
    cfg = app_state.config
    provider, model = resolve_model(cfg, model_id)
    if provider == "anthropic":
        reply = await chat_anthropic(cfg, app_state.mcp, messages, trace, model)
    elif provider in ("openai", "ollama", "gemini", "litellm"):
        reply = await chat_openai_compatible(cfg, app_state.mcp, messages, provider, trace, model)
    else:
        raise RuntimeError(
            f"Unknown provider '{provider}'. Use one of: anthropic, openai, gemini, litellm, ollama."
        )
    return reply, f"{provider}:{model}"


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
    # Which configured model to answer with ("provider:model"), from the
    # sidebar dropdown. Omitted -> whatever config.yaml starts on.
    model_id: str | None = None


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    """Non-secret view of the active config, shown in the UI as a teaching aid."""
    cfg = state.config
    provider, model = active_model(cfg)
    return {
        "provider": provider,
        "model": model or "(default)",
        # Populates the model dropdown; `active_model_id` is its start value.
        "models": model_catalog(cfg),
        "active_model_id": f"{provider}:{model}",
        "system_prompt": cfg.get("system_prompt", ""),
        "tools": [{"server": t["server"], "name": t["name"], "description": t["description"]} for t in state.mcp.tools],
        "mcp_errors": state.mcp.errors,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # `trace` captures each request/response in the tool-use loop for the
    # "Under the hood" panel. It's returned even on error, so the UI can show
    # exactly what was sent before things went wrong.
    trace: list[dict] = []
    try:
        reply, used = await run_chat(state, req.messages, trace, req.model_id)
        return {"reply": reply, "trace": trace, "model": used}
    except Exception as exc:
        # Surface errors to the UI — debugging config mistakes is part of the lesson.
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(exc), "trace": trace})


# Serve any other static assets (kept last so it doesn't shadow the API routes).
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    print(f"Loading config from: {CONFIG_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

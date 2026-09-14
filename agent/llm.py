"""
Builds the chat model used by the agent.

IMPORTANT / UPDATED (Sept 2026): Groq deprecated every general-purpose Llama
chat model on its free tier - llama-3.1-8b-instant and llama-3.3-70b-versatile
were shut down August 16, 2026 (Llama 4 Scout/Maverick were shut down earlier
in 2026). The only Llama model still on Groq is llama-guard-4-12b, a safety
classifier, not a chat model. See https://console.groq.com/docs/deprecations

Two supported providers, both free, both zero cost:

  PROVIDER="groq"   -> Groq's currently-active free-tier model
                        (openai/gpt-oss-120b - NOT a Llama model anymore,
                        but still free, fast, and tool-calling capable).
                        Needs a free Groq API key.

  PROVIDER="ollama" -> An ACTUAL Llama model (e.g. llama3.1:8b), run
                        entirely locally on your Mac via Ollama - free
                        forever, no API key, no deprecation risk, and
                        genuinely "Llama" for the TCS conversation.
                        Ollama also exposes an OpenAI-compatible endpoint,
                        so this reuses the exact same ChatOpenAI wrapper.

Set PROVIDER via the LLM_PROVIDER env var (defaults to "ollama" so this
project keeps working with real Llama with zero ongoing dependency on
Groq's model lifecycle).
"""

from __future__ import annotations
import os
from langchain_openai import ChatOpenAI

GROQ_MODEL = "openai/gpt-oss-120b"      # current, active, free-tier Groq model
OLLAMA_MODEL = "llama3.1:8b"             # real Llama, run 100% locally, free forever


def get_llm(api_key: str | None = None, model: str | None = None,
            provider: str | None = None, temperature: float = 0.1):
    """temperature=0.1 (low, not 0.7+ default) is a deliberate choice for a
    policy assistant: consistency and predictability matter far more than
    creativity when the answer needs to reflect an exact company rule."""
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()

    if resolved_provider == "ollama":
        # Ollama must be running locally (`ollama serve`, usually automatic)
        # with the model pulled once (`ollama pull llama3.1:8b`). No API key,
        # no network call leaves your machine, no cost, ever.
        return ChatOpenAI(
            model=model or OLLAMA_MODEL,
            api_key="ollama",  # unused, but the client requires a non-empty string
            base_url="http://localhost:11434/v1",
            temperature=temperature,
        )

    if resolved_provider == "groq":
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Groq API key found. Get a free one at https://console.groq.com "
                "and set GROQ_API_KEY (or pass api_key=)."
            )
        return ChatOpenAI(
            model=model or GROQ_MODEL,
            api_key=resolved_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{resolved_provider}' - use 'ollama' or 'groq'.")

"""Async client for an OpenAI-compatible server (the local vLLM Qwen server).

Every call returns the generated text plus the top-k log-probabilities of the
*first* generated token, which is where agents commit to an answer letter.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List

from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    base_url: str = os.getenv("CONFTOPO_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("CONFTOPO_API_KEY", "EMPTY")
    model: str = os.getenv("CONFTOPO_MODEL", "gpt-4o-mini")  # vLLM alias of Qwen2.5-7B-Instruct
    max_concurrency: int = 64
    top_logprobs: int = 20
    timeout: float = 300.0
    max_retries: int = 4


@dataclass
class LLMResult:
    text: str
    first_token_logprobs: Dict[str, float]  # token string -> logprob (top-k)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        self.client = AsyncOpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key,
                                  timeout=self.cfg.timeout, max_retries=0)
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self.usage = Usage()

    async def chat(self, messages: List[dict], max_tokens: int) -> LLMResult:
        # temperature=0 -> greedy text, and vLLM reports logprobs of the
        # untempered distribution (it scales logits by T before log-softmax).
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=self.cfg.model, messages=messages, max_tokens=max_tokens,
                        temperature=0.0, logprobs=True, top_logprobs=self.cfg.top_logprobs)
                choice = resp.choices[0]
                first = choice.logprobs.content[0]
                top = {}
                for t in first.top_logprobs:
                    # the same string can appear under several token ids; keep the max
                    top[t.token] = max(top.get(t.token, float("-inf")), t.logprob)
                self.usage.calls += 1
                self.usage.prompt_tokens += resp.usage.prompt_tokens
                self.usage.completion_tokens += resp.usage.completion_tokens
                return LLMResult(choice.message.content or "", top,
                                 resp.usage.prompt_tokens, resp.usage.completion_tokens)
            except Exception as e:  # network hiccup / overloaded shared GPU
                last_err = e
                await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} tries: {last_err!r}")

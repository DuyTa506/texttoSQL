"""
SQL Inference — multi-provider generation engine.

Supports two backends via the ``provider`` parameter:

  "local"   (default)
      Loads a HuggingFace / Unsloth fine-tuned model with AutoModelForCausalLM.
      Requires ``model_path`` to point to the SFT/RL checkpoint.
      Use this for training evaluation on GPU hardware.

  "openai"
      Calls the OpenAI Chat Completions API (or any OpenAI-compatible endpoint).
      No GPU required — good for pipeline testing and ablation studies.
      Requires ``api_key`` or OPENAI_API_KEY env var.
      Set ``model_path`` to the model name, e.g. "gpt-4o", "gpt-4o-mini".

Both backends expose the same public interface:

    inference = SQLInference.from_config(gen_cfg)
    result    = inference.generate(question, schema_context)
    # result → {"sql": str, "reasoning": str, "raw_output": str}

Generation modes ("standard", "cot_plan", "divide_conquer") and
multi-candidate generation (n_candidates > 1) are supported by both backends.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

logger = logging.getLogger(__name__)

# ── System prompt fragments for each generation mode ─────────────────────────

_COT_PLAN_INSTRUCTION = (
    "Think step by step like a query planner: "
    "identify the relevant tables, determine the necessary JOINs, "
    "apply the correct filters and aggregations, then write the SQL."
)

_DIVIDE_CONQUER_INSTRUCTION = (
    "Break this query into parts. "
    "Solve each part as a sub-query or CTE, "
    "then combine them into a single final SQL statement."
)

_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural-language question, "
    "generate a correct SQL query.\n"
    "Always wrap your final SQL in a ```sql ... ``` code block."
)


# ── Public factory ────────────────────────────────────────────────────────────


class SQLInference:
    """
    Multi-provider SQL inference engine.

    Do not instantiate directly — use :meth:`from_config` or one of the
    provider-specific constructors :meth:`local` / :meth:`openai`.
    """

    # Delegates all real work to a *backend* object.
    # Backend must implement:
    #   backend.complete(messages: list[dict], temperature: float) -> str

    def __init__(self, backend, *, max_new_tokens: int = 512,
                 temperature: float = 0.0, candidate_temperature: float = 0.8):
        self._backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.candidate_temperature = candidate_temperature

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, gen_cfg: dict) -> "SQLInference":
        """
        Build a SQLInference from a ``generation`` config dict (from YAML).

        Keys read:
          provider            "local" | "openai"            default "local"
          model_path          model path or API model name  required
          api_key             OpenAI API key                optional (env fallback)
          api_base_url        custom OpenAI-compatible URL  optional
          max_new_tokens      int                           default 512
          temperature         float                         default 0.0
          candidate_temperature float                       default 0.8
        """
        provider = gen_cfg.get("provider", "local").lower()
        model_path = gen_cfg.get("model_path", "")
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
        temperature = float(gen_cfg.get("temperature", 0.0))
        candidate_temperature = float(gen_cfg.get("candidate_temperature", 0.8))

        if provider == "openai":
            api_key = gen_cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
            api_base_url = gen_cfg.get("api_base_url", "")
            backend = _OpenAIBackend(
                model=model_path or "gpt-4o-mini",
                api_key=api_key or None,
                api_base_url=api_base_url or None,
                max_tokens=max_new_tokens,
            )
        else:
            if not model_path:
                raise ValueError(
                    "generation.model_path must be set for provider='local'. "
                    "Point it to your SFT/RL checkpoint."
                )
            backend = _LocalBackend(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
            )

        return cls(
            backend,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            candidate_temperature=candidate_temperature,
        )

    @classmethod
    def local(cls, model_path: str, **kwargs) -> "SQLInference":
        """Convenience constructor for the local HuggingFace backend."""
        backend = _LocalBackend(
            model_path=model_path,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
        )
        return cls(backend, **kwargs)

    @classmethod
    def openai(
        cls,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        api_base_url: str | None = None,
        **kwargs,
    ) -> "SQLInference":
        """Convenience constructor for the OpenAI backend."""
        backend = _OpenAIBackend(
            model=model,
            api_key=api_key,
            api_base_url=api_base_url,
            max_tokens=kwargs.get("max_new_tokens", 512),
        )
        return cls(backend, **kwargs)

    # ── Public API (same interface regardless of backend) ─────────────────────

    def generate(
        self,
        question: str,
        schema_context: str,
        mode: str = "standard",
        n_candidates: int = 1,
    ):
        """
        Generate SQL from question + schema context.

        Parameters
        ----------
        question : str
            Natural-language question (or a full correction prompt string).
        schema_context : str
            Formatted schema string from SchemaFilter / GraphRetriever.
        mode : str
            "standard" | "cot_plan" | "divide_conquer"
        n_candidates : int
            >1 triggers diverse candidate sampling; returns list[dict].

        Returns
        -------
        dict | list[dict]
            dict keys: ``sql``, ``reasoning``, ``raw_output``.
        """
        if n_candidates > 1:
            return self._generate_candidates(question, schema_context, mode, n_candidates)

        messages = _build_messages(question, schema_context, mode)
        raw = self._backend.complete(messages, temperature=self.temperature)
        return _parse_output(raw)

    def generate_batch(
        self,
        questions: list[str],
        schema_contexts: list[str],
        mode: str = "standard",
    ) -> list[dict]:
        """Generate SQL for a batch of (question, schema) pairs."""
        return [
            self.generate(q, sc, mode=mode)
            for q, sc in zip(questions, schema_contexts)
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_candidates(
        self, question: str, schema_context: str, mode: str, n: int,
    ) -> list[dict]:
        messages = _build_messages(question, schema_context, mode)
        results = []
        for _ in range(n):
            raw = self._backend.complete(
                messages, temperature=self.candidate_temperature
            )
            results.append(_parse_output(raw))
        return results

    # ── Async API ─────────────────────────────────────────────────────────────

    async def agenerate(
        self,
        question: str,
        schema_context: str,
        mode: str = "standard",
        n_candidates: int = 1,
    ):
        """Async version of :meth:`generate`.

        Uses ``backend.async_complete()`` for OpenAI backends, or falls
        back to ``asyncio.to_thread()`` for local backends.
        """
        if n_candidates > 1:
            return await self._agenerate_candidates(
                question, schema_context, mode, n_candidates
            )

        messages = _build_messages(question, schema_context, mode)

        if hasattr(self._backend, "async_complete"):
            raw = await self._backend.async_complete(
                messages, temperature=self.temperature
            )
        else:
            raw = await asyncio.to_thread(
                self._backend.complete, messages, self.temperature
            )

        return _parse_output(raw)

    async def _agenerate_candidates(
        self, question: str, schema_context: str, mode: str, n: int,
    ) -> list[dict]:
        messages = _build_messages(question, schema_context, mode)
        results = []
        for _ in range(n):
            if hasattr(self._backend, "async_complete"):
                raw = await self._backend.async_complete(
                    messages, temperature=self.candidate_temperature
                )
            else:
                raw = await asyncio.to_thread(
                    self._backend.complete, messages, self.candidate_temperature
                )
            results.append(_parse_output(raw))
        return results

    async def agenerate_batch(
        self,
        items: list[tuple[str, str]],
        mode: str = "standard",
        n_candidates: int = 1,
        concurrency: int = 20,
    ) -> list:
        """Async batch generation with concurrency control.

        Parameters
        ----------
        items:
            List of ``(question, schema_context)`` tuples.
        mode:
            Generation mode.
        n_candidates:
            Candidates per example.
        concurrency:
            Max parallel API calls (semaphore limit).

        Returns
        -------
        list[dict | list[dict]]
            One result per input item (dict or list[dict] if n_candidates > 1).
        """
        # For local backends with batch_complete, use GPU batching
        if hasattr(self._backend, "batch_complete") and n_candidates == 1:
            messages_list = [
                _build_messages(q, sc, mode) for q, sc in items
            ]
            raws = self._backend.batch_complete(
                messages_list, temperature=self.temperature
            )
            return [_parse_output(raw) for raw in raws]

        # For API backends, use semaphore-controlled async gather
        semaphore = asyncio.Semaphore(concurrency)

        async def _throttled(question: str, schema_context: str):
            async with semaphore:
                for attempt in range(3):
                    try:
                        return await self.agenerate(
                            question, schema_context,
                            mode=mode, n_candidates=n_candidates,
                        )
                    except Exception as exc:
                        err_msg = str(exc).lower()
                        is_rate_limit = (
                            "rate" in err_msg
                            or "429" in err_msg
                            or "too many" in err_msg
                        )
                        if is_rate_limit and attempt < 2:
                            wait = (2 ** attempt) * 1.0
                            logger.warning(
                                "Rate limit hit, retrying in %.1fs (attempt %d/3): %s",
                                wait, attempt + 1, exc,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise

        tasks = [_throttled(q, sc) for q, sc in items]
        return await asyncio.gather(*tasks)


# ── Backends ──────────────────────────────────────────────────────────────────


class _LocalBackend:
    """
    HuggingFace / Unsloth local model backend.

    Lazy-loads model + tokenizer on first call so startup is fast when the
    OpenAI backend is selected instead.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 512):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading local model from: %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Local model loaded.")

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        """
        Run local model inference on the given chat messages.

        Converts the messages list to a single prompt string using the
        ChatML template (<|im_start|> / <|im_end|>) that Qwen3 expects.
        """
        prompt = _messages_to_chatml(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        import torch
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-7),
            "pad_token_id": (
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            ),
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def batch_complete(
        self,
        messages_list: list[list[dict]],
        temperature: float = 0.0,
        sub_batch_size: int = 8,
    ) -> list[str]:
        """Run batched local inference with left-pad tokenization.

        Processes ``messages_list`` in sub-batches of ``sub_batch_size`` to
        avoid GPU OOM.  Each sub-batch is a single ``model.generate()`` call
        with left-padded inputs.

        Parameters
        ----------
        messages_list:
            List of chat message lists, one per example.
        temperature:
            Sampling temperature (0.0 = greedy).
        sub_batch_size:
            Max examples per GPU forward pass.

        Returns
        -------
        list[str]
            One completion string per input.
        """
        import torch

        results: list[str] = []

        # Ensure tokenizer is left-padded for batched generation
        tokenizer = self.tokenizer
        model = self.model

        original_padding_side = getattr(tokenizer, "padding_side", "right")
        original_pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        try:
            for start in range(0, len(messages_list), sub_batch_size):
                batch_msgs = messages_list[start : start + sub_batch_size]
                prompts = [_messages_to_chatml(m) for m in batch_msgs]

                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(model.device)

                gen_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": temperature > 0,
                    "temperature": max(temperature, 1e-7),
                    "pad_token_id": tokenizer.pad_token_id,
                }

                with torch.no_grad():
                    output_ids = model.generate(**inputs, **gen_kwargs)

                input_len = inputs["input_ids"].shape[1]
                for j in range(len(batch_msgs)):
                    new_tokens = output_ids[j][input_len:]
                    text = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    results.append(text)
        finally:
            # Restore original tokenizer state
            tokenizer.padding_side = original_padding_side
            tokenizer.pad_token_id = original_pad_token_id

        return results


class _OpenAIBackend:
    """
    OpenAI Chat Completions backend (or any OpenAI-compatible API).

    Lazy-loads the ``openai`` client on first call.
    Supports custom ``api_base_url`` for compatible endpoints
    (Azure OpenAI, Together.ai, Groq, local vLLM, etc.).
    """

    # Models that use hidden reasoning tokens (counted against max_completion_tokens).
    # These need a much higher token budget than the visible output alone.
    _REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini",
                         "o4-mini",
                         "gpt-5-nano", "gpt-5-mini", "gpt-5"}
    _REASONING_MIN_TOKENS = 4096  # minimum budget for reasoning models

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base_url: str | None = None,
        max_tokens: int = 512,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base_url = api_base_url

        # Auto-bump for reasoning models: reasoning tokens are hidden but
        # counted against max_completion_tokens, so 512 is far too low.
        if (model in self._REASONING_MODELS
                and max_tokens < self._REASONING_MIN_TOKENS):
            logger.info(
                "Reasoning model '%s' detected — bumping max_tokens %d → %d "
                "(reasoning tokens count against the limit).",
                model, max_tokens, self._REASONING_MIN_TOKENS,
            )
            max_tokens = self._REASONING_MIN_TOKENS

        self.max_tokens = max_tokens
        self._client = None
        self._compat_strategy: int | None = None  # cached working strategy index

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for provider='openai'. "
                "Install with: pip install openai"
            )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set generation.api_key in config, pass --api_key, "
                "or export OPENAI_API_KEY=<your-key>."
            )
        kwargs: dict = {"api_key": self.api_key}
        if self.api_base_url:
            kwargs["base_url"] = self.api_base_url
        self._client = OpenAI(**kwargs)
        logger.info(
            "OpenAI backend ready (model=%s%s).",
            self.model,
            f", base_url={self.api_base_url}" if self.api_base_url else "",
        )
        return self._client

    def complete(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Call the OpenAI Chat Completions API and return the assistant text.

        Handles API differences across model generations:
        - Newer reasoning models (o1, gpt-5-nano, etc.) require
          ``max_completion_tokens`` and only accept ``temperature=1``.
        - Older models use ``max_tokens`` and accept any temperature.
        """
        client = self._get_client()
        try:
            response = self._call_with_compat(client, messages, temperature)
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise

    def _call_with_compat(self, client, messages, temperature):
        """Try the newest API surface first, then fall back progressively.

        Caches the working strategy index after the first successful call
        so subsequent calls go directly without retries.
        """
        strategies = [
            {"max_completion_tokens": self.max_tokens, "temperature": temperature},
            {"max_completion_tokens": self.max_tokens},
            {"max_tokens": self.max_tokens, "temperature": temperature},
        ]

        # Fast path: use cached strategy
        if self._compat_strategy is not None:
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                **strategies[self._compat_strategy],
            )

        # Discovery: try each strategy
        last_err = None
        for idx, kwargs in enumerate(strategies):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                self._compat_strategy = idx
                logger.info(
                    "OpenAI compat: strategy %d works for %s (%s)",
                    idx, self.model, list(kwargs.keys()),
                )
                return response
            except Exception as err:
                err_msg = str(err)
                if ("max_completion_tokens" in err_msg
                        or "max_tokens" in err_msg
                        or "temperature" in err_msg):
                    last_err = err
                    continue
                raise
        raise last_err

    def _get_async_client(self):
        """Lazy-init an AsyncOpenAI client (same config as sync client)."""
        if not hasattr(self, "_async_client") or self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for async provider='openai'. "
                    "Install with: pip install openai"
                )
            kwargs: dict = {"api_key": self.api_key}
            if self.api_base_url:
                kwargs["base_url"] = self.api_base_url
            self._async_client = AsyncOpenAI(**kwargs)
            logger.info("AsyncOpenAI client initialized for model=%s", self.model)
        return self._async_client

    async def async_complete(
        self, messages: list[dict], temperature: float = 0.0
    ) -> str:
        """Async version of complete() — uses AsyncOpenAI client.

        Reuses the cached ``_compat_strategy`` discovered by the sync path.
        If no strategy is cached yet, does a sync discovery call first
        (one-time cost) so the async path doesn't need its own fallback chain.
        """
        client = self._get_async_client()

        # Ensure compat strategy is discovered (sync path does this lazily)
        if self._compat_strategy is None:
            # Do one sync call to discover the working strategy
            self.complete(messages, temperature)

        strategies = [
            {"max_completion_tokens": self.max_tokens, "temperature": temperature},
            {"max_completion_tokens": self.max_tokens},
            {"max_tokens": self.max_tokens, "temperature": temperature},
        ]
        kwargs = strategies[self._compat_strategy]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("Async OpenAI API call failed: %s", exc)
            raise


# ── Shared prompt / output helpers ────────────────────────────────────────────


def _build_messages(
    question: str, schema_context: str, mode: str = "standard"
) -> list[dict]:
    """
    Build an OpenAI-style messages list from question + schema + mode.

    Both backends consume this format:
      - _OpenAIBackend passes it directly to the API.
      - _LocalBackend converts it to ChatML via _messages_to_chatml().

    If schema_context is empty (e.g. a correction prompt already contains
    the full context), the question is treated as the complete user message.
    """
    if mode == "cot_plan":
        instruction = _COT_PLAN_INSTRUCTION
    elif mode == "divide_conquer":
        instruction = _DIVIDE_CONQUER_INSTRUCTION
    else:
        instruction = "Think step-by-step, then generate the SQL."

    if not schema_context.strip():
        # Correction prompt or bare question — pass through as-is
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

    user_content = (
        f"Question: {question}\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"{instruction}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def _messages_to_chatml(messages: list[dict]) -> str:
    """Convert messages list to ChatML string. Delegates to shared llms.chat_format."""
    from src.llms.chat_format import messages_to_chatml
    return messages_to_chatml(messages)


def _parse_output(raw: str) -> dict:
    """
    Parse raw model/API output into a result dict.

    Extraction order:
      1. ```sql ... ``` code block  (preferred — both OpenAI and local use this)
      2. Explicit "SQL:" marker
      3. First SELECT statement found in the text
      4. Fallback: last non-empty line
    """
    sql = _extract_sql(raw)
    reasoning = _extract_reasoning(raw)
    return {"sql": sql, "reasoning": reasoning, "raw_output": raw}


def _extract_sql(text: str) -> str:
    # 1. ```sql ... ``` block
    m = re.search(r"```sql\s*(.+?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Explicit "SQL:" marker (scan from the bottom)
    for line in reversed(text.strip().split("\n")):
        stripped = line.strip()
        if stripped.upper().startswith("SQL:"):
            return stripped[4:].strip()

    # 3. First SELECT statement
    m = re.search(r"(SELECT\s+.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        sql = re.split(r"\n\n|\n(?=[A-Z])", sql)[0]
        return sql.strip()

    # 4. Last non-empty line
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _extract_reasoning(text: str) -> str:
    # Qwen3 <think>...</think>
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Text before "SQL:"
    parts = re.split(r"(?i)\bSQL\s*:", text, maxsplit=1)
    if len(parts) > 1:
        return parts[0].strip()

    return ""

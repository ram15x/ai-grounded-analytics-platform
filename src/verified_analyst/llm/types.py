"""Request, response and error types shared by the router, adapters and cassettes."""

from __future__ import annotations

from dataclasses import dataclass, field

PUBLIC = "public"
INTERNAL = "internal"

# error kinds a provider adapter may report
RATE_LIMIT = "rate_limit"
SERVER = "server"
TIMEOUT = "timeout"
NETWORK = "network"
MALFORMED = "malformed"
AUTH = "auth"
BAD_REQUEST = "bad_request"
BLOCKED = "blocked"
NO_KEY = "no_key"
CONFIG = "config"
RETRYABLE = frozenset({RATE_LIMIT, SERVER, TIMEOUT, NETWORK, MALFORMED})


@dataclass(frozen=True)
class LLMRequest:
    """One model call. `datasets` names every table whose data appears in the messages.

    `sensitivity` is fail-closed: it defaults to internal and only the router stamps a request
    public, after resolving `datasets` against the configured tags.
    """

    task: str  # route | plan | compose | repair
    messages: tuple[tuple[str, str], ...]  # (role, content); role is system | user | assistant
    datasets: tuple[str, ...] = ()
    json_mode: bool = True
    temperature: float = 0.0
    max_tokens: int = 1024
    prompt_id: str = ""
    prompt_hash: str = ""
    schema_name: str = ""
    sensitivity: str = INTERNAL

    def prompt_chars(self) -> int:
        return sum(len(c) for _, c in self.messages)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    tokens_estimated: bool = False
    from_cassette: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(Exception):
    """Base class for everything the router raises on purpose."""


class ProviderError(LLMError):
    def __init__(
        self,
        kind: str,
        detail: str = "",
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail
        self.status = status
        self.retry_after = retry_after


class SensitivityRefusal(LLMError):
    """Internal data was about to reach a hosted provider, or no eligible provider exists."""


class BudgetExceeded(LLMError):
    """The per-run token cap would be exceeded. The pipeline turns this into abstain/budget."""


class CassetteMiss(LLMError):
    """Replay mode found no recorded response for the request. There is no network fallback."""


class CassetteLeak(LLMError):
    """A cassette line looked like it contained a credential, so nothing was written."""


@dataclass
class AllProvidersFailed(LLMError):
    attempts: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        kinds = ", ".join(f"{a['provider']}:{a['outcome']}" for a in self.attempts) or "none tried"
        return f"all providers failed ({kinds})"

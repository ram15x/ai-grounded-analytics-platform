"""HTTP transport. Adapters take a `Transport` callable so tests can mock it; the default uses
only the standard library. Nothing here logs headers or bodies."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field


class TransportTimeout(Exception):
    pass


class TransportError(Exception):
    """Connection failed (DNS, refused, reset, TLS)."""


@dataclass
class HttpResponse:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)  # keys lower-cased

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


# (url, headers, body, timeout_seconds) -> HttpResponse. POST only.
Transport = Callable[[str, dict[str, str], bytes, float], HttpResponse]


# (url, headers, timeout_seconds) -> HttpResponse. GET only, used for model listing.
Getter = Callable[[str, dict[str, str], float], HttpResponse]

RATE_HEADER_RE = re.compile(r"ratelimit|rate-limit|retry-after|quota|remaining", re.I)


def rate_headers(headers: dict[str, str]) -> dict[str, str]:
    """The subset of response headers that describe limits. Values are short, never credentials."""
    return {k: v[:80] for k, v in headers.items() if RATE_HEADER_RE.search(k)}


def user_agent() -> str:
    """An explicit, honest User-Agent. Python's default (`Python-urllib/3.x`) is blocked by some
    providers' edge firewalls (observed as an HTTP 403 on Groq while curl got 200; the cause is a
    hypothesis, not confirmed). Contains no key and no machine-specific data."""
    from .. import __version__  # noqa: PLC0415

    return f"verified-analyst/{__version__}"


def _fetch(req: urllib.request.Request, timeout: float) -> HttpResponse:
    if not req.has_header("User-agent"):
        req.add_header("User-Agent", user_agent())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return HttpResponse(
                resp.status, resp.read(), {k.lower(): v for k, v in resp.headers.items()}
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            exc.code, exc.read() or b"", {k.lower(): v for k, v in (exc.headers or {}).items()}
        )
    except TimeoutError as exc:
        raise TransportTimeout(str(exc) or "timed out") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TransportTimeout("timed out") from exc
        raise TransportError(type(exc.reason).__name__) from exc
    except OSError as exc:
        raise TransportError(type(exc).__name__) from exc


def urllib_get(url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
    if not url.startswith("https://"):
        raise TransportError("only https URLs are allowed")
    return _fetch(urllib.request.Request(url, headers=headers, method="GET"), timeout)  # noqa: S310


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> HttpResponse:
    if not url.startswith("https://"):
        raise TransportError("only https URLs are allowed")
    return _fetch(
        urllib.request.Request(url, data=body, headers=headers, method="POST"),  # noqa: S310
        timeout,
    )


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def loopback_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> HttpResponse:
    """POST to a model server on this computer only (for example Ollama). Plain http is allowed
    for loopback addresses and nowhere else; credentials in the URL are refused."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or parts.hostname not in _LOOPBACK_HOSTS:
        raise TransportError("only loopback http addresses are allowed here")
    if parts.username or parts.password:
        raise TransportError("credentials in a URL are not allowed")
    return _fetch(
        urllib.request.Request(url, data=body, headers=headers, method="POST"),  # noqa: S310
        timeout,
    )

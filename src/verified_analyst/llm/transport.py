"""HTTP transport. Adapters take a `Transport` callable so tests can mock it; the default uses
only the standard library. Nothing here logs headers or bodies."""

from __future__ import annotations

import urllib.error
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


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> HttpResponse:
    if not url.startswith("https://"):
        raise TransportError("only https URLs are allowed")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310
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

"""Inbound bearer-token authentication for the HTTP surfaces (v0.6.2).

CEmbedding exposes two HTTP surfaces and, until v0.6.2, neither had any
inbound authentication:

* the Starlette/uvicorn Streamable HTTP MCP app (``EMBEDDING_TRANSPORT=
  streamable-http``), which binds ``0.0.0.0`` by default;
* the aiohttp REST app (``/embed``, ``/index``, ``/search``, ``/remove``,
  ``/purge``) that runs alongside MCP stdio and binds loopback.

Both are covered here. A loopback bind is deliberately *not* treated as a
security boundary: a tunnel or reverse proxy forwards to loopback, so the
bind address says nothing about who can reach the process. That premise
already cost the sibling CPersona server 13 days of unauthenticated public
exposure, so the warning emitted when no token is configured never claims
that a loopback bind makes the endpoint safe.

Behaviour (additive, default-off — see ``CEMBEDDING_REQUIRE_AUTH``):

* ``CEMBEDDING_AUTH_TOKEN`` unset -> requests are served exactly as before
  and a warning is logged once per surface.
* ``CEMBEDDING_AUTH_TOKEN`` set   -> every request must carry
  ``Authorization: Bearer <token>``. Missing headers, wrong schemes and
  wrong tokens are all rejected with 401; there is no branch in which a
  configured token goes unchecked.
* ``CEMBEDDING_REQUIRE_AUTH=true`` -> starting without a token is a hard
  error instead of a warning. Opt-in in this release; the intent is to make
  it the default in a later, deliberately breaking one.

The middleware takes its token as a constructor argument rather than
reading module state, so tests exercise the real class at the ASGI/aiohttp
layer instead of a hand-copied replica.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping

logger = logging.getLogger(__name__)

AUTH_TOKEN_ENV = "CEMBEDDING_AUTH_TOKEN"
REQUIRE_AUTH_ENV = "CEMBEDDING_REQUIRE_AUTH"

_UNAUTHORIZED_BODY = b'{"error":"unauthorized"}'
_UNAUTHORIZED_HEADERS = [
    (b"content-type", b"application/json"),
    (b"www-authenticate", b"Bearer"),
]


class AuthConfigError(RuntimeError):
    """Raised when ``CEMBEDDING_REQUIRE_AUTH`` is set but no token is."""


def resolve_auth_token(env: Mapping[str, str] | None = None) -> str:
    """Read the configured inbound token. Whitespace-only means unset."""
    source = os.environ if env is None else env
    return source.get(AUTH_TOKEN_ENV, "").strip()


def require_auth_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return source.get(REQUIRE_AUTH_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def extract_bearer(header_value: str) -> str:
    """Return the credential from an ``Authorization`` value, or ``""``.

    Anything that is not exactly a ``Bearer`` scheme with a non-empty
    credential yields ``""``, which never compares equal to a configured
    token.
    """
    if not header_value:
        return ""
    scheme, _, credential = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return credential.strip()


def token_matches(presented: str, expected: str) -> bool:
    """Constant-time comparison. An empty presented token is always wrong."""
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def check_startup(surface: str, host: str, token: str, *, require: bool | None = None) -> None:
    """Log the auth posture of a surface, or refuse to start.

    ``host`` is reported verbatim and never interpreted as evidence of
    safety — see the module docstring.
    """
    if require is None:
        require = require_auth_enabled()
    if token:
        logger.info("%s: inbound bearer authentication enabled (bound to %s)", surface, host)
        return
    if require:
        raise AuthConfigError(
            f"{surface}: {REQUIRE_AUTH_ENV} is set but {AUTH_TOKEN_ENV} is empty. "
            f"Set {AUTH_TOKEN_ENV} or unset {REQUIRE_AUTH_ENV} to start unauthenticated."
        )
    logger.warning(
        "%s: UNAUTHENTICATED — no %s configured, so every request that reaches %s is served. "
        "A loopback bind is not evidence that requests are local: a tunnel or reverse proxy "
        "forwards to loopback. Set %s, or set %s=true to make this a startup error.",
        surface,
        AUTH_TOKEN_ENV,
        host,
        AUTH_TOKEN_ENV,
        REQUIRE_AUTH_ENV,
    )


class BearerTokenMiddleware:
    """ASGI middleware guarding the Streamable HTTP MCP app.

    ``OPTIONS`` is forwarded so CORS preflight still works: a preflight
    carries no ``Authorization`` header by design and the CORS layer answers
    it with headers only, never with tool data.
    """

    def __init__(self, app, auth_token: str = ""):
        self.app = app
        self.auth_token = auth_token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") == "OPTIONS" or not self.auth_token:
            await self.app(scope, receive, send)
            return

        header = ""
        for key, value in scope.get("headers") or ():
            if key.lower() == b"authorization":
                header = value.decode("latin-1")
                break

        if not token_matches(extract_bearer(header), self.auth_token):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": _UNAUTHORIZED_HEADERS,
                }
            )
            await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
            return

        await self.app(scope, receive, send)


def aiohttp_bearer_middleware(auth_token: str):
    """Build the aiohttp equivalent guarding the REST endpoints."""
    from aiohttp import web

    @web.middleware
    async def middleware(request: "web.Request", handler):
        if request.method == "OPTIONS":
            return await handler(request)
        if not token_matches(extract_bearer(request.headers.get("authorization", "")), auth_token):
            return web.json_response(
                {"error": "unauthorized"},
                status=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await handler(request)

    return middleware

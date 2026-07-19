"""Inbound authentication regression suite (v0.6.2).

The tests drive the real middleware classes — the ASGI one at the ASGI
layer and the aiohttp one through a live test server — rather than a
replica. When the sibling CScheduler server shipped this same class of
bypass, the middleware lived in a closure and no test could reach it.
"""

import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cembedding.auth import (
    AuthConfigError,
    BearerTokenMiddleware,
    aiohttp_bearer_middleware,
    check_startup,
    extract_bearer,
    require_auth_enabled,
    resolve_auth_token,
    token_matches,
)

TOKEN = "s3cret-token"


# ---------------------------------------------------------------- helpers


async def _asgi_call(app, headers=None, method="POST", scope_type="http"):
    """Drive an ASGI app once and return (status, body)."""
    scope = {
        "type": scope_type,
        "method": method,
        "path": "/embedding/mcp",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


async def _inner_app(scope, receive, send):
    """Stub 'protected resource' standing in for the MCP endpoint."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"TOOL-DATA"})


def _rest_client(auth_token):
    async def handler(_request):
        return web.json_response({"ok": True})

    middlewares = [aiohttp_bearer_middleware(auth_token)] if auth_token else []
    app = web.Application(middlewares=middlewares)
    app.router.add_post("/embed", handler)
    app.router.add_route("OPTIONS", "/embed", handler)
    return TestClient(TestServer(app))


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Bearer",
        "Bearer ",
        "Bearer   ",
        "Basic " + TOKEN,
        "Token " + TOKEN,
        TOKEN,
    ],
)
def test_extract_bearer_rejects_non_bearer_values(header):
    assert extract_bearer(header) == ""


def test_extract_bearer_accepts_the_scheme_case_insensitively():
    assert extract_bearer(f"Bearer {TOKEN}") == TOKEN
    assert extract_bearer(f"bearer {TOKEN}") == TOKEN
    assert extract_bearer(f"BEARER {TOKEN}") == TOKEN


def test_token_matches_is_false_for_empty_operands():
    assert not token_matches("", TOKEN)
    assert not token_matches(TOKEN, "")
    assert not token_matches("", "")
    assert token_matches(TOKEN, TOKEN)


def test_token_matches_handles_non_ascii_without_raising():
    assert token_matches("トークン", "トークン")
    assert not token_matches("トークン", TOKEN)


def test_resolve_auth_token_treats_whitespace_as_unset():
    assert resolve_auth_token({"CEMBEDDING_AUTH_TOKEN": "   "}) == ""
    assert resolve_auth_token({}) == ""
    assert resolve_auth_token({"CEMBEDDING_AUTH_TOKEN": f" {TOKEN} "}) == TOKEN


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("1", True), ("on", True), ("TRUE", True), ("false", False), ("", False), ("maybe", False)],
)
def test_require_auth_enabled_parsing(value, expected):
    assert require_auth_enabled({"CEMBEDDING_REQUIRE_AUTH": value}) is expected


# ---------------------------------------------------------------- startup


def test_check_startup_warns_without_claiming_loopback_is_safe(caplog):
    with caplog.at_level(logging.WARNING):
        check_startup("REST endpoint", "127.0.0.1:8401", "", require=False)
    message = caplog.text
    assert "UNAUTHENTICATED" in message
    # The premise that burned CPersona for 13 days must be contradicted, not
    # repeated: the warning may not imply a loopback bind is a boundary.
    assert "not evidence that requests are local" in message


def test_check_startup_refuses_to_start_when_auth_is_required():
    with pytest.raises(AuthConfigError):
        check_startup("Streamable HTTP MCP", "0.0.0.0:8403", "", require=True)


def test_check_startup_is_quiet_when_a_token_is_configured(caplog):
    with caplog.at_level(logging.WARNING):
        check_startup("Streamable HTTP MCP", "0.0.0.0:8403", TOKEN, require=True)
    assert caplog.text == ""


# ---------------------------------------------------- ASGI (MCP transport)


async def test_asgi_unconfigured_token_preserves_existing_behaviour():
    """v0.6.2 is additive: with no token set, nothing changes."""
    app = BearerTokenMiddleware(_inner_app, auth_token="")
    status, body = await _asgi_call(app)
    assert status == 200
    assert body == b"TOOL-DATA"


@pytest.mark.parametrize(
    "headers",
    [
        {},  # the bug-010 shape: no Authorization header at all
        {"authorization": ""},
        {"authorization": "Bearer"},
        {"authorization": "Bearer "},
        {"authorization": f"Basic {TOKEN}"},
        {"authorization": f"Bearer {TOKEN}x"},
        {"authorization": f"Bearer {TOKEN.upper()}"},
        {"authorization": "Bearer wrong"},
    ],
)
async def test_asgi_rejects_every_credential_that_is_not_the_token(headers):
    app = BearerTokenMiddleware(_inner_app, auth_token=TOKEN)
    status, body = await _asgi_call(app, headers)
    assert status == 401
    assert b"TOOL-DATA" not in body


async def test_asgi_accepts_the_configured_token():
    app = BearerTokenMiddleware(_inner_app, auth_token=TOKEN)
    status, body = await _asgi_call(app, {"authorization": f"Bearer {TOKEN}"})
    assert status == 200
    assert body == b"TOOL-DATA"


async def test_asgi_header_lookup_is_case_insensitive():
    app = BearerTokenMiddleware(_inner_app, auth_token=TOKEN)
    status, _ = await _asgi_call(app, {"Authorization": f"Bearer {TOKEN}"})
    assert status == 200


async def test_asgi_forwards_cors_preflight():
    app = BearerTokenMiddleware(_inner_app, auth_token=TOKEN)
    status, _ = await _asgi_call(app, method="OPTIONS")
    assert status == 200


async def test_asgi_passes_non_http_scopes_through():
    app = BearerTokenMiddleware(_inner_app, auth_token=TOKEN)
    status, body = await _asgi_call(app, scope_type="lifespan")
    assert status == 200
    assert body == b"TOOL-DATA"


# ---------------------------------------------------- aiohttp (REST /embed)


async def test_rest_unconfigured_token_preserves_existing_behaviour():
    async with _rest_client("") as client:
        response = await client.post("/embed", json={})
        assert response.status == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer wrong"},
    ],
)
async def test_rest_rejects_every_credential_that_is_not_the_token(headers):
    async with _rest_client(TOKEN) as client:
        response = await client.post("/embed", json={}, headers=headers)
        assert response.status == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_rest_accepts_the_configured_token():
    async with _rest_client(TOKEN) as client:
        response = await client.post("/embed", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status == 200
        assert await response.json() == {"ok": True}

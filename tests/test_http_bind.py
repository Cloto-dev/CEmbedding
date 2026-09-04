"""Where the REST endpoint binds, and whether its own log says so.

Until this suite existed the address was written into the socket call as a
literal, so ``/embed`` could only ever be reached from the machine running the
process. Inside a container that loopback is the container's own, and a peer
service -- the memory server this endpoint exists to serve -- cannot reach it
at all.

Both directions are measured. A test that only pins the environment arm stays
green if the default silently becomes "every interface", which is the change
nobody would want made for them.

The address is read off the call that opens the socket, not off the resolver,
because the defect being guarded against was a resolver whose answer never
reached the bind.
"""

import asyncio
import contextlib
import logging
import socket

import pytest
from aiohttp import web

from cembedding import server


def _free_port() -> int:
    """A port that was free a moment ago -- the usual, unavoidable race."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _wait_until_listening(host: str, port: int, timeout: float = 5.0) -> bool:
    """True once something accepts a connection on any address ``host`` names.

    Every address is tried because a name like ``localhost`` can resolve to
    IPv6, IPv4 or both depending on the host, and which one answers is not
    what these tests are about.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for family, socktype, proto, _canon, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
            with socket.socket(family, socktype, proto) as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(sockaddr) == 0:
                    return True
        await asyncio.sleep(0.05)
    return False


@contextlib.asynccontextmanager
async def _serving(monkeypatch, port: int, expect_host: str, **kwargs):
    """Run the real endpoint, recording the address handed to the socket."""
    recorded: dict[str, object] = {}
    real_site = web.TCPSite

    def recording_site(runner, host, site_port, *args, **kw):
        recorded["host"] = host
        recorded["port"] = site_port
        return real_site(runner, host, site_port, *args, **kw)

    monkeypatch.setattr(server.web, "TCPSite", recording_site)

    task = asyncio.create_task(server.run_http_server(port, **kwargs))
    try:
        listening = await _wait_until_listening(expect_host, port)
        assert listening, f"nothing accepted a connection on {expect_host}:{port}"
        yield recorded
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ------------------------------------------------------------------ resolver


def test_the_default_address_is_loopback(monkeypatch):
    monkeypatch.delenv(server.HTTP_HOST_ENV, raising=False)
    assert server.resolve_http_host() == "127.0.0.1"


def test_the_environment_names_the_address(monkeypatch):
    monkeypatch.setenv(server.HTTP_HOST_ENV, "0.0.0.0")
    assert server.resolve_http_host() == "0.0.0.0"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_a_blank_setting_is_not_an_address(value):
    """Matches how the auth token reads its own variable: blank means unset."""
    assert server.resolve_http_host({server.HTTP_HOST_ENV: value}) == "127.0.0.1"


# ------------------------------------------------------------- the real bind


async def test_an_unset_environment_still_binds_loopback(monkeypatch):
    monkeypatch.delenv(server.HTTP_HOST_ENV, raising=False)
    port = _free_port()
    async with _serving(monkeypatch, port, "127.0.0.1") as recorded:
        assert recorded["host"] == "127.0.0.1"
        assert recorded["port"] == port


async def test_the_environment_moves_the_address_that_is_bound(monkeypatch):
    """``localhost`` rather than ``0.0.0.0``: a distinct string is all the
    assertion needs, and binding every interface would ask the developer's
    firewall for permission in the middle of a test run."""
    monkeypatch.setenv(server.HTTP_HOST_ENV, "localhost")
    port = _free_port()
    async with _serving(monkeypatch, port, "localhost") as recorded:
        assert recorded["host"] == "localhost"


async def test_an_explicit_argument_outranks_the_environment(monkeypatch):
    monkeypatch.setenv(server.HTTP_HOST_ENV, "0.0.0.0")
    port = _free_port()
    async with _serving(monkeypatch, port, "127.0.0.1", host="127.0.0.1") as recorded:
        assert recorded["host"] == "127.0.0.1"


async def test_the_startup_lines_name_the_address_that_was_bound(monkeypatch, caplog):
    """A log that names a literal is worse than no log: it is evidence for a
    place the process is not listening. Both lines are checked -- the auth
    banner reports the surface it is guarding, and the endpoint line is what an
    operator copies into a client."""
    monkeypatch.setenv(server.HTTP_HOST_ENV, "localhost")
    monkeypatch.delenv("CEMBEDDING_AUTH_TOKEN", raising=False)
    port = _free_port()
    with caplog.at_level(logging.INFO):
        async with _serving(monkeypatch, port, "localhost"):
            pass

    banner = [record.getMessage() for record in caplog.records]
    assert any(f"localhost:{port}" in line for line in banner), banner
    assert any(f"http://localhost:{port}/embed" in line for line in banner), banner
    assert not any("127.0.0.1" in line for line in banner), banner

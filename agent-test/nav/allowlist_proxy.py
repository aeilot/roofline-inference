"""Minimal HTTPS CONNECT proxy that only tunnels to an allowlist of hosts.

The navigation-experiment agent runs inside a macOS sandbox profile that denies all network
access except to this proxy on localhost, so the agent can talk to its model API but cannot
reach anything else (e.g. this project's public GitHub repository).

    python agent-test/nav/allowlist_proxy.py --port 18080 --log runs/proxy.log
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt

ALLOW_SUFFIXES = ("openai.com", "chatgpt.com", "oaistatic.com", "oaiusercontent.com")


def allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(host == s or host.endswith("." + s) for s in ALLOW_SUFFIXES)


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def handle(reader, writer, log):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except Exception:
        writer.close()
        return
    line = head.split(b"\r\n", 1)[0].decode(errors="replace")
    parts = line.split()
    ok = len(parts) >= 2 and parts[0] == "CONNECT"
    host, port = (parts[1].rsplit(":", 1) + ["443"])[:2] if ok else ("", "0")
    verdict = "ALLOW" if ok and allowed(host) else "DENY"
    with open(log, "a") as f:
        f.write(f"{dt.datetime.now().isoformat()} {verdict} {line}\n")
    if verdict == "DENY":
        writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    try:
        ur, uw = await asyncio.open_connection(host, int(port))
    except Exception:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        writer.close()
        return
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(pipe(reader, uw), pipe(ur, writer))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    server = await asyncio.start_server(lambda r, w: handle(r, w, a.log), "127.0.0.1", a.port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

"""Bound inference request bodies before multipart parsing."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    """Reject oversized inference requests, including chunked request bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limited_paths = {"/api/v1/predict", "/api/v1/bulk-predict"}
        if scope["type"] != "http" or scope.get("path") not in limited_paths:
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value.decode("latin-1")
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_json(
                        send,
                        413,
                        b'{"detail":"Request exceeds the maximum allowed size."}',
                    )
                    return
            except ValueError:
                await self._send_json(
                    send, 400, b'{"detail":"Invalid Content-Length header."}'
                )
                return

        received = 0
        too_large = False
        response: list[Message] = []

        async def bounded_receive() -> Message:
            nonlocal received, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def buffered_send(message: Message) -> None:
            if not too_large:
                response.append(message)

        await self.app(scope, bounded_receive, buffered_send)
        if too_large:
            await self._send_json(
                send,
                413,
                b'{"detail":"Request exceeds the maximum allowed size."}',
            )
            return
        for message in response:
            await send(message)

    async def _send_json(self, send: Send, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

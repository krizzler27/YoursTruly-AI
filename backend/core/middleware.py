"""Request id middleware - generates and propagates correlation id."""

import uuid6
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from core.logging import clear_context, set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind X-Request-Id per request for logs and traces."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("X-Request-Id")
        try:
            request_id = incoming.strip() if incoming and incoming.strip() else uuid6.uuid7().hex
        except Exception:
            request_id = uuid6.uuid7().hex
        set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            clear_context()
        response.headers["X-Request-Id"] = request_id
        return response

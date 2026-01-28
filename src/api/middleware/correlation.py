import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.config.context import correlation_id_ctx


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-Id", str(uuid.uuid4()))

        # ✅ store correlation id in context var
        correlation_id_ctx.set(correlation_id)

        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-Id"] = correlation_id
        return response

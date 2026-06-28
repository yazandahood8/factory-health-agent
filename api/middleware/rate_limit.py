"""Per-tenant rate limiting (fixed-window).

Redis-backed when available, in-memory otherwise. Keyed by tenant so one noisy
customer cannot exhaust another's quota. Returns 429 with ``Retry-After``.
"""
from __future__ import annotations

import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from sdk.config import get_settings

OPEN_PATHS = {"/v1/health", "/docs", "/openapi.json", "/redoc", "/"}


class _InMemoryWindow:
    def __init__(self) -> None:
        self._hits: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)
        self._lock = threading.Lock()

    def incr(self, key: str, window: int) -> int:
        now = int(time.time())
        bucket = now - (now % window)
        with self._lock:
            start, count = self._hits.get(key, (bucket, 0))
            if start != bucket:
                start, count = bucket, 0
            count += 1
            self._hits[key] = (start, count)
            return count


class _RedisWindow:
    def __init__(self, url: str) -> None:
        import redis

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()

    def incr(self, key: str, window: int) -> int:
        now = int(time.time())
        bucket = now - (now % window)
        rkey = f"ratelimit:{key}:{bucket}"
        pipe = self._r.pipeline()
        pipe.incr(rkey)
        pipe.expire(rkey, window)
        count, _ = pipe.execute()
        return int(count)


class RateLimitMiddleware(BaseHTTPMiddleware):
    WINDOW_SECONDS = 60

    def __init__(self, app) -> None:
        super().__init__(app)
        settings = get_settings()
        self.limit = settings.rate_limit_per_minute
        if settings.redis_url:
            try:
                self.backend = _RedisWindow(settings.redis_url)
            except Exception:  # pragma: no cover
                self.backend = _InMemoryWindow()
        else:
            self.backend = _InMemoryWindow()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)

        tenant = getattr(request.state, "tenant", None)
        key = tenant.id if tenant else (request.client.host if request.client else "anon")
        count = self.backend.incr(key, self.WINDOW_SECONDS)
        if count > self.limit:
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(self.WINDOW_SECONDS)},
            )
        return await call_next(request)

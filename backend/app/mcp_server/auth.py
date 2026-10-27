import hmac

from starlette.responses import JSONResponse


class MCPAuth:
    def __init__(self, app, token: str, port: int):
        self.app = app
        self.expected = f"Bearer {token}".encode()
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self.origins = {f"http://{host}" for host in self.hosts}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rejection = self.rejection(scope)
        if rejection is not None:
            return await rejection(scope, receive, send)
        await self.app(scope, receive, send)

    def rejection(self, scope):
        headers = scope.get("headers", [])
        values = {}
        for key, value in headers:
            values.setdefault(key.lower(), []).append(value)
        host = values.get(b"host", [])
        origin = values.get(b"origin", [])
        if len(host) != 1 or host[0].decode("latin-1").lower() not in self.hosts or (
            origin and (len(origin) != 1 or origin[0].decode("latin-1") not in self.origins)
        ):
            return JSONResponse({"error": "FORBIDDEN_ORIGIN"}, status_code=403)
        auth = values.get(b"authorization", [])
        if len(auth) != 1 or not hmac.compare_digest(auth[0], self.expected):
            return JSONResponse(
                {"error": "UNAUTHORIZED"}, status_code=401,
                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
            )
        return None

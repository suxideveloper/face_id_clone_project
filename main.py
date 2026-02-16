from logging import debug
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import routes
import os
import logging

# Suppress "Invalid HTTP request received" warnings (caused by RTSP/network noise)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)


# --- Security Headers Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(settings.IMAGES_DIR, exist_ok=True)
    print(f"Server started [{settings.ENV}]. Data directory: {settings.DATA_DIR}")
    
    # Start attendance processor background task
    from app.api.routes import start_attendance_processor
    import asyncio
    asyncio.create_task(routes.process_attendance_queue())
    
    yield
    # Shutdown
    print("Server shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,   # Disable Swagger in production
    redoc_url="/redoc" if settings.DEBUG else None,  # Disable Redoc in production
)

# --- Middleware ---
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="data/images"), name="images")
# Mount attendance snapshots
os.makedirs("data/attendance_snapshots", exist_ok=True)
app.mount("/snapshots", StaticFiles(directory="data/attendance_snapshots"), name="snapshots")

# Include routes
app.include_router(routes.router)

if __name__ == "__main__":
    import uvicorn
    
    if settings.ENV == "prod":
        # Production: bind to localhost only (Nginx handles external traffic)
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8080,
            reload=False,
            workers=1,         # Single worker — camera is shared resource
            log_level="warning",
            access_log=False,
        )
    else:
        # Development: bind to all interfaces, enable hot reload
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            debug=True,
            log_level="debug",
        )

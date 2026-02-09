from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import routes
import os
import logging

# Suppress "Invalid HTTP request received" warnings (caused by RTSP/network noise)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(settings.IMAGES_DIR, exist_ok=True)
    print(f"Server started. Data directory: {settings.DATA_DIR}")
    
    # Start attendance processor background task
    from app.api.routes import start_attendance_processor
    import asyncio
    asyncio.create_task(routes.process_attendance_queue())
    
    yield
    # Shutdown
    print("Server shutting down...")

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

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
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

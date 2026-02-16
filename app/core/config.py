import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Employee Attendance System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment: "dev" or "prod"
    ENV: str = os.getenv("ENV", "dev")
    DEBUG: bool = ENV == "dev"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-use-a-strong-random-key")
    
    # CORS — comma-separated origins
    CORS_ORIGINS: list = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")
    ]
    
    # Allowed hosts (for future middleware)
    ALLOWED_HOSTS: list = [
        h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",")
    ]
    
    # Path settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    
    # Camera settings
    # Use 0 for webcam, or RTSP URL for IP camera
    # If CAMERA_ID is a digit string, convert to int, otherwise keep as string
    _camera_id_env = os.getenv("CAMERA_ID", "0")
    CAMERA_ID = int(_camera_id_env) if _camera_id_env.isdigit() else _camera_id_env
    
    # Model settings
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n-face.pt")

    # RTSP Settings — moved from hardcoded to .env
    RTSP_URL: str = os.getenv("RTSP_URL", "")

settings = Settings()

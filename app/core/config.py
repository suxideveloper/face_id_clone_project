import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Employee Attendance System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
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

    # RTSP Settings
    RTSP_URL = "rtsp://admin:Airidevs34@192.168.1.171:554/Streaming/Channels/302"

settings = Settings()

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHT_DIRS = (_PROJECT_ROOT, _PROJECT_ROOT.parent)

def _weight_path(name: str) -> str:

    for directory in _WEIGHT_DIRS:
        path = directory / name
        if path.exists():
            return str(path)
    return name


CROSSWALK_ZONE_COUNT = 5
ZONE_RESIDENCY_FRAMES = 2
ZONE_CONFIG_PATH = str(_PROJECT_ROOT / "data" / "zone_config.json")
TRACK_GRACE_FRAMES = 5  

TRACK_INHERIT_DISTANCE_CM = 8.0
TRACK_INHERIT_DISTANCE_PX = 50.0

CROSSWALK_REAL_WIDTH_CM = 19 
CROSSWALK_REAL_LENGTH_CM = 29.2

SPEED_WINDOW_SEC = 0.5
SPEED_MIN_SAMPLES = 2
SPEED_STOPPED_THRESHOLD_CM_S = 0.5


PEDESTRIAN_LABEL = "person"    
FOOT_POINT_OFFSET_RATIO = 0.1
DETECTION_MODEL_PATH = _weight_path("yolov8n-pose.pt")  
DETECTION_CONFIDENCE_THRESHOLD = 0.2
DETECTION_TRACKER = "bytetrack.yaml"
DETECTION_IMGSZ = 320

CAMERA_SOURCE = "picamera2"
CAMERA_RESOLUTION = (640, 480)
CAMERA_MOUNT_ANGLE_DEG = None #자동 측정   

SERIAL_PORT = None #자동 탐색
SERIAL_BAUDRATE = 115200   
SERIAL_MESSAGE_FORMAT = "ascii-lines"
SERIAL_READY_TIMEOUT_SEC = 5.0
SERIAL_STATE_HEARTBEAT_SEC = 1.0



FALL_CONFIG = {
    "crosswalk_roi": (0.15, 0.30, 0.85, 0.95),
    "fall_angle_deg": 10.0,      
    "fall_confirm_sec": 0.0,     
    "fall_gap_sec": 1.0,         
    "fall_gap_frames": 2,                                    
    "fall_gap_max_sec": 4.0,     
    "fall_clear_sec": 3.0,       
    "fall_aspect_ratio": 1.3,    
    "fall_roi_overlap": 0.3,     
    "track_grace_sec": 1.5,
    "track_grace_frames": 2,                                
    "track_grace_max_sec": 4.0,  
    "track_inherit_overlap": 0.3,   
}


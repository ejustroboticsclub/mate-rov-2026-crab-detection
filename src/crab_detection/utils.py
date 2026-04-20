from ultralytics import YOLO
from pathlib import Path
import gdown 
CACHE_DIR = Path.home() / ".cache/crab_detection_rov_2026"
MODEL_NAME = "european-crab-detection-model_v2.pt"  # make sure that each different model version is named so we can always change as we like from here
DEFAULT_MODEL_PATH = CACHE_DIR / MODEL_NAME


MODEL_URL = "https://drive.google.com/uc?id=1mlyFzMtdtb8WPMoMTdU25y_tRrRkafAX"


def get_model(path: str | None = None) -> YOLO:
    if path is not None:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        return YOLO(model_path)

    # always make the cache dir if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_MODEL_PATH.exists():
        _download_model(MODEL_NAME)

    return YOLO(DEFAULT_MODEL_PATH)


def _download_model(model_name: str) -> None:
    print(f"--> Downloading model {model_name} from {MODEL_URL} for crab detection")

    gdown.download(
        url=MODEL_URL,
        output=str(DEFAULT_MODEL_PATH),
        quiet=False
    )
    print("Model downloaded successfully")

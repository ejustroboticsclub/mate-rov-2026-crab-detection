from ultralytics import YOLO
from pathlib import Path

CACHE_DIR = Path.home() / ".cache/crab_detection_rov_2026"
MODEL_NAME = "trained_example_name_v1.pt"  # make sure that each different model version is named so we can always change as we like from here
DEFAULT_MODEL_PATH = CACHE_DIR / MODEL_NAME


# TODO: upload the model and add its url
def _todo(msg: str):
    raise NotImplementedError(msg)


MODEL_URL: str = _todo("MODEL_URL is not set yet")


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
    print(f"-->Downloading model {model_name} from {MODEL_URL} for crab detection")
    raise NotImplementedError

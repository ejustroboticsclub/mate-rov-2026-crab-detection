from .utils import get_model


class CrabDetector:
    # should initialize the required detection model even if download is required
    def __init__(self):
        self.model = get_model()
        raise NotImplementedError

    # TODO: annotate your return types clearly
    def detect(self) -> tuple:
        raise NotImplementedError

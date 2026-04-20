# mate-rov-2026-crab-detection
library for providing crab-detection model for task 2.1 in MATE-ROV-2026. the task includes finding the number of invasive crab species (European Green Crab) among two other species (Native Rock Crab and Native Jonah Crab) in an image that contains multiple resized and reoriented instances of the 3 crab species. Note that the exact same 3 images for each species which means we might detect using a YOLO model or a SIFT/ORB feature matcher(still measuring which is better).\
This library's goal is to provide an easy api for detection and NOT to build a ROS node. Building the ros node that takes the feed from the camera and delivers it to the GUI will be the responsiblity of the ROV workspace repository.\
The api is detailed below. After completing the solution in the repo we should tag the latest commit to be used as a mark when we add this project as a dependency to the ROV workspace.

## Required tools for development

### 1. uv
uv is the current best python project manager and we need it to sync dependencies across different projects. you can find how to install it from its [docs](https://docs.astral.sh/uv/). Learning how to use uv will make deployment much easier especially when we don't have time for solving dependency issues.

### 2. just (optional)
you will notice that I added the important commands in the Justfile. you can copy paste them in the terminal to run them normally or you can install just for doing something like:
```
just test
```
for testing with pytest.\
To install it you can see its official [website](https://just.systems/). or install it with `uv` using `uv tool install rust-just`.

## Getting Started

1. Clone the repository and cd into it
```bash
git clone https://github.com/ejustroboticsclub/mate-rov-2026-crab-detection.git crab-detection
cd crab-detection
```
1. run
```bash
uv sync
```

the above commands are enough to install all dependencies required for the project inside `.venv` file in root directory. it will also build the project as a library.

## Usage
This package is intended to be added as a project dependency using uv (or pixi for the rov workspace):
```bash
uv add git+https://github.com/ejustroboticsclub/mate-rov-2026-crab-detection
```
and then used inside the main workspace by importing it:
```python
from crab_detection import CrabDetector

crab_detector = CrabDetector() # installs required setup including downloading trained yolo model if Required
image,num_crab = crab_detector.detect(input_image) # takes camera image as input and returns annotated image and number of detections
```
NOTE that the API is still not determined and the above sample is just an example of what could be done.
### important points to mention:
- you can assume that the input image is a np.ndarray or a format that the yolo model is able to process.
- the output image should have the number of detections written on it on the top right or top left.
- the detections should be for only 1 crab type.
- the detector should know how to download the trained .pt model by itself from Gdrive or Github or however it may want but it can't rely on a pre-existing model to use.
- the detector should cache the model in a an absolute path not a relative one since it will be used as a library. A good place for cachining is "~/.cache/crab_detector_rov_2026". you can always have access that directory using the code in utils.py that uses `pathlib`.
- any training or testing code shouldn't be inside src to avoid shipping it with the library. so if anyone still wants to train then he/she should either create a `./training` directory that is self enclosed or just a `./scripts/train.py` and write how to invoke it in the Justfile using `just train`.

## TODOs for later
- we still need to implement actual business logic
- tests should be written for each image inside `tests/images`
- sanity check script should be written inside `scripts/laptop_webcam_detect.py` to test it freely.

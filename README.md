# Ruurd Photos ML

[![Python Quality Checks](https://github.com/RuurdBijlsma/ruurd-photos-ml/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/RuurdBijlsma/ruurd-photos-ml/actions/workflows/quality-checks.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ruurd Photos ML is a Python package providing machine learning tools for image analysis. It serves as the backend for the [Ruurd Photos](https://github.com/RuurdBijlsma/photos-backend) project and is designed to be integrated with Rust applications using [PyO3](https://pyo3.rs/).

## Features

This library provides interfaces for the following pre-trained models:

### Image Captioning
Generates descriptive captions for images and supports visual question answering.
* **InstructBLIP**: Generates detailed descriptions and answers questions about image content.
* **Salesforce BLIP**: Generates standard high-quality image captions.

### Facial Recognition
Detects and analyzes faces within images.
* **InsightFace**: A toolkit used for:
    * Multi-face detection.
    * Age and gender estimation.
    * Facial landmark identification (eyes, nose, mouth).
    * Facial embedding generation for clustering and recognition.

### Object Detection
Identifies and locates objects within an image.
* **ResNet**: Detects common objects and returns associated labels and bounding boxes.

### Optical Character Recognition (OCR)
Detects and extracts text from images.
* **ResNet & Tesseract**: A two-stage pipeline. ResNet first determines if an image contains legible text; Tesseract then extracts the text and bounding boxes.

## Installation

Install the package via pip:

```bash
pip install ruurd-photos-ml
```

## Usage

The following examples demonstrate the main functionalities of the library.

**Prerequisite: Loading an Image**

```python
from PIL import Image

image = Image.open("path/to/your/image.jpg")
```

### Image Captioning

```python
from ruurd_photos_ml import get_captioner, CaptionerProvider

# Initialize the captioner
captioner = get_captioner(CaptionerProvider.BLIP_INSTRUCT)

# Generate a caption
caption = captioner.caption(image)
print(f"Caption: {caption}")

# Visual Question Answering
question = "What color is the main object?"
answer = captioner.caption(image, instruction=question)
print(f"Answer: {answer}")
```

### Facial Recognition

```python
from ruurd_photos_ml import get_facial_recognition, FacialRecognitionProvider

# Initialize the facial recognition model
face_detector = get_facial_recognition(FacialRecognitionProvider.INSIGHT)

# Detect faces
faces = face_detector.get_faces(image)

for face in faces:
    print(f"Position: {face.position}, Confidence: {face.confidence}")
    print(f"  - Age: {face.age}")
    print(f"  - Gender: {face.sex}")
    print(f"  - Embedding: {face.embedding[:5]}...")
```

### Object Detection

```python
from ruurd_photos_ml import get_object_detection, ObjectDetectionProvider

# Initialize the object detector
object_detector = get_object_detection(ObjectDetectionProvider.RESNET)

# Detect objects
objects = object_detector.detect_objects(image)

for obj in objects:
    print(f"Detected '{obj.label}' with confidence {obj.confidence}")
```

### Optical Character Recognition (OCR)

```python
from ruurd_photos_ml import get_ocr, OCRProvider

# Initialize the OCR model
ocr = get_ocr(OCRProvider.RESNET_TESSERACT)

# Check for legible text before extraction
if ocr.has_legible_text(image):
    # Extract text string
    text = ocr.get_text(image, languages=("eng", "nld"))
    print(f"Extracted Text: {text}")

    # Extract text with bounding boxes
    boxes = ocr.get_boxes(image, languages=("eng", "nld"))
    for box in boxes:
        print(f"Found text: '{box.text}' at position {box.position}")
```

## Development

Follow these steps to set up a local development environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RuurdBijlsma/ruurd-photos-ml.git
   cd ruurd-photos-ml
   ```

2. **Install dependencies:**
   This project uses `uv` for dependency management.
   ```bash
   uv sync --all-extras --dev
   ```

3. **Run tests:**
   ```bash
   uv run pytest
   ```

4. **Run quality checks:**
   ```bash
   pre-commit run -a
   ```

## Project Links

* [Repository](https://github.com/RuurdBijlsma/ruurd-photos-ml)
* [Documentation](https://ruurdbijlsma.github.io/ruurd-photos-ml)

## License

This project is licensed under the MIT License.
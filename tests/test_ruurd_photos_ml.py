"""Tests for ruurd_photos_ml."""

import pytest

from ruurd_photos_ml import get_captioner, get_facial_recognition, get_object_detection, get_ocr
from ruurd_photos_ml.analysis.caption.protocol import CaptionerProvider
from ruurd_photos_ml.analysis.facial_recognition.protocol import FacialRecognitionProvider
from ruurd_photos_ml.analysis.object_detection.protocol import ObjectDetectionProvider
from ruurd_photos_ml.analysis.ocr.protocol import OCRProvider
from tests.helpers.get_test_image import get_test_image


def test_ocr() -> None:
    """Test OCR."""
    ocr_image = get_test_image("ocr.jpg")
    ocr = get_ocr(OCRProvider.RESNET_TESSERACT)
    has_text = ocr.has_legible_text(ocr_image)
    image_text = ocr.get_text(ocr_image, ("nld", "eng"))
    image_boxes = ocr.get_boxes(ocr_image, ("nld", "eng"))
    assert has_text
    assert "aldi" in image_text.lower()
    assert len(image_boxes) > 0


def test_object_detection() -> None:
    """Test object detection."""
    tent_image = get_test_image("tent.jpg")
    detector = get_object_detection(ObjectDetectionProvider.RESNET)
    objects = detector.detect_objects(tent_image)
    assert len(objects) > 0


def test_facial_recognition_1() -> None:
    """Test facial recognition on an image with 1 face."""
    walter = get_test_image("walter.jpg")
    detector = get_facial_recognition(FacialRecognitionProvider.INSIGHT)
    faces = detector.get_faces(walter)
    expected_face_count = 1
    assert len(faces) == expected_face_count


def test_facial_recognition_2() -> None:
    """Test facial recognition on an image with 15 faces."""
    many_faces = get_test_image("faces.webp")
    detector = get_facial_recognition(FacialRecognitionProvider.INSIGHT)
    faces = detector.get_faces(many_faces)
    expected_face_count = 15
    assert len(faces) == expected_face_count


@pytest.mark.cuda
def test_captioner_blip_instruct() -> None:
    """Test blip instruct captioner."""
    horse_image = get_test_image("paard.jpg")
    captioner = get_captioner(CaptionerProvider.BLIP_INSTRUCT)
    caption = captioner.caption(horse_image)
    assert "horse" in caption.lower()
    assert "sand" in caption.lower()
    is_animal = captioner.caption(horse_image, "Is this an animal? yes or no.")
    assert "yes" in is_animal.lower()
    animal_type = captioner.caption(horse_image, "What animal is this?")
    assert "horse" in animal_type.lower()


def test_captioner_sf() -> None:
    """Test salesforce captioner."""
    horse_image = get_test_image("paard.jpg")
    captioner = get_captioner(CaptionerProvider.SF_BLIP)
    caption = captioner.caption(horse_image)
    assert "horse" in caption.lower()
    # sf captioner can't really do question & answer, so it's not tested.

"""Tests for ruurd_photos_ml."""

import numpy as np
import pytest

from ruurd_photos_ml import (
    CaptionerProvider,
    EmbedderProvider,
    FacialRecognitionProvider,
    ObjectDetectionProvider,
    OCRProvider,
    get_captioner,
    get_embedder,
    get_facial_recognition,
    get_object_detection,
    get_ocr,
)
from tests.helpers.get_test_image import get_test_image


@pytest.mark.parametrize("provider", [EmbedderProvider.OPEN_CLIP, EmbedderProvider.ZERO_CLIP])
def test_embedder_sorted(provider: EmbedderProvider) -> None:
    """Test embedder and print texts sorted by their similarity to the image."""
    tent_image = get_test_image("tent.jpg")
    embedder = get_embedder(provider)
    image_embedding = embedder.embed_image(tent_image)

    texts = [
        "an image of a campsite with a car parked.",
        "A tent",
        "a beautiful sunset over a city next to a sea.",
        "A sunset over the ocean.",
        "An portrait of an old man.",
        "A portrait of walter white from breaking bad",
        "An image of a horse laying on a sand floor, with grass in the background.",
        "Image of a horse.",
    ]
    text_embeddings = embedder.embed_texts(texts)

    # Calculate cosine similarity (dot product).
    # Since the embedder already normalizes the vectors, this is sufficient.
    similarities = np.dot(text_embeddings, image_embedding.T)

    # Combine the texts and their similarity scores
    scored_texts = list(zip(texts, similarities, strict=False))

    # Sort the list of (text, score) tuples in descending order based on the score
    sorted_scored_texts = sorted(scored_texts, key=lambda item: item[1], reverse=True)

    print("Texts sorted by similarity to the image:")
    # Print the sorted list
    for text, score in sorted_scored_texts:
        print(f"Score: {score:.4f} - '{text}'")

    assert "campsite" in sorted_scored_texts[0][0]
    assert "tent" in sorted_scored_texts[1][0]


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
    is_animal = captioner.caption(horse_image, "Question: Is this an animal? yes or no. Answer:")
    assert "yes" in is_animal.lower()
    animal_type = captioner.caption(horse_image, "Question: What animal is this? Answer:")
    assert "horse" in animal_type.lower()



@pytest.mark.cuda
def test_captioner_sf() -> None:
    """Test salesforce captioner."""
    horse_image = get_test_image("paard.jpg")
    captioner = get_captioner(CaptionerProvider.SF_BLIP)
    caption = captioner.caption(horse_image)
    assert "horse" in caption.lower()
    # sf captioner can't really do question & answer, so it's not tested.

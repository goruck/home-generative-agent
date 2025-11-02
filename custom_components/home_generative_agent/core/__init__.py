"""Home Generative Agent core module."""

from .image_entity import ImageEntity
from .migrations import migrate_person_gallery
from .person_gallery import PersonGalleryDAO
from .recognized_sensor import RecognizedPeopleSensor
from .runtime import HGAData
from .utils import (
    CannotConnectError,
    InvalidAuthError,
    default_mobile_notify_service,
    discover_mobile_notify_service,
    dispatch_on_loop,
    ensure_http_url,
    gemini_healthy,
    generate_embeddings,
    list_mobile_notify_services,
    ollama_healthy,
    openai_healthy,
    reasoning_field,
    validate_db_uri,
    validate_face_api_url,
    validate_gemini_key,
    validate_ollama_url,
    validate_openai_key,
)
from .video_analyzer import VideoAnalyzer
from .video_helpers import (
    crop_resize_encode_jpeg,
    dedupe_desc,
    dhash_bytes,
    ensure_dir,
    epoch_from_path,
    hamming64,
    latest_target,
    load_image_rgb,
    mirror_to_www,
    order_batch,
    publish_latest_atomic,
    put_with_backpressure,
)

__all__ = [
    "CannotConnectError",
    "HGAData",
    "ImageEntity",
    "InvalidAuthError",
    "PersonGalleryDAO",
    "RecognizedPeopleSensor",
    "VideoAnalyzer",
    "crop_resize_encode_jpeg",
    "dedupe_desc",
    "default_mobile_notify_service",
    "dhash_bytes",
    "discover_mobile_notify_service",
    "dispatch_on_loop",
    "ensure_dir",
    "ensure_http_url",
    "epoch_from_path",
    "gemini_healthy",
    "generate_embeddings",
    "hamming64",
    "latest_target",
    "list_mobile_notify_services",
    "load_image_rgb",
    "migrate_person_gallery",
    "mirror_to_www",
    "ollama_healthy",
    "openai_healthy",
    "order_batch",
    "publish_latest_atomic",
    "put_with_backpressure",
    "reasoning_field",
    "validate_db_uri",
    "validate_face_api_url",
    "validate_gemini_key",
    "validate_ollama_url",
    "validate_openai_key",
]

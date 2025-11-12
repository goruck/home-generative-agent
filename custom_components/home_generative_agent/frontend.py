"""Frontend resource management for Home Generative Agent."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.components.http import StaticPathConfig
from homeassistant.helpers.httpx_client import get_async_client

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# GitHub repository configuration
CARD_REPO_OWNER = "lemming1337"
CARD_REPO_NAME = "homeassistant-assist-card"
CARD_REPO = f"{CARD_REPO_OWNER}/{CARD_REPO_NAME}"
CARD_FILENAME = "homeassistant-assist-card.js"

# Version to download (can be updated to track latest)
CARD_VERSION = "v0.0.1"

# Local cache configuration
CACHE_DIR_NAME = "homeassistant-assist-card"


def _get_cache_dir(hass: HomeAssistant) -> Path:
    """Get the cache directory for frontend resources."""
    cache_dir = Path(hass.config.path("custom_components", "home_generative_agent", "www_cache", CACHE_DIR_NAME))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cached_file_path(hass: HomeAssistant) -> Path:
    """Get the path to the cached card file."""
    return _get_cache_dir(hass) / CARD_FILENAME


def _get_version_file_path(hass: HomeAssistant) -> Path:
    """Get the path to the version tracking file."""
    return _get_cache_dir(hass) / "version.txt"


def _get_download_url() -> str:
    """Get the download URL for the card from GitHub releases."""
    return f"https://github.com/{CARD_REPO}/releases/download/{CARD_VERSION}/{CARD_FILENAME}"


async def _download_card(hass: HomeAssistant) -> bool:
    """Download the card from GitHub releases.

    Returns:
        True if download was successful, False otherwise.
    """
    url = _get_download_url()
    cached_file = _get_cached_file_path(hass)
    version_file = _get_version_file_path(hass)

    LOGGER.info("Downloading assist card from %s", url)

    try:
        client = get_async_client(hass, verify_ssl=True)
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()

        # Write the downloaded content
        content = response.content
        cached_file.write_bytes(content)

        # Calculate and log checksum for verification
        sha256_hash = hashlib.sha256(content).hexdigest()
        LOGGER.info(
            "Downloaded %s (%d bytes, SHA256: %s)",
            CARD_FILENAME,
            len(content),
            sha256_hash[:16]
        )

        # Save version information
        version_file.write_text(f"{CARD_VERSION}\n{sha256_hash}")

        return True

    except Exception as err:
        LOGGER.error("Failed to download assist card from %s: %s", url, err)
        return False


async def _ensure_card_available(hass: HomeAssistant) -> bool:
    """Ensure the card file is available locally.

    Downloads if not cached or if version mismatch.

    Returns:
        True if card is available, False otherwise.
    """
    cached_file = _get_cached_file_path(hass)
    version_file = _get_version_file_path(hass)

    # Check if we need to download
    needs_download = False

    if not cached_file.exists():
        LOGGER.info("Assist card not found in cache, will download")
        needs_download = True
    elif not version_file.exists():
        LOGGER.info("Version file not found, will re-download assist card")
        needs_download = True
    else:
        # Check version
        try:
            cached_version = version_file.read_text().strip().split("\n")[0]
            if cached_version != CARD_VERSION:
                LOGGER.info(
                    "Assist card version mismatch (cached: %s, required: %s), will update",
                    cached_version,
                    CARD_VERSION
                )
                needs_download = True
        except Exception as err:
            LOGGER.warning("Failed to read version file: %s, will re-download", err)
            needs_download = True

    if needs_download:
        if not await _download_card(hass):
            return False
    else:
        LOGGER.debug("Using cached assist card version %s", CARD_VERSION)

    return cached_file.exists()


async def async_register_frontend(hass: HomeAssistant) -> bool:
    """Register frontend resources for the custom assist card.

    Downloads the card from GitHub releases if needed and registers
    it as a static resource.

    Returns:
        True if registration was successful, False otherwise.
    """
    # Only register once
    if "_hga_frontend_registered" in hass.data:
        return True

    # Ensure card is available (download if needed)
    if not await _ensure_card_available(hass):
        LOGGER.error("Failed to make assist card available")
        return False

    # Get cache directory for static path registration
    cache_dir = _get_cache_dir(hass)

    # Register static path
    try:
        await hass.http.async_register_static_paths([
            StaticPathConfig(
                "/home_generative_agent",
                str(cache_dir),
                False,  # Disable cache for easier updates
            )
        ])

        hass.data["_hga_frontend_registered"] = True

        LOGGER.info(
            "Registered assist card frontend resources at /home_generative_agent -> %s (version %s)",
            cache_dir,
            CARD_VERSION,
        )

        return True

    except Exception as err:
        LOGGER.error("Failed to register frontend resources: %s", err)
        return False

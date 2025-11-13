"""Travel times tool using Google Maps Routes API for Home Generative Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

import httpx
from homeassistant.exceptions import HomeAssistantError
from langchain_core.runnables import RunnableConfig  # noqa: TC002
from langchain_core.tools import InjectedToolArg, tool

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

LOGGER = logging.getLogger(__name__)

# Google Maps Routes API configuration
ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_API_TIMEOUT = 30  # seconds
DEFAULT_TRAVEL_MODE = "DRIVE"
DEFAULT_ROUTING_PREFERENCE = "TRAFFIC_AWARE"


async def _get_google_maps_api_key(hass: HomeAssistant) -> str | None:
    """Get Google Maps API key from integration configuration.

    Args:
        hass: Home Assistant instance

    Returns:
        API key if configured, None otherwise
    """
    entry_data = hass.config_entries.async_entries("home_generative_agent")
    if not entry_data:
        return None

    entry = entry_data[0]
    merged_config = {**entry.data, **entry.options}
    return merged_config.get("google_maps_api_key")


async def _compute_route(
    api_key: str,
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    travel_mode: str = DEFAULT_TRAVEL_MODE,
    routing_preference: str = DEFAULT_ROUTING_PREFERENCE,
) -> dict:
    """Call Google Maps Routes API to compute a route.

    Args:
        api_key: Google Maps API key
        origin_lat: Origin latitude
        origin_lng: Origin longitude
        destination_lat: Destination latitude
        destination_lng: Destination longitude
        travel_mode: Travel mode (DRIVE, TRANSIT, WALK, BICYCLE, TWO_WHEELER)
        routing_preference: Routing preference (TRAFFIC_AWARE, TRAFFIC_AWARE_OPTIMAL)

    Returns:
        Route data including duration and distance

    Raises:
        HomeAssistantError: If API call fails
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.description",
    }

    request_body = {
        "origin": {
            "location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}
        },
        "destination": {
            "location": {
                "latLng": {"latitude": destination_lat, "longitude": destination_lng}
            }
        },
        "travelMode": travel_mode,
        "routingPreference": routing_preference,
        "languageCode": "en-US",
        "units": "METRIC",
    }

    try:
        async with httpx.AsyncClient(timeout=ROUTES_API_TIMEOUT) as client:
            response = await client.post(
                ROUTES_API_URL,
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("routes"):
                msg = "No routes found for the given origin and destination"
                raise HomeAssistantError(msg)

            return data["routes"][0]

    except httpx.HTTPStatusError as err:
        msg = f"Google Maps API error: {err.response.status_code} - {err.response.text}"
        LOGGER.error(msg)
        raise HomeAssistantError(msg) from err
    except httpx.RequestError as err:
        msg = f"Network error calling Google Maps API: {err}"
        LOGGER.error(msg)
        raise HomeAssistantError(msg) from err
    except Exception as err:
        msg = f"Unexpected error calling Google Maps API: {err}"
        LOGGER.exception(msg)
        raise HomeAssistantError(msg) from err


def _format_duration(duration_seconds: int) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60

    if hours > 0:
        return f"{hours}h {minutes}min"
    return f"{minutes}min"


def _format_distance(distance_meters: int) -> str:
    """Format distance in meters to human-readable string.

    Args:
        distance_meters: Distance in meters

    Returns:
        Formatted distance string
    """
    if distance_meters >= 1000:
        km = distance_meters / 1000
        return f"{km:.1f} km"
    return f"{distance_meters} m"


@tool(parse_docstring=True)
async def get_travel_time(  # noqa: D417
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: float,
    destination_longitude: float,
    travel_mode: str = DEFAULT_TRAVEL_MODE,
    *,
    # Hide these arguments from the model.
    config: Annotated[RunnableConfig, InjectedToolArg()],
) -> str:
    """
    Get travel time and distance between two locations using Google Maps.

    This tool computes the optimal route between an origin and destination,
    taking into account current traffic conditions. It provides both the
    estimated travel time and distance.

    Args:
        origin_latitude: Latitude of the starting location (e.g., 37.7749).
        origin_longitude: Longitude of the starting location (e.g., -122.4194).
        destination_latitude: Latitude of the destination (e.g., 37.3382).
        destination_longitude: Longitude of the destination (e.g., -121.8863).
        travel_mode: Mode of transportation. Valid options are DRIVE (default, by car),
            TRANSIT (public transportation), WALK (walking), BICYCLE (cycling),
            or TWO_WHEELER (motorcycle/scooter).

    """
    if "configurable" not in config:
        return "Configuration not found. Please check your setup."

    hass: HomeAssistant = config["configurable"]["hass"]

    # Get API key from configuration
    api_key = await _get_google_maps_api_key(hass)
    if not api_key:
        return (
            "Google Maps API key not configured. Please add 'google_maps_api_key' "
            "to the integration configuration."
        )

    # Validate travel mode
    valid_modes = ["DRIVE", "TRANSIT", "WALK", "BICYCLE", "TWO_WHEELER"]
    travel_mode_upper = travel_mode.upper()
    if travel_mode_upper not in valid_modes:
        return (
            f"Invalid travel mode '{travel_mode}'. "
            f"Valid options are: {', '.join(valid_modes)}"
        )

    # Validate coordinates
    if not (-90 <= origin_latitude <= 90) or not (-180 <= origin_longitude <= 180):
        return "Invalid origin coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."

    if not (-90 <= destination_latitude <= 90) or not (
        -180 <= destination_longitude <= 180
    ):
        return "Invalid destination coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."

    try:
        # Determine routing preference based on travel mode
        routing_pref = (
            DEFAULT_ROUTING_PREFERENCE if travel_mode_upper == "DRIVE" else ""
        )

        route = await _compute_route(
            api_key=api_key,
            origin_lat=origin_latitude,
            origin_lng=origin_longitude,
            destination_lat=destination_latitude,
            destination_lng=destination_longitude,
            travel_mode=travel_mode_upper,
            routing_preference=routing_pref,
        )

        # Extract duration and distance
        duration_str = route.get("duration", "")
        distance_meters = route.get("distanceMeters", 0)

        # Parse duration (format: "1234s")
        duration_seconds = 0
        if duration_str.endswith("s"):
            try:
                duration_seconds = int(duration_str[:-1])
            except ValueError:
                LOGGER.warning("Could not parse duration: %s", duration_str)

        # Format response
        duration_formatted = _format_duration(duration_seconds)
        distance_formatted = _format_distance(distance_meters)

        mode_description = {
            "DRIVE": "by car",
            "TRANSIT": "by public transit",
            "WALK": "walking",
            "BICYCLE": "by bicycle",
            "TWO_WHEELER": "by motorcycle",
        }.get(travel_mode_upper, travel_mode_upper.lower())

        return (
            f"Travel time {mode_description}: {duration_formatted} "
            f"({distance_formatted})"
        )

    except HomeAssistantError as err:
        return f"Error getting travel time: {err}"
    except Exception as err:
        LOGGER.exception("Unexpected error in get_travel_time")
        return f"Unexpected error getting travel time: {err}"

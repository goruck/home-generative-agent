"""Home Generative Agent Initalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from homeassistant.const import Platform
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN
from .coordinator import HomeGenerativeAgentCoordinator

if TYPE_CHECKING:
    from collections.abc import Callable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.device_registry import DeviceEntry
    from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]


@dataclass
class RuntimeData:
    """Class to hold our data."""

    coordinator: DataUpdateCoordinator
    cancel_update_listener: Callable


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up Integration from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Initialize the coordinator that manages data updates from your api.
    # This is defined in coordinator.py
    coordinator = HomeGenerativeAgentCoordinator(hass, config_entry)

    # Perform an initial data load from api.
    # "async_config_entry_first_refresh()"" is special
    # in that it does not log errors if it fails
    await coordinator.async_config_entry_first_refresh()

    # Test to see if api initialised correctly, else raise
    # ConfigNotReady to make HA retry setup
    # TODO: Change this to match how api will know if connected or successful update
    _LOGGER.debug("API connected status: %s", coordinator.api.connected)
    if not coordinator.api.connected:
        _LOGGER.error("API connection failed, retrying...")
        raise ConfigEntryNotReady


    # Initialise a listener for config flow options changes.
    # See config_flow for defining an options setting that shows
    # up as configure on the integration.
    cancel_update_listener = config_entry.add_update_listener(_async_update_listener)

    # Add the coordinator and update listener to hass data to make
    # accessible throughout your integration
    # Note: this will change on HA2024.6 to save on the config entry.
    hass.data[DOMAIN][config_entry.entry_id] = RuntimeData(
        coordinator, cancel_update_listener
    )

    # Setup platforms (based on the list of entity types in PLATFORMS defined above)
    # This calls the async_setup method in each of your entity type files.
    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    # Return true to denote a successful setup.
    return True


async def _async_update_listener(hass: HomeAssistant, config_entry: ConfigEntry) -> None:
    """Handle config options update."""
    # Reload the integration when the options change.
    await hass.config_entries.async_reload(config_entry.entry_id)


async def async_remove_config_entry_device(
    hass: HomeAssistant, config_entry: ConfigEntry, device_entry: DeviceEntry
) -> bool:
    """Delete device if selected from UI."""
    # Adding this function shows the delete device option in the UI.
    # Remove this function if you do not want that option.
    # You may need to do some checks here before allowing devices to be removed.
    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # This is called when you remove your integration or shutdown HA.
    # If you have created any custom services, they need to be removed here too.

    # Remove the config options update listener
    hass.data[DOMAIN][config_entry.entry_id].cancel_update_listener()

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(
        config_entry, PLATFORMS
    )

    # Remove the config entry from the hass data object.
    if unload_ok:
        hass.data[DOMAIN].pop(config_entry.entry_id)

    # Return that unloading was successful.
    return unload_ok

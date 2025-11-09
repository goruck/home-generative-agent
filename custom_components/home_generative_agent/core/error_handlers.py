"""Centralized error handling utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.exceptions import HomeAssistantError
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from ..agent.graph import ToolErrorType
    from ..agent.tool_metrics import ToolCallMetrics

LOGGER = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for tools and async operations."""

    @staticmethod
    async def execute_with_standard_handling(
        coro: Coroutine,
        operation_name: str,
        timeout: float | None = None,
        metric: ToolCallMetrics | None = None,
        error_type_classifier: type[ToolErrorType] | None = None,
    ) -> tuple[Any | None, Exception | None]:
        """
        Execute async operation with standard error handling.

        Args:
            coro: The coroutine to execute
            operation_name: Name for logging purposes
            timeout: Optional timeout in seconds
            metric: Optional metric to update on success/failure
            error_type_classifier: Optional ToolErrorType enum for classification

        Returns:
            (result, error) - one will be None

        """
        try:
            if timeout:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            if metric:
                response_size = len(str(result)) if result else 0
                metric.finalize(success=True, response_size_bytes=response_size)

            return result, None

        except TimeoutError as err:
            LOGGER.warning("%s timed out after %.1fs", operation_name, timeout)
            if metric:
                metric.finalize(
                    success=False,
                    error_type="timeout",
                    error_message=f"Operation timed out after {timeout}s",
                )
            return None, err

        except (HomeAssistantError, ValidationError, ValueError, TypeError) as err:
            error_type = (
                error_type_classifier.classify(err)
                if error_type_classifier
                else "validation"
            )
            LOGGER.warning("%s failed: %s", operation_name, err)
            if metric:
                metric.finalize(
                    success=False,
                    error_type=error_type.value
                    if hasattr(error_type, "value")
                    else str(error_type),
                    error_message=str(err),
                )
            return None, err

        except Exception as err:
            LOGGER.exception("Unexpected error in %s", operation_name)
            if metric:
                metric.finalize(
                    success=False,
                    error_type="execution",
                    error_message=str(err),
                )
            return None, err

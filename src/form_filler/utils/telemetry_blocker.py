#!/usr/bin/env python3
"""
Telemetry Blocker for CrewAI and LangChain.

This module provides functions to block telemetry and tracking
by monkey patching network-related functions.
"""

import logging
import os
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


def block_telemetry():
    """
    Block telemetry by patching network modules and setting environment variables.

    This function:
    1. Sets environment variables to disable tracking
    2. Patches urllib3, requests, aiohttp to block connections to telemetry endpoints
    3. Patches specific modules in CrewAI and LangChain that handle telemetry
    """
    # Set environment variables
    os.environ["CREWAI_DO_NOT_TRACK"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACKING"] = "false"

    logger.info("Setting environment variables to disable telemetry")

    # Track original functions
    patched_functions = {}

    # List of telemetry domains to block
    telemetry_domains = [
        "telemetry.crewai.com",
        "api.segment.io",
        "api.mixpanel.com",
        "api.amplitude.com",
        "telemetry.langchain.com",
    ]

    # Patch urllib3 connection
    try:
        import urllib3

        def patched_https_request(self, method, url, *args, **kwargs):
            """Block requests to telemetry endpoints."""
            for domain in telemetry_domains:
                if domain in self.host:
                    logger.debug(f"Blocked telemetry request to {self.host}")
                    return None

            return original_request(self, method, url, *args, **kwargs)

        # Store original method
        original_request = urllib3.connection.HTTPSConnection.request
        patched_functions["urllib3.connection.HTTPSConnection.request"] = original_request

        # Apply patch
        urllib3.connection.HTTPSConnection.request = patched_https_request
        logger.info("Patched urllib3 to block telemetry requests")
    except ImportError:
        logger.debug("urllib3 not found, skipping patch")

    # Patch requests library if available
    try:
        import requests

        original_requests_request = requests.Session.request

        def patched_requests_request(self, method, url, *args, **kwargs):
            """Block requests to telemetry endpoints."""
            for domain in telemetry_domains:
                if domain in url:
                    logger.debug(f"Blocked telemetry request to {url}")
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = ""
                    return mock_response

            return original_requests_request(self, method, url, *args, **kwargs)

        # Store original method
        patched_functions["requests.Session.request"] = original_requests_request

        # Apply patch
        requests.Session.request = patched_requests_request
        logger.info("Patched requests to block telemetry requests")
    except ImportError:
        logger.debug("requests not found, skipping patch")

    # Patch aiohttp if available
    try:
        import aiohttp

        original_aiohttp_request = aiohttp.ClientSession._request

        async def patched_aiohttp_request(self, method, url, *args, **kwargs):
            """Block requests to telemetry endpoints."""
            url_str = str(url)
            for domain in telemetry_domains:
                if domain in url_str:
                    logger.debug(f"Blocked telemetry request to {url_str}")
                    mock_response = MagicMock()
                    mock_response.status = 200
                    mock_response.text = async_mock_text
                    mock_response.release.return_value = None
                    mock_response.__aenter__.return_value = mock_response
                    mock_response.__aexit__.return_value = None
                    return mock_response

            return await original_aiohttp_request(self, method, url, *args, **kwargs)

        async def async_mock_text():
            return ""

        # Store original method
        patched_functions["aiohttp.ClientSession._request"] = original_aiohttp_request

        # Apply patch
        aiohttp.ClientSession._request = patched_aiohttp_request
        logger.info("Patched aiohttp to block telemetry requests")
    except ImportError:
        logger.debug("aiohttp not found, skipping patch")

    # Try to patch CrewAI telemetry directly
    try:
        import crewai.trackers.posthog

        # Replace the capture method with a no-op
        crewai.trackers.posthog.Posthog.capture = lambda *args, **kwargs: None
        logger.info("Patched CrewAI's Posthog tracker")
    except (ImportError, AttributeError):
        logger.debug("CrewAI Posthog tracker not found, skipping patch")

    # Try to patch LangChain telemetry
    try:
        import langchain.callbacks.tracers.langchain

        # Replace the _persist_event method with a no-op
        langchain.callbacks.tracers.langchain.LangChainTracer._persist_event = (
            lambda *args, **kwargs: None
        )
        logger.info("Patched LangChain tracer")
    except (ImportError, AttributeError):
        logger.debug("LangChain tracer not found, skipping patch")

    logger.info("Telemetry blocking enabled for privacy protection")
    return patched_functions


def restore_original_functions(patched_functions):
    """Restore original functions after patching."""
    # Restore urllib3
    try:
        import urllib3

        if "urllib3.connection.HTTPSConnection.request" in patched_functions:
            urllib3.connection.HTTPSConnection.request = patched_functions[
                "urllib3.connection.HTTPSConnection.request"
            ]
    except ImportError:
        pass

    # Restore requests
    try:
        import requests

        if "requests.Session.request" in patched_functions:
            requests.Session.request = patched_functions["requests.Session.request"]
    except ImportError:
        pass

    # Restore aiohttp
    try:
        import aiohttp

        if "aiohttp.ClientSession._request" in patched_functions:
            aiohttp.ClientSession._request = patched_functions["aiohttp.ClientSession._request"]
    except ImportError:
        pass

    logger.info("Restored original network functions")

"""
Shared test fixtures for the AI Resume Enhancer project.

Fixtures defined here are available to all test modules in the tests/ directory.

This file provides a reusable TestClient instance for simulating API requests without running a live server.
"""

import pytest                                           # Test runner 
from fastapi.testclient import TestClient               # Simulates API calls to FastAPI
from app.main import app                                # FastAPI app instance

# Defines a pytest fixture that creates a reusable FastAPI test client 
@pytest.fixture(scope="module")
def client():
    return TestClient(app)
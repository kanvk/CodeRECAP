# tests/test.py

import pytest
from project.main import hello_world  # Adjust the import path as necessary


def test_hello_world():
    """Test the hello_world function."""
    result = hello_world("World")
    assert (
        result == "Hello, World!"
    )  # Check if the function returns the expected output

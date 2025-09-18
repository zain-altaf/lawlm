#!/usr/bin/env python3
"""
Unit tests for API utilities and circuit breaker

Tests the core API functionality including:
- Circuit breaker implementation
- URL sanitization
- Exception sanitization
- Fetch with retry logic
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
import requests

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import (
    CircuitBreaker,
    sanitize_url_for_logging,
    sanitize_exception_message,
    fetch_with_retry
)


class TestCircuitBreaker(unittest.TestCase):
    """Test CircuitBreaker implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        self.assertEqual(self.cb.state, "CLOSED")
        self.assertEqual(self.cb.failure_count, 0)
        self.assertFalse(self.cb.is_open())

    def test_successful_calls(self):
        """Test successful function calls."""
        def success_func():
            return "success"

        result = self.cb.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.cb.failure_count, 0)
        self.assertEqual(self.cb.state, "CLOSED")

    def test_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        def failing_func():
            raise Exception("Test failure")

        # Call should fail 3 times before opening
        for i in range(3):
            with self.assertRaises(Exception):
                self.cb.call(failing_func)

            if i < 2:
                self.assertEqual(self.cb.state, "CLOSED")
            else:
                self.assertEqual(self.cb.state, "OPEN")

    def test_open_circuit_blocks_calls(self):
        """Test that open circuit blocks calls."""
        # Force circuit to open
        self.cb.state = "OPEN"
        self.cb.last_failure_time = time.time()

        def any_func():
            return "should not execute"

        with self.assertRaises(Exception) as cm:
            self.cb.call(any_func)

        self.assertIn("Circuit breaker is OPEN", str(cm.exception))

    def test_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        # Force circuit to open and set old failure time
        self.cb.state = "OPEN"
        self.cb.last_failure_time = time.time() - 2  # 2 seconds ago

        def success_func():
            return "recovered"

        # Should transition to HALF_OPEN then CLOSED on success
        result = self.cb.call(success_func)
        self.assertEqual(result, "recovered")
        self.assertEqual(self.cb.state, "CLOSED")
        self.assertEqual(self.cb.failure_count, 0)

    def test_half_open_failure(self):
        """Test failure during half-open state."""
        # Force circuit to open and set old failure time
        self.cb.state = "OPEN"
        self.cb.last_failure_time = time.time() - 2

        def failing_func():
            raise Exception("Still failing")

        # Should transition to HALF_OPEN then back to OPEN on failure
        with self.assertRaises(Exception):
            self.cb.call(failing_func)

        self.assertEqual(self.cb.state, "OPEN")


class TestSanitizationFunctions(unittest.TestCase):
    """Test URL and exception sanitization functions."""

    def test_sanitize_url_basic(self):
        """Test basic URL sanitization."""
        test_cases = [
            (
                "https://api.example.com/data?api_key=secret123&other=value",
                "https://api.example.com/data?api_key=%2A%2A%2AREDACTED%2A%2A%2A&other=value"
            ),
            (
                "https://api.example.com/data?token=abc123&limit=10",
                "https://api.example.com/data?token=%2A%2A%2AREDACTED%2A%2A%2A&limit=10"
            ),
            (
                "https://api.example.com/data?normal=param",
                "https://api.example.com/data?normal=param"
            ),
        ]

        for input_url, expected in test_cases:
            with self.subTest(url=input_url):
                result = sanitize_url_for_logging(input_url)
                # Check that sensitive parameters are redacted
                if "api_key=secret123" in input_url:
                    self.assertNotIn("secret123", result)
                    self.assertIn("REDACTED", result)
                elif "token=abc123" in input_url:
                    self.assertNotIn("abc123", result)
                    self.assertIn("REDACTED", result)
                else:
                    self.assertEqual(result, expected)

    def test_sanitize_url_no_query(self):
        """Test URL sanitization with no query parameters."""
        url = "https://api.example.com/data"
        result = sanitize_url_for_logging(url)
        self.assertEqual(result, url)

    def test_sanitize_exception_message(self):
        """Test exception message sanitization."""
        test_cases = [
            ("Error with api_key=secret123", "Error with api_key=***REDACTED***"),
            ("Authorization token abc123def", "Authorization ***REDACTED***"),
            ("Simple error message", "Simple error message"),
            ("Multiple api_key=key1 and token=key2", "Multiple api_key=***REDACTED*** and token=***REDACTED***"),
        ]

        for input_msg, expected in test_cases:
            with self.subTest(message=input_msg):
                result = sanitize_exception_message(input_msg)
                # Check that sensitive data is redacted
                if "secret123" in input_msg:
                    self.assertNotIn("secret123", result)
                    self.assertIn("REDACTED", result)
                elif "abc123def" in input_msg:
                    self.assertNotIn("abc123def", result)
                    self.assertIn("REDACTED", result)


class TestFetchWithRetry(unittest.TestCase):
    """Test fetch_with_retry function."""

    @patch('main.requests.get')
    def test_successful_request(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        mock_get.return_value = mock_response

        result = fetch_with_retry("https://api.example.com/data")

        self.assertEqual(result, {"data": "success"})
        mock_get.assert_called_once()

    @patch('main.requests.get')
    def test_rate_limit_retry(self, mock_get):
        """Test rate limit (429) retry logic."""
        # First call returns 429, second call succeeds
        responses = [
            Mock(status_code=429, headers={'Retry-After': '1'}),
            Mock(status_code=200, json=lambda: {"data": "success"})
        ]
        mock_get.side_effect = responses

        with patch('main.time.sleep') as mock_sleep:
            result = fetch_with_retry("https://api.example.com/data", max_retries=2)

        self.assertEqual(result, {"data": "success"})
        mock_sleep.assert_called_once_with(1)  # Retry-After value
        self.assertEqual(mock_get.call_count, 2)

    @patch('main.requests.get')
    def test_server_error_retry(self, mock_get):
        """Test server error (5xx) retry logic."""
        # First call returns 500, second call succeeds
        responses = [
            Mock(status_code=500),
            Mock(status_code=200, json=lambda: {"data": "success"})
        ]
        mock_get.side_effect = responses

        with patch('main.time.sleep') as mock_sleep:
            result = fetch_with_retry("https://api.example.com/data", max_retries=2, delay=1)

        self.assertEqual(result, {"data": "success"})
        mock_sleep.assert_called_once_with(1)  # delay value
        self.assertEqual(mock_get.call_count, 2)

    @patch('main.requests.get')
    def test_client_error_no_retry(self, mock_get):
        """Test client error (4xx) doesn't retry."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetch_with_retry("https://api.example.com/data", max_retries=3)

        self.assertIsNone(result)
        mock_get.assert_called_once()  # No retries for 4xx

    @patch('main.requests.get')
    def test_max_retries_exceeded(self, mock_get):
        """Test max retries exceeded."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with patch('main.time.sleep'):
            result = fetch_with_retry("https://api.example.com/data", max_retries=2, delay=0.1)

        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 3)  # Initial + 2 retries

    @patch('main.requests.get')
    def test_request_exception_retry(self, mock_get):
        """Test request exception retry logic."""
        # First call raises exception, second call succeeds
        mock_get.side_effect = [
            requests.exceptions.RequestException("Connection error"),
            Mock(status_code=200, json=lambda: {"data": "success"})
        ]

        with patch('main.time.sleep') as mock_sleep:
            result = fetch_with_retry("https://api.example.com/data", max_retries=2, delay=1)

        self.assertEqual(result, {"data": "success"})
        mock_sleep.assert_called_once_with(1)
        self.assertEqual(mock_get.call_count, 2)

    @patch('main.requests.get')
    def test_circuit_breaker_integration(self, mock_get):
        """Test fetch_with_retry with circuit breaker."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "success"}
        mock_get.return_value = mock_response

        circuit_breaker = CircuitBreaker()
        result = fetch_with_retry(
            "https://api.example.com/data",
            circuit_breaker=circuit_breaker
        )

        self.assertEqual(result, {"data": "success"})
        self.assertEqual(circuit_breaker.failure_count, 0)

    @patch('main.requests.get')
    def test_circuit_breaker_failure(self, mock_get):
        """Test circuit breaker with API failures."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        circuit_breaker = CircuitBreaker(failure_threshold=2)

        # Should fail and increment failure count
        result = fetch_with_retry(
            "https://api.example.com/data",
            max_retries=0,
            circuit_breaker=circuit_breaker
        )

        self.assertIsNone(result)
        self.assertEqual(circuit_breaker.failure_count, 1)


if __name__ == '__main__':
    unittest.main()
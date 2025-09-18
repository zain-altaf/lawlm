#!/usr/bin/env python3
"""
Unit tests for Redis Rate Limit Hook

Tests the core Redis functionality including:
- Input validation
- Atomic operations
- Rate limiting logic
- Error handling
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'airflow'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from airflow.hooks.redis_rate_limit_hook import (
    RedisRateLimitHook,
    validate_redis_key,
    validate_redis_value,
    sanitize_log_message
)


class TestRedisValidation(unittest.TestCase):
    """Test input validation functions."""

    def test_validate_redis_key_valid(self):
        """Test valid Redis keys."""
        valid_keys = [
            "simple_key",
            "key:with:colons",
            "key-with-dashes",
            "key_with_underscores",
            "key123",
        ]
        for key in valid_keys:
            with self.subTest(key=key):
                self.assertTrue(validate_redis_key(key))

    def test_validate_redis_key_invalid(self):
        """Test invalid Redis keys."""
        invalid_keys = [
            "key with spaces",
            "key\nwith\nnewlines",
            "key\twith\ttabs",
            "key\rwith\rreturns",
            "key\"with\"quotes",
            "key'with'quotes",
            "key\\with\\backslashes",
            "key\x00with\x00nulls",
            123,  # Not a string
            None,
            "",
            "x" * 600,  # Too long
        ]
        for key in invalid_keys:
            with self.subTest(key=key):
                self.assertFalse(validate_redis_key(key))

    def test_validate_redis_value_valid(self):
        """Test valid Redis values."""
        valid_values = [
            "simple_string",
            123,
            45.67,
            {"key": "value"},
            ["item1", "item2"],
            None,
            b"bytes_value",
        ]
        for value in valid_values:
            with self.subTest(value=value):
                self.assertTrue(validate_redis_value(value))

    def test_validate_redis_value_invalid(self):
        """Test invalid Redis values."""
        invalid_values = [
            "x" * (1024 * 1024 + 1),  # Too large string
            "EVAL some script",
            "SCRIPT LOAD",
            "CONFIG SET",
            "FLUSHALL",
            "FLUSHDB",
            10**16,  # Too large number
            -10**16,  # Too large negative number
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertFalse(validate_redis_value(value))

    def test_sanitize_log_message(self):
        """Test log message sanitization."""
        test_cases = [
            ("api_key=abc123def456", "api_key=***REDACTED***"),
            ("token: xyz789", "token: ***REDACTED***"),
            ('{"api_key": "secret123"}', '{"api_key": "***REDACTED***"}'),
            ("Authorization: Token abc123", "Authorization: ***REDACTED***"),
            ("password=mypass", "password=***REDACTED***"),
            ("normal log message", "normal log message"),
        ]

        for input_msg, expected in test_cases:
            with self.subTest(input_msg=input_msg):
                result = sanitize_log_message(input_msg)
                self.assertEqual(result, expected)


class TestRedisRateLimitHook(unittest.TestCase):
    """Test RedisRateLimitHook functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.hook = RedisRateLimitHook()
        self.mock_client = Mock()

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_atomic_increment_validation(self, mock_redis):
        """Test input validation in atomic increment."""
        mock_redis.return_value = self.mock_client

        # Test invalid inputs
        with self.assertRaises(ValueError):
            self.hook.atomic_increment_api_calls(-1)  # Negative calls

        with self.assertRaises(ValueError):
            self.hook.atomic_increment_api_calls("invalid")  # String instead of int

        with self.assertRaises(ValueError):
            self.hook.atomic_increment_api_calls(5, limit=-10)  # Negative limit

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_atomic_increment_success(self, mock_redis):
        """Test successful atomic increment."""
        mock_redis.return_value = self.mock_client
        self.mock_client.script_load.return_value = "sha123"
        self.mock_client.evalsha.return_value = [10, 1]  # new_count, allowed

        # Mock the get_current_hour_key method
        with patch.object(self.hook, 'get_current_hour_key', return_value='test:key'):
            result = self.hook.atomic_increment_api_calls(5)

            self.assertIsInstance(result, dict)
            self.assertEqual(result['new_count'], 10)
            self.assertTrue(result['allowed'])

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_rate_limit_exceeded(self, mock_redis):
        """Test rate limit exceeded scenario."""
        mock_redis.return_value = self.mock_client
        self.mock_client.script_load.return_value = "sha123"
        self.mock_client.evalsha.return_value = [5001, 0]  # Exceeded limit

        with patch.object(self.hook, 'get_current_hour_key', return_value='test:key'):
            result = self.hook.atomic_increment_api_calls(1)

            self.assertEqual(result['new_count'], 5001)
            self.assertFalse(result['allowed'])

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_hour_boundary_logic(self, mock_redis):
        """Test hour boundary detection and reset."""
        mock_redis.return_value = self.mock_client

        with patch.object(self.hook, 'get_current_hour_key') as mock_hour_key:
            mock_hour_key.return_value = 'test:2024010112'  # Mock hour key

            # Mock Redis operations
            self.mock_client.get.return_value = b'100'  # Current count
            self.mock_client.exists.return_value = True

            result = self.hook.check_hour_boundary_and_reset()

            self.assertIsInstance(result, dict)
            self.assertIn('current_hour', result)
            self.assertIn('reset_occurred', result)

    def test_key_generation(self):
        """Test Redis key generation methods."""
        # Test hour key generation
        with patch('airflow.hooks.redis_rate_limit_hook.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.strftime.return_value = '2024010112'

            hour_key = self.hook.get_current_hour_key()
            self.assertIn('2024010112', hour_key)
            self.assertTrue(hour_key.startswith('courtlistener:rate_limit:'))


class TestRedisHookErrorHandling(unittest.TestCase):
    """Test error handling in Redis hook."""

    def setUp(self):
        """Set up test fixtures."""
        self.hook = RedisRateLimitHook()

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_connection_failure(self, mock_redis):
        """Test Redis connection failure handling."""
        mock_redis.side_effect = Exception("Connection failed")

        with self.assertRaises(Exception):
            self.hook.get_conn()

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_script_load_failure(self, mock_redis):
        """Test script loading failure."""
        mock_client = Mock()
        mock_client.script_load.side_effect = Exception("Script load failed")
        mock_redis.return_value = mock_client

        with self.assertRaises(Exception):
            self.hook._load_scripts()

    @patch('airflow.hooks.redis_rate_limit_hook.redis.Redis')
    def test_graceful_degradation(self, mock_redis):
        """Test graceful degradation when Redis is unavailable."""
        mock_redis.side_effect = Exception("Redis unavailable")

        # Test that the hook handles Redis unavailability gracefully
        # This would depend on implementation of fallback mechanisms
        with self.assertRaises(Exception):
            self.hook.get_conn()


if __name__ == '__main__':
    unittest.main()
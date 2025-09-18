#!/usr/bin/env python3
"""
Basic functionality tests for core components

Tests essential functionality without external dependencies:
- Configuration loading
- Utility functions
- Error handling patterns
"""

import unittest
import tempfile
import json
import os
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import only what we can test without external dependencies
from main import (
    CircuitBreaker,
    sanitize_url_for_logging,
    sanitize_exception_message
)

from config import (
    DataIngestionConfig,
    TextSplitterConfig,
    VectorProcessingConfig,
    QdrantConfig
)


class TestBasicConfiguration(unittest.TestCase):
    """Test basic configuration functionality."""

    def test_data_ingestion_config_creation(self):
        """Test DataIngestionConfig can be created and has expected attributes."""
        config = DataIngestionConfig()

        # Check essential attributes exist
        self.assertTrue(hasattr(config, 'court'))
        self.assertTrue(hasattr(config, 'num_dockets'))
        self.assertTrue(hasattr(config, 'api_base_url'))

        # Check default values
        self.assertEqual(config.court, "scotus")
        self.assertEqual(config.num_dockets, 5)
        self.assertIsInstance(config.timeout_seconds, int)

    def test_text_splitter_config_creation(self):
        """Test TextSplitterConfig can be created and has expected attributes."""
        config = TextSplitterConfig()

        # Check essential attributes exist
        self.assertTrue(hasattr(config, 'chunk_size_chars'))
        self.assertTrue(hasattr(config, 'overlap_chars'))

        # Check types and reasonable values
        self.assertIsInstance(config.chunk_size_chars, int)
        self.assertIsInstance(config.overlap_chars, int)
        self.assertGreater(config.chunk_size_chars, 0)
        self.assertGreater(config.overlap_chars, 0)

    def test_vector_processing_config_creation(self):
        """Test VectorProcessingConfig can be created and has expected attributes."""
        config = VectorProcessingConfig()

        # Check essential attributes exist
        self.assertTrue(hasattr(config, 'embedding_model'))
        self.assertTrue(hasattr(config, 'batch_size'))
        self.assertTrue(hasattr(config, 'device'))

        # Check default values
        self.assertIsInstance(config.embedding_model, str)
        self.assertIsInstance(config.batch_size, int)
        self.assertGreater(config.batch_size, 0)

    def test_qdrant_config_creation(self):
        """Test QdrantConfig can be created and has expected attributes."""
        config = QdrantConfig()

        # Check essential attributes exist
        self.assertTrue(hasattr(config, 'url'))
        self.assertTrue(hasattr(config, 'timeout'))
        self.assertTrue(hasattr(config, 'use_cloud'))

        # Check default values
        self.assertIsInstance(config.url, str)
        self.assertIsInstance(config.timeout, int)
        self.assertIsInstance(config.use_cloud, bool)


class TestCircuitBreakerBasics(unittest.TestCase):
    """Test basic circuit breaker functionality."""

    def test_circuit_breaker_creation(self):
        """Test CircuitBreaker can be created with defaults."""
        cb = CircuitBreaker()

        self.assertEqual(cb.state, "CLOSED")
        self.assertEqual(cb.failure_count, 0)
        self.assertIsNone(cb.last_failure_time)

    def test_circuit_breaker_custom_params(self):
        """Test CircuitBreaker with custom parameters."""
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout=120)

        self.assertEqual(cb.failure_threshold, 10)
        self.assertEqual(cb.recovery_timeout, 120)
        self.assertEqual(cb.state, "CLOSED")

    def test_circuit_breaker_successful_call(self):
        """Test circuit breaker with successful function call."""
        cb = CircuitBreaker()

        def success_function():
            return "success"

        result = cb.call(success_function)

        self.assertEqual(result, "success")
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.state, "CLOSED")

    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker tracks failures."""
        cb = CircuitBreaker(failure_threshold=3)

        def failing_function():
            raise Exception("Test failure")

        # First failure
        with self.assertRaises(Exception):
            cb.call(failing_function)

        self.assertEqual(cb.failure_count, 1)
        self.assertEqual(cb.state, "CLOSED")

        # Second failure
        with self.assertRaises(Exception):
            cb.call(failing_function)

        self.assertEqual(cb.failure_count, 2)
        self.assertEqual(cb.state, "CLOSED")

        # Third failure - should open circuit
        with self.assertRaises(Exception):
            cb.call(failing_function)

        self.assertEqual(cb.failure_count, 3)
        self.assertEqual(cb.state, "OPEN")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for security and logging."""

    def test_sanitize_url_basic(self):
        """Test URL sanitization with no sensitive data."""
        clean_url = "https://api.example.com/data?limit=10&page=1"
        result = sanitize_url_for_logging(clean_url)
        self.assertEqual(result, clean_url)

    def test_sanitize_url_with_api_key(self):
        """Test URL sanitization removes API keys."""
        sensitive_url = "https://api.example.com/data?api_key=secret123&limit=10"
        result = sanitize_url_for_logging(sensitive_url)

        # Should not contain the actual key
        self.assertNotIn("secret123", result)
        # Should contain redacted marker
        self.assertIn("REDACTED", result)
        # Should still contain non-sensitive parameters
        self.assertIn("limit=10", result)

    def test_sanitize_url_with_token(self):
        """Test URL sanitization removes tokens."""
        sensitive_url = "https://api.example.com/data?token=abc123def456&sort=name"
        result = sanitize_url_for_logging(sensitive_url)

        # Should not contain the actual token
        self.assertNotIn("abc123def456", result)
        # Should contain redacted marker
        self.assertIn("REDACTED", result)

    def test_sanitize_exception_message_clean(self):
        """Test exception sanitization with clean message."""
        clean_message = "Connection timeout after 30 seconds"
        result = sanitize_exception_message(clean_message)
        self.assertEqual(result, clean_message)

    def test_sanitize_exception_message_with_api_key(self):
        """Test exception sanitization removes API keys."""
        sensitive_message = "Request failed: api_key=secret123 invalid"
        result = sanitize_exception_message(sensitive_message)

        # Should not contain the actual key
        self.assertNotIn("secret123", result)
        # Should contain redacted marker
        self.assertIn("REDACTED", result)

    def test_sanitize_exception_message_with_token(self):
        """Test exception sanitization removes tokens."""
        sensitive_message = "Authorization failed: token abc123"
        result = sanitize_exception_message(sensitive_message)

        # Should not contain the actual token
        self.assertNotIn("abc123", result)
        # Should contain redacted marker
        self.assertIn("REDACTED", result)


class TestErrorHandlingPatterns(unittest.TestCase):
    """Test error handling patterns used throughout the codebase."""

    def test_circuit_breaker_open_state_blocks_calls(self):
        """Test that open circuit breaker blocks calls."""
        import time
        cb = CircuitBreaker(recovery_timeout=60)  # Long timeout to prevent recovery
        cb.state = "OPEN"  # Force open state
        cb.last_failure_time = time.time()  # Recent failure time

        def any_function():
            return "should not execute"

        with self.assertRaises(Exception) as context:
            cb.call(any_function)

        self.assertIn("Circuit breaker is OPEN", str(context.exception))

    def test_configuration_handles_missing_values(self):
        """Test configuration handles missing or None values gracefully."""
        # Test that configs can be created even with missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = DataIngestionConfig()
            # Should not raise an exception
            self.assertIsInstance(config, DataIngestionConfig)

    def test_utility_functions_handle_edge_cases(self):
        """Test utility functions handle edge cases."""
        # Test with None input
        result = sanitize_exception_message(None)
        self.assertEqual(result, "None")

        # Test with empty string
        result = sanitize_exception_message("")
        self.assertEqual(result, "")

        # Test with non-string input
        result = sanitize_exception_message(123)
        self.assertEqual(result, "123")


class TestBasicImports(unittest.TestCase):
    """Test that essential modules can be imported."""

    def test_config_module_imports(self):
        """Test that config module imports successfully."""
        import config
        self.assertTrue(hasattr(config, 'DataIngestionConfig'))
        self.assertTrue(hasattr(config, 'QdrantConfig'))

    def test_main_module_imports(self):
        """Test that main module imports successfully."""
        import main
        self.assertTrue(hasattr(main, 'CircuitBreaker'))
        self.assertTrue(hasattr(main, 'sanitize_url_for_logging'))

    def test_configuration_creation_without_files(self):
        """Test that configurations can be created without external files."""
        from config import PipelineConfig

        # Should be able to create config without external dependencies
        config = PipelineConfig()
        self.assertIsInstance(config, PipelineConfig)


if __name__ == '__main__':
    unittest.main()
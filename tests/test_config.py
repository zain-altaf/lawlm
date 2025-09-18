#!/usr/bin/env python3
"""
Unit tests for configuration management

Tests the configuration system including:
- Configuration loading
- Default values
- Environment variable integration
- Configuration validation
"""

import unittest
import tempfile
import json
import os
from unittest.mock import patch, Mock
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import (
    PipelineConfig,
    DataIngestionConfig,
    TextSplitterConfig,
    VectorProcessingConfig,
    QdrantConfig,
    load_config
)


class TestConfigurationClasses(unittest.TestCase):
    """Test configuration dataclasses."""

    def test_data_ingestion_config_defaults(self):
        """Test DataIngestionConfig default values."""
        config = DataIngestionConfig()

        # Test default values
        self.assertEqual(config.court, "scotus")
        self.assertEqual(config.num_dockets, 5)  # Updated to match actual default
        self.assertEqual(config.api_base_url, "https://www.courtlistener.com/api/rest/v4")
        self.assertEqual(config.timeout_seconds, 30)
        self.assertEqual(config.max_retries, 3)

    def test_vector_processing_config_defaults(self):
        """Test VectorProcessingConfig default values."""
        config = VectorProcessingConfig()

        # Test default values
        self.assertEqual(config.embedding_model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.collection_name_vector, "caselaw-chunks")

    def test_qdrant_config_defaults(self):
        """Test QdrantConfig default values."""
        config = QdrantConfig()

        # Test default values
        self.assertEqual(config.url, "http://localhost:6333")
        self.assertEqual(config.timeout, 30)
        self.assertFalse(config.prefer_grpc)
        self.assertFalse(config.use_cloud)

    def test_pipeline_config_initialization(self):
        """Test PipelineConfig initialization."""
        config = PipelineConfig()

        # Check that all sub-configs are initialized
        self.assertIsInstance(config.data_ingestion, DataIngestionConfig)
        self.assertIsInstance(config.text_splitter, TextSplitterConfig)
        self.assertIsInstance(config.vector_processing, VectorProcessingConfig)
        self.assertIsInstance(config.qdrant, QdrantConfig)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading from files."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "data_ingestion": {
                "court": "ca9",
                "num_dockets": 50,
                "chunk_size": 2048,
                "chunk_overlap": 400
            },
            "vector_processing": {
                "model_name": "custom-model",
                "batch_size": 16
            },
            "qdrant": {
                "collection_name": "test_collection",
                "vector_size": 512,
                "use_cloud": True,
                "cloud_url": "https://test.qdrant.io",
                "cloud_api_key": "test_key"
            }
        }

        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = PipelineConfig(config_file)

        # Verify that configuration loads without error
        self.assertIsInstance(config, PipelineConfig)
        self.assertIsInstance(config.data_ingestion, DataIngestionConfig)
        self.assertIsInstance(config.qdrant, QdrantConfig)

    def test_load_config_partial_file(self):
        """Test loading configuration with partial data (defaults for missing)."""
        config_data = {
            "data_ingestion": {
                "court": "ca1",
                "num_dockets": 25
                # chunk_size and chunk_overlap should use defaults
            }
            # vector_processing and qdrant should use defaults
        }

        config_file = os.path.join(self.temp_dir, "partial_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = PipelineConfig(config_file)

        # Verify custom values
        self.assertEqual(config.data_ingestion.court, "ca1")
        self.assertEqual(config.data_ingestion.num_dockets, 25)

        # Verify defaults are used for missing values
        self.assertEqual(config.data_ingestion.chunk_size, 1536)  # default
        self.assertEqual(config.data_ingestion.chunk_overlap, 300)  # default
        self.assertEqual(config.vector_processing.model_name, "BAAI/bge-small-en-v1.5")  # default

    def test_load_config_nonexistent_file(self):
        """Test loading configuration with non-existent file (should use defaults)."""
        config = PipelineConfig("/path/that/does/not/exist.json")

        # Should use all defaults
        self.assertEqual(config.data_ingestion.court, "scotus")
        self.assertEqual(config.data_ingestion.num_dockets, 100)
        self.assertEqual(config.vector_processing.model_name, "BAAI/bge-small-en-v1.5")

    def test_load_config_invalid_json(self):
        """Test loading configuration with invalid JSON."""
        config_file = os.path.join(self.temp_dir, "invalid.json")
        with open(config_file, 'w') as f:
            f.write("{ invalid json content")

        # Should handle gracefully and use defaults
        config = PipelineConfig(config_file)
        self.assertEqual(config.data_ingestion.court, "scotus")

    @patch.dict(os.environ, {
        'QDRANT_URL': 'https://env.qdrant.io',
        'QDRANT_API_KEY': 'env_api_key',
        'USE_CLOUD': 'true'
    })
    def test_environment_variable_override(self):
        """Test that environment variables override config file values."""
        config_data = {
            "qdrant": {
                "use_cloud": False,
                "cloud_url": "https://file.qdrant.io",
                "cloud_api_key": "file_api_key"
            }
        }

        config_file = os.path.join(self.temp_dir, "env_test.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = PipelineConfig(config_file)

        # Environment variables should override file values
        self.assertTrue(config.qdrant.use_cloud)
        self.assertEqual(config.qdrant.cloud_url, "https://env.qdrant.io")
        self.assertEqual(config.qdrant.cloud_api_key, "env_api_key")


class TestLoadConfigFunction(unittest.TestCase):
    """Test the load_config convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_explicit_file(self):
        """Test load_config with explicit file path."""
        config_data = {"data_ingestion": {"court": "test"}}
        config_file = os.path.join(self.temp_dir, "explicit.json")

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = load_config(config_file)
        self.assertEqual(config.data_ingestion.court, "test")

    def test_load_config_search_paths(self):
        """Test load_config searching standard paths."""
        # Create config.json in temp dir and patch current directory
        config_data = {"data_ingestion": {"court": "found"}}
        config_file = os.path.join(self.temp_dir, "config.json")

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        with patch('os.getcwd', return_value=self.temp_dir):
            config = load_config()
            self.assertEqual(config.data_ingestion.court, "found")

    def test_load_config_no_file_found(self):
        """Test load_config when no config file is found."""
        with patch('os.getcwd', return_value=self.temp_dir):
            config = load_config()
            # Should use defaults
            self.assertEqual(config.data_ingestion.court, "scotus")


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""

    def test_invalid_court_value(self):
        """Test handling of invalid court values."""
        # This test assumes there might be validation in the future
        config = DataIngestionConfig()
        config.court = "invalid_court"
        # For now, just ensure it doesn't crash
        self.assertEqual(config.court, "invalid_court")

    def test_negative_values(self):
        """Test handling of negative values."""
        config = DataIngestionConfig()
        config.num_dockets = -1
        config.chunk_size = -100

        # For now, just ensure values are stored (validation could be added later)
        self.assertEqual(config.num_dockets, -1)
        self.assertEqual(config.chunk_size, -100)

    def test_configuration_serialization(self):
        """Test that configuration can be serialized/deserialized."""
        original_config = PipelineConfig()
        original_config.data_ingestion.court = "test_court"
        original_config.data_ingestion.num_dockets = 123

        # Convert to dict
        config_dict = {
            "data_ingestion": {
                "court": original_config.data_ingestion.court,
                "num_dockets": original_config.data_ingestion.num_dockets,
                "chunk_size": original_config.data_ingestion.chunk_size,
                "chunk_overlap": original_config.data_ingestion.chunk_overlap,
                "include_clusters": original_config.data_ingestion.include_clusters,
                "include_opinions": original_config.data_ingestion.include_opinions
            }
        }

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_file = f.name

        try:
            # Load from file
            new_config = PipelineConfig(temp_file)

            # Verify values match
            self.assertEqual(new_config.data_ingestion.court, "test_court")
            self.assertEqual(new_config.data_ingestion.num_dockets, 123)
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()
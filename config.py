"""
Configuration Management for Legal Document Processing Pipeline

Centralized configuration system with environment variable support,
validation, and monitoring capabilities.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from CourtListener API."""
    api_key: str = ""
    api_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    min_text_length: int = 100
    min_word_count: int = 50
    court: str = "scotus"           
    num_dockets: int = 5            
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("CASELAW_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ CASELAW_API_KEY not set - ingestion will fail")


@dataclass
class TextSplitterConfig:
    """Configuration for RecursiveCharacterTextSplitter."""
    chunk_size_chars: int = 1536        # Character count for chunks (was 384 tokens * 4)
    overlap_chars: int = 300            # Character overlap between chunks (was 75 tokens * 4)
    min_chunk_size_chars: int = 400     # Minimum chunk size in characters (was 100 tokens * 4)
    quality_threshold: float = 0.3      # Quality threshold for chunks
    separators: Optional[List[str]] = None  # Will use legal-optimized defaults if None



@dataclass
class VectorProcessingConfig:
    """Configuration for hybrid vector processing (dense + sparse vectors)."""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 50
    memory_cleanup_frequency: int = 100
    device: str = "auto"
    collection_name_vector: str = "caselaw-chunks"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cpu"  # Force CPU-only processing


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
    use_cloud: bool = False  # Flag to indicate cloud usage
    cluster_name: Optional[str] = None  # Cloud cluster name for reference
    free_tier_limit_mb: float = 1024.0  # 1GB free tier limit
    
    def __post_init__(self):
        # Check environment variables
        env_url = os.getenv("QDRANT_URL")
        if env_url:
            self.url = env_url
            # Detect if this is a cloud URL
            if "cloud.qdrant.io" in env_url or "qdrant.tech" in env_url:
                self.use_cloud = True
        
        env_api_key = os.getenv("QDRANT_API_KEY")
        if env_api_key:
            self.api_key = env_api_key
            
        env_cluster_name = os.getenv("QDRANT_CLUSTER_NAME")
        if env_cluster_name:
            self.cluster_name = env_cluster_name
            
        # If API key is provided, assume cloud usage
        if self.api_key and not self.use_cloud:
            if "localhost" not in self.url and "127.0.0.1" not in self.url:
                self.use_cloud = True


@dataclass
class ProcessingConfig:
    """Configuration for general processing settings."""
    working_directory: str = "data"
    log_level: str = "INFO"
    progress_reporting: bool = True
    save_intermediate_files: bool = True
    cleanup_temp_files: bool = False
    max_workers: int = 1  # For future parallel processing
    
    def __post_init__(self):
        # Ensure working directory exists
        Path(self.working_directory).mkdir(parents=True, exist_ok=True)
        
        # Set up logging level
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enable_memory_monitoring: bool = True
    memory_check_frequency: int = 50  # Every N operations
    log_performance_metrics: bool = True
    save_processing_summary: bool = True
    alert_memory_threshold_mb: float = 8192  # 8GB


class PipelineConfig:
    """
    Main configuration class that combines all component configurations.
    Supports loading from files, environment variables, and runtime updates.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        # Initialize all component configurations
        self.data_ingestion = DataIngestionConfig()
        self.text_splitter = TextSplitterConfig()
        self.vector_processing = VectorProcessingConfig()
        self.qdrant = QdrantConfig()
        self.processing = ProcessingConfig()
        self.monitoring = MonitoringConfig()

        # Load from file if provided (this is a JSON that you need to provide)
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                        else:
                            logger.warning(f"Unknown config key: {section}.{key}")
                else:
                    logger.warning(f"Unknown config section: {section}")
            
            logger.info(f"📋 Loaded configuration from: {config_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration from {config_file}: {e}")
            raise
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'data_ingestion': asdict(self.data_ingestion),
                'text_splitter': asdict(self.text_splitter),
                'vector_processing': asdict(self.vector_processing),
                'qdrant': asdict(self.qdrant),
                'processing': asdict(self.processing),
                'monitoring': asdict(self.monitoring)
            }
            
            # Don't save sensitive information
            if 'api_key' in config_data['data_ingestion']:
                config_data['data_ingestion']['api_key'] = "***REDACTED***"
            if config_data['qdrant']['api_key']:
                config_data['qdrant']['api_key'] = "***REDACTED***"
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Saved configuration to: {config_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration to {config_file}: {e}")
            raise
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages
        """
        issues = []
        
        # Check required API key
        if not self.data_ingestion.api_key:
            issues.append("❌ CASELAW_API_KEY is required for data ingestion")
        
        # Validate chunk sizes
        if self.text_splitter.min_chunk_size_chars >= self.text_splitter.chunk_size_chars:
            issues.append("❌ min_chunk_size_chars must be less than chunk_size_chars")
        
        # Validate quality threshold
        if not 0 < self.text_splitter.quality_threshold < 1:
            issues.append("❌ quality_threshold must be between 0 and 1")
        
        # Check working directory
        if not Path(self.processing.working_directory).exists():
            issues.append(f"⚠️ Working directory does not exist: {self.processing.working_directory}")
        
        # Test Qdrant connection (non-blocking)
        try:
            from qdrant_client import QdrantClient
            
            # Create client with or without API key
            if self.qdrant.api_key:
                client = QdrantClient(
                    url=self.qdrant.url, 
                    api_key=self.qdrant.api_key,
                    timeout=5
                )
            else:
                client = QdrantClient(self.qdrant.url, timeout=5)
                
            client.get_collections()
            connection_type = "cloud" if self.qdrant.use_cloud else "local"
            logger.info(f"✅ Qdrant {connection_type} connection successful: {self.qdrant.url}")
            
        except Exception as e:
            issues.append(f"⚠️ Cannot connect to Qdrant at {self.qdrant.url}: {e}")
            
        # Check cloud-specific requirements
        if self.qdrant.use_cloud:
            if not self.qdrant.api_key:
                issues.append("❌ QDRANT_API_KEY is required for cloud usage")
            if "localhost" in self.qdrant.url or "127.0.0.1" in self.qdrant.url:
                issues.append("⚠️ Cloud flag enabled but URL appears to be local")
        
        # Log validation results
        if issues:
            logger.warning(f"Configuration validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("✅ Configuration validation passed")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'data_ingestion': {
                'api_configured': bool(self.data_ingestion.api_key),
                'timeout': self.data_ingestion.timeout_seconds,
                'min_text_length': self.data_ingestion.min_text_length
            },
            'text_chunking': {
                'splitter_type': 'RecursiveCharacterTextSplitter',
                'chunk_size_chars': self.text_splitter.chunk_size_chars,
                'overlap_chars': self.text_splitter.overlap_chars,
                'min_chunk_size_chars': self.text_splitter.min_chunk_size_chars,
                'quality_threshold': self.text_splitter.quality_threshold
            },
            'batch_processing': {
                'enabled': self.batch_processing.enable_batch_processing,
                'default_batch_size': self.batch_processing.default_batch_size,
                'max_batch_size': self.batch_processing.max_batch_size,
                'vector_batch_size': self.batch_processing.vector_processing_batch_size,
                'checkpoint_interval': self.batch_processing.checkpoint_interval
            },
            'hybrid_processing': {
                'embedding_model': self.vector_processing.embedding_model,
                'batch_size': self.vector_processing.batch_size,
                'device': self.vector_processing.device,
                'search_capabilities': 'semantic + keyword (RRF fusion)'
            },
            'qdrant': {
                'url': self.qdrant.url,
                'api_key_configured': bool(self.qdrant.api_key),
                'use_cloud': self.qdrant.use_cloud,
                'cluster_name': self.qdrant.cluster_name,
                'free_tier_limit_mb': self.qdrant.free_tier_limit_mb
            },
            'processing': {
                'working_directory': self.processing.working_directory,
                'log_level': self.processing.log_level,
                'save_intermediate': self.processing.save_intermediate_files
            },
            'monitoring': {
                'memory_monitoring': self.monitoring.enable_memory_monitoring,
                'performance_metrics': self.monitoring.log_performance_metrics
            }
        }

def create_default_config_file(config_path: str = "config.json") -> None:
    """Create a default configuration file."""
    config = PipelineConfig()
    config.save_to_file(config_path)
    print(f"📋 Created default configuration file: {config_path}")
    print("🔧 Edit this file to customize your pipeline settings")


def load_config(config_file: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration with fallbacks.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        PipelineConfig instance
    """
    # Try different config file locations
    config_paths = []
    
    if config_file:
        config_paths.append(config_file)
    
    # Standard locations
    config_paths.extend([
        "config.json",
        "pipeline_config.json",
        os.path.expanduser("~/.lawlm/config.json"),
        "/etc/lawlm/config.json"
    ])
    
    # Find first existing config file
    config_file_found = None
    for path in config_paths:
        if Path(path).exists():
            config_file_found = path
            break
    
    # Load configuration
    config = PipelineConfig(config_file_found)
    
    if config_file_found:
        logger.info(f"📋 Using configuration file: {config_file_found}")
    else:
        logger.info("📋 Using default configuration (no config file found)")
    
    return config


# Global configuration instance
_global_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: PipelineConfig) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config
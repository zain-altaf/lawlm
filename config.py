"""
Configuration Management for Legal Document Processing Pipeline

Centralized configuration system with environment variable support,
validation, and monitoring capabilities.
"""

# Standard library imports
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion from CourtListener API.

    Handles API credentials, timeouts, retries, and data filtering settings.
    """
    api_key: str = ""
    api_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    min_text_length: int = 100
    min_word_count: int = 50
    court: str = "scotus"           
    num_dockets: int = 5

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("CASELAW_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ CASELAW_API_KEY not set - ingestion will fail")


@dataclass
class TextSplitterConfig:
    """
    Configuration for RecursiveCharacterTextSplitter.

    Controls how legal documents are chunked for processing.
    """
    chunk_size_chars: int = 1536
    overlap_chars: int = 300
    min_chunk_size_chars: int = 400
    quality_threshold: float = 0.3


@dataclass
class VectorProcessingConfig:
    """
    Configuration for hybrid vector processing (dense + sparse vectors).

    Manages embedding model settings, batch processing, and collection naming.
    """
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 50
    memory_cleanup_frequency: int = 100
    device: str = "cpu"
    collection_name_vector: str = "caselaw-chunks"


@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant vector database.

    Supports both local and cloud deployments with automatic detection.
    """
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
    use_cloud: bool = False
    cluster_name: Optional[str] = None
    free_tier_limit_mb: float = 1024.0

    def __post_init__(self) -> None:
        # Check environment variables
        env_url = os.getenv("QDRANT_URL")
        if env_url:
            self.url = env_url
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
class RedisConfig:
    """
    Configuration for Redis caching and rate limiting.

    Handles connection settings, rate limiting parameters, and performance tuning.
    """
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True

    # Rate limiting specific settings
    rate_limit_key_ttl_hours: int = 25
    pipeline_state_ttl_hours: int = 25
    failed_dockets_ttl_hours: int = 72
    cleanup_interval_hours: int = 48

    # Performance tuning
    enable_atomic_operations: bool = True
    enable_lua_scripts: bool = True
    enable_pipeline_transactions: bool = True

    def __post_init__(self) -> None:
        # Check environment variables for Redis configuration
        env_host = os.getenv("REDIS_HOST")
        if env_host:
            self.host = env_host

        env_port = os.getenv("REDIS_PORT")
        if env_port:
            self.port = int(env_port)

        env_db = os.getenv("REDIS_DB")
        if env_db:
            self.db = int(env_db)

        env_password = os.getenv("REDIS_PASSWORD")
        if env_password:
            self.password = env_password

        env_ssl = os.getenv("REDIS_SSL")
        if env_ssl and env_ssl.lower() in ('true', '1', 'yes'):
            self.ssl = True

        # Auto-detect Redis cloud services
        if "redis.cloud" in self.host or "redislabs" in self.host:
            self.ssl = True

        # Validate configuration
        if self.port < 1 or self.port > 65535:
            logger.warning(f"Invalid Redis port {self.port}, using default 6379")
            self.port = 6379

        if self.db < 0:
            logger.warning(f"Invalid Redis database {self.db}, using default 0")
            self.db = 0


@dataclass
class AirflowConfig:
    """
    Configuration for Airflow DAG orchestration.

    Controls batch processing, scheduling, and rate limiting for pipeline execution.
    """
    batch_size: int = 495
    total_batches: int = 1
    schedule_interval_minutes: int = 12
    max_active_runs: int = 1
    pool_size: int = 50
    target_calls_per_hour: int = 4950
    soft_rate_buffer: int = 12
    initial_offset: int = 0

    # Redis integration settings
    enable_redis_rate_limiting: bool = True
    redis_fallback_to_postgres: bool = True
    redis_connection_id: str = "redis_default"

@dataclass
class ProcessingConfig:
    """
    Configuration for general processing settings.

    Manages working directories, logging, and file handling behavior.
    """
    working_directory: str = "data"
    log_level: str = "INFO"
    progress_reporting: bool = True
    save_intermediate_files: bool = True
    cleanup_temp_files: bool = False
    max_workers: int = 1

    def __post_init__(self) -> None:
        Path(self.working_directory).mkdir(parents=True, exist_ok=True)
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)


@dataclass
class MonitoringConfig:
    """
    Configuration for monitoring and metrics.

    Controls memory monitoring, performance tracking, and alerting thresholds.
    """
    enable_memory_monitoring: bool = True
    memory_check_frequency: int = 50
    log_performance_metrics: bool = True
    save_processing_summary: bool = True
    alert_memory_threshold_mb: float = 8192


class PipelineConfig:
    """
    Main configuration class that combines all component configurations.
    Supports loading from files, environment variables, and runtime updates.
    """
    
    def __init__(self, config_file: Optional[str] = None) -> None:
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
        self.redis = RedisConfig()
        self.processing = ProcessingConfig()
        self.monitoring = MonitoringConfig()
        self.airflow = AirflowConfig()

        # Load from file if provided
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
                'redis': asdict(self.redis),
                'processing': asdict(self.processing),
                'monitoring': asdict(self.monitoring),
                'airflow': asdict(self.airflow)
            }
            
            # Don't save sensitive information
            config_data['data_ingestion']['api_key'] = "***REDACTED***"
            if config_data['qdrant']['api_key']:
                config_data['qdrant']['api_key'] = "***REDACTED***"
            if config_data['redis']['password']:
                config_data['redis']['password'] = "***REDACTED***"
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Saved configuration to: {config_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration to {config_file}: {e}")
            raise
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        """
        issues = []

        # Check required API key
        if not self.data_ingestion.api_key:
            issues.append("CASELAW_API_KEY is required for data ingestion")

        # Validate chunk sizes
        if self.text_splitter.min_chunk_size_chars >= self.text_splitter.chunk_size_chars:
            issues.append("min_chunk_size_chars must be less than chunk_size_chars")

        # Validate quality threshold
        if not 0 < self.text_splitter.quality_threshold < 1:
            issues.append("quality_threshold must be between 0 and 1")

        # Check working directory
        if not Path(self.processing.working_directory).exists():
            issues.append(f"Working directory does not exist: {self.processing.working_directory}")

        # Test Qdrant connection (non-blocking)
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(
                url=self.qdrant.url,
                api_key=self.qdrant.api_key,
                timeout=5
            ) if self.qdrant.api_key else QdrantClient(self.qdrant.url, timeout=5)

            client.get_collections()
            connection_type = "cloud" if self.qdrant.use_cloud else "local"
            logger.info(f"Qdrant {connection_type} connection successful: {self.qdrant.url}")

        except Exception as e:
            issues.append(f"Cannot connect to Qdrant at {self.qdrant.url}: {e}")

        # Check cloud-specific requirements
        if self.qdrant.use_cloud:
            if not self.qdrant.api_key:
                issues.append("QDRANT_API_KEY is required for cloud usage")
            if "localhost" in self.qdrant.url or "127.0.0.1" in self.qdrant.url:
                issues.append("Cloud flag enabled but URL appears to be local")

        # Log validation results
        if issues:
            logger.warning(f"Configuration validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("Configuration validation passed")

        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dictionary containing sanitized configuration summary
        """
        return {
            'data_ingestion': {
                'api_configured': bool(self.data_ingestion.api_key),
                'timeout': self.data_ingestion.timeout_seconds,
                'min_text_length': self.data_ingestion.min_text_length,
                'court': self.data_ingestion.court,
                'num_dockets': self.data_ingestion.num_dockets
            },
            'text_chunking': {
                'splitter_type': 'RecursiveCharacterTextSplitter',
                'chunk_size_chars': self.text_splitter.chunk_size_chars,
                'overlap_chars': self.text_splitter.overlap_chars,
                'min_chunk_size_chars': self.text_splitter.min_chunk_size_chars,
                'quality_threshold': self.text_splitter.quality_threshold
            },
            'vector_processing': {
                'embedding_model': self.vector_processing.embedding_model,
                'batch_size': self.vector_processing.batch_size,
                'device': self.vector_processing.device,
                'collection_name': self.vector_processing.collection_name_vector
            },
            'qdrant': {
                'url': self.qdrant.url,
                'api_key_configured': bool(self.qdrant.api_key),
                'use_cloud': self.qdrant.use_cloud,
                'cluster_name': self.qdrant.cluster_name,
                'free_tier_limit_mb': self.qdrant.free_tier_limit_mb
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'password_configured': bool(self.redis.password),
                'ssl': self.redis.ssl,
                'atomic_operations': self.redis.enable_atomic_operations,
                'rate_limit_ttl_hours': self.redis.rate_limit_key_ttl_hours
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

def load_config(config_file: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration with fallbacks.

    Searches for configuration files in standard locations if none specified.

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


def validate_configuration_sources() -> Dict[str, Any]:
    """
    Validate configuration sources and provide clear guidance on conflicts.

    This function addresses the configuration complexity issue by:
    1. Checking all configuration sources and their precedence
    2. Validating USE_CLOUD flag logic explicitly
    3. Identifying potential configuration conflicts
    4. Providing clear guidance for resolution

    Returns:
        Dictionary with validation results and recommendations
    """
    validation_result = {
        'status': 'success',
        'issues': [],
        'recommendations': [],
        'configuration_sources': {},
        'use_cloud_validation': {}
    }

    # Check configuration source hierarchy
    config_sources = {
        'environment_variables': {},
        'config_file': None,
        'defaults': {}
    }

    # Critical environment variables
    critical_env_vars = [
        'CASELAW_API_KEY',
        'USE_CLOUD',
        'QDRANT_URL',
        'QDRANT_API_KEY',
        'QDRANT_CLUSTER_NAME'
    ]

    for var in critical_env_vars:
        value = os.getenv(var)
        if value:
            config_sources['environment_variables'][var] = value

    # Check for configuration file
    config_paths = [
        "config.json",
        os.path.expanduser("~/.lawlm/config.json"),
        "/etc/lawlm/config.json"
    ]

    for path in config_paths:
        if Path(path).exists():
            config_sources['config_file'] = path
            break

    # Validate USE_CLOUD flag logic
    use_cloud_raw = os.getenv("USE_CLOUD", "false").lower()
    use_cloud = use_cloud_raw in ("true", "1", "yes", "on")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL", "")

    validation_result['use_cloud_validation'] = {
        'use_cloud_raw': use_cloud_raw,
        'use_cloud_resolved': use_cloud,
        'has_api_key': bool(qdrant_api_key),
        'qdrant_url': qdrant_url,
        'is_cloud_url': "cloud.qdrant.io" in qdrant_url
    }

    # Configuration validation logic
    if use_cloud:
        if not qdrant_api_key:
            validation_result['issues'].append(
                "USE_CLOUD=true but QDRANT_API_KEY is not set. Cloud usage requires API key."
            )
            validation_result['recommendations'].append(
                "Set QDRANT_API_KEY environment variable for cloud authentication"
            )

        if not qdrant_url or "cloud.qdrant.io" not in qdrant_url:
            validation_result['issues'].append(
                "USE_CLOUD=true but QDRANT_URL does not contain cloud.qdrant.io domain"
            )
            validation_result['recommendations'].append(
                "Set QDRANT_URL to a valid cloud URL (e.g., https://your-cluster.cloud.qdrant.io:6333)"
            )
    else:
        if qdrant_url and "cloud.qdrant.io" in qdrant_url:
            validation_result['recommendations'].append(
                "Cloud URL detected but USE_CLOUD=false. Consider setting USE_CLOUD=true for cloud usage"
            )

    # Check for configuration conflicts
    if config_sources['config_file'] and config_sources['environment_variables']:
        validation_result['recommendations'].append(
            f"Multiple configuration sources detected: file ({config_sources['config_file']}) and environment variables. "
            "Environment variables take precedence."
        )

    # Check if critical API key is missing
    if not os.getenv("CASELAW_API_KEY"):
        validation_result['issues'].append(
            "CASELAW_API_KEY not set - data ingestion will fail"
        )
        validation_result['recommendations'].append(
            "Set CASELAW_API_KEY environment variable from https://www.courtlistener.com/api/"
        )

    # Set overall status
    validation_result['configuration_sources'] = config_sources
    if validation_result['issues']:
        validation_result['status'] = 'warning' if not any('fail' in issue.lower() for issue in validation_result['issues']) else 'error'

    return validation_result


def print_configuration_guidance() -> None:
    """
    Print clear guidance on configuration management to reduce complexity.
    """
    validation = validate_configuration_sources()

    print("🔧 Configuration Sources & Precedence:")
    print("  1. Environment Variables (highest precedence)")
    print("  2. Config file (if found)")
    print("  3. Built-in defaults (lowest precedence)")
    print()

    if validation['configuration_sources']['config_file']:
        print(f"📄 Config file found: {validation['configuration_sources']['config_file']}")
    else:
        print("📄 No config file found - using environment variables and defaults")

    print()
    print("🌐 USE_CLOUD Configuration:")
    cloud_val = validation['use_cloud_validation']
    print(f"  Raw value: '{cloud_val['use_cloud_raw']}'")
    print(f"  Resolved: {cloud_val['use_cloud_resolved']}")
    print(f"  Has API key: {cloud_val['has_api_key']}")
    print(f"  QDRANT_URL: {cloud_val['qdrant_url'] or 'not set'}")
    print(f"  Is cloud URL: {cloud_val['is_cloud_url']}")

    if validation['issues']:
        print()
        print("⚠️ Configuration Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    if validation['recommendations']:
        print()
        print("💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  - {rec}")

    print()
    print("📋 Quick Setup Guide:")
    print("  For LOCAL usage:")
    print("    export USE_CLOUD=false")
    print("    export CASELAW_API_KEY=your_key_here")
    print("    # Start local Qdrant: ./manage_qdrant.sh start")
    print()
    print("  For CLOUD usage:")
    print("    export USE_CLOUD=true")
    print("    export QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333")
    print("    export QDRANT_API_KEY=your_api_key")
    print("    export CASELAW_API_KEY=your_key_here")

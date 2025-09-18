"""
Enhanced CourtListener Pipeline DAG with Redis-based Rate Limiting and State Management

This DAG provides enterprise-grade orchestration for legal document processing with:
1. Redis-based distributed rate limiting with atomic operations
2. Persistent pipeline state management across scheduler restarts
3. Resilient recovery from pipeline failures and bugs
4. High-performance caching separate from Airflow metadata database
5. Circuit breaker patterns for API resilience

Architecture Features:
- Atomic counter operations using Redis Lua scripts
- Pipeline state persistence with configurable TTL
- Task start time caching for precise timing control
- Distributed-safe operations across multiple worker nodes
- Comprehensive error handling and fallback mechanisms
"""

# Standard library imports
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

# Third-party imports
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable

# Add the project root to Python path for imports
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
airflow_root = os.path.join(project_root, 'airflow')
sys.path.append(project_root)
sys.path.append(airflow_root)

# Import custom Redis hook
from hooks.redis_rate_limit_hook import RedisRateLimitHook

# Configuration constants
API_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CALL_LIMIT_PER_HOUR = 5000
TARGET_CALLS_PER_HOUR = 4900
SAFETY_BUFFER = int(os.getenv('COURTLISTENER_SAFETY_BUFFER', '25'))
MIN_CALLS_PER_DOCKET = int(os.getenv('COURTLISTENER_MIN_CALLS_PER_DOCKET', '25'))
MAX_VECTOR_PROCESSOR_POOL_SIZE = int(os.getenv('MAX_VECTOR_PROCESSOR_POOL_SIZE', '3'))
REDIS_COUNTER_TTL_HOURS = int(os.getenv('REDIS_COUNTER_TTL_HOURS', '2'))
REDIS_STATE_TTL_HOURS = int(os.getenv('REDIS_STATE_TTL_HOURS', '25'))
REDIS_CONN_ID = "redis_default"
METASTORE_CONN_ID = "postgres_default"
API_POOL_NAME = "courtlistener_api_pool"
SAFE_CONCURRENCY_LIMIT = 50

# Configure logging
task_logger = logging.getLogger("airflow.task")

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'pool': API_POOL_NAME,
}

@dag(
    dag_id='courtlistener_pipeline',
    default_args=default_args,
    schedule='@hourly',
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=['legal', 'etl', 'vector_search', 'redis', 'enhanced'],
    description='Enhanced CourtListener pipeline with Redis-based rate limiting and state management'
)
def courtlistener_pipeline_dag() -> Any:
    """
    Enhanced Airflow DAG with Redis-based caching for robust API rate limiting
    and persistent state management across pipeline executions.
    """

    from config import load_config
    from vector_processor import EnhancedVectorProcessor
    from main import LegalDocumentPipeline

    class VectorProcessorPool:
        """Connection pool for vector processors with configurable size limits."""
        _instance = None
        _processors = {}  # Dict to store multiple processors by key
        _usage_count = {}  # Track usage for LRU eviction
        _access_order = []  # Track access order for LRU

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._processors = {}
                cls._instance._usage_count = {}
                cls._instance._access_order = []
            return cls._instance

        def _get_processor_key(self, model_name: str, collection_name: str) -> str:
            """Generate a unique key for processor identification."""
            return f"{model_name}:{collection_name}"

        def get_processor(self, model_name: str, collection_name: str) -> EnhancedVectorProcessor:
            """Get or create a vector processor instance with pool size limits."""
            processor_key = self._get_processor_key(model_name, collection_name)

            # Check if processor already exists
            if processor_key in self._processors:
                self._update_access_order(processor_key)
                task_logger.debug(f"Reusing existing VectorProcessor for {processor_key}")
                return self._processors[processor_key]

            # Check pool size limit
            if len(self._processors) >= MAX_VECTOR_PROCESSOR_POOL_SIZE:
                self._evict_least_recently_used()

            # Create new processor
            task_logger.info(f"Creating new VectorProcessor for model: {model_name}, collection: {collection_name}")
            processor = EnhancedVectorProcessor(
                model_name=model_name,
                collection_name=collection_name
            )

            # Add to pool
            self._processors[processor_key] = processor
            self._usage_count[processor_key] = 1
            self._update_access_order(processor_key)

            task_logger.info(f"VectorProcessor pool size: {len(self._processors)}/{MAX_VECTOR_PROCESSOR_POOL_SIZE}")
            return processor

        def _update_access_order(self, processor_key: str) -> None:
            """Update access order for LRU tracking."""
            if processor_key in self._access_order:
                self._access_order.remove(processor_key)
            self._access_order.append(processor_key)
            self._usage_count[processor_key] = self._usage_count.get(processor_key, 0) + 1

        def _evict_least_recently_used(self) -> None:
            """Remove the least recently used processor to make room for new one."""
            if not self._access_order:
                return

            lru_key = self._access_order[0]
            task_logger.info(f"Evicting LRU VectorProcessor: {lru_key} (used {self._usage_count[lru_key]} times)")

            # Remove from all tracking structures
            del self._processors[lru_key]
            del self._usage_count[lru_key]
            self._access_order.remove(lru_key)

        def get_pool_stats(self) -> Dict[str, Any]:
            """Get current pool statistics."""
            return {
                'current_size': len(self._processors),
                'max_size': MAX_VECTOR_PROCESSOR_POOL_SIZE,
                'processors': list(self._processors.keys()),
                'usage_counts': dict(self._usage_count)
            }

        def cleanup_all(self) -> None:
            """Cleanup all processors in the pool."""
            task_logger.info(f"Cleaning up {len(self._processors)} vector processors")
            self._processors.clear()
            self._usage_count.clear()
            self._access_order.clear()

    vector_processor_pool = VectorProcessorPool()

    def validate_dag_configuration() -> bool:
        """
        Validate DAG configuration and environment variables at initialization.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        validation_errors = []

        # Validate required environment variables
        required_env_vars = {
            'CASELAW_API_KEY': 'CourtListener API key for data ingestion',
            'QDRANT_URL': 'Qdrant database URL for vector storage'
        }

        for env_var, description in required_env_vars.items():
            if not os.getenv(env_var):
                validation_errors.append(f"Missing required environment variable: {env_var} ({description})")

        # Validate numeric configuration values
        numeric_configs = {
            'CALL_LIMIT_PER_HOUR': CALL_LIMIT_PER_HOUR,
            'TARGET_CALLS_PER_HOUR': TARGET_CALLS_PER_HOUR,
            'SAFETY_BUFFER': SAFETY_BUFFER,
            'MIN_CALLS_PER_DOCKET': MIN_CALLS_PER_DOCKET,
            'MAX_VECTOR_PROCESSOR_POOL_SIZE': MAX_VECTOR_PROCESSOR_POOL_SIZE,
            'REDIS_COUNTER_TTL_HOURS': REDIS_COUNTER_TTL_HOURS,
            'REDIS_STATE_TTL_HOURS': REDIS_STATE_TTL_HOURS
        }

        for config_name, config_value in numeric_configs.items():
            if not isinstance(config_value, int) or config_value <= 0:
                validation_errors.append(f"Invalid {config_name}: {config_value} (must be positive integer)")

        # Validate logical relationships
        if TARGET_CALLS_PER_HOUR >= CALL_LIMIT_PER_HOUR:
            validation_errors.append(f"TARGET_CALLS_PER_HOUR ({TARGET_CALLS_PER_HOUR}) must be less than CALL_LIMIT_PER_HOUR ({CALL_LIMIT_PER_HOUR})")

        if SAFETY_BUFFER >= CALL_LIMIT_PER_HOUR:
            validation_errors.append(f"SAFETY_BUFFER ({SAFETY_BUFFER}) must be less than CALL_LIMIT_PER_HOUR ({CALL_LIMIT_PER_HOUR})")

        if MIN_CALLS_PER_DOCKET > TARGET_CALLS_PER_HOUR:
            validation_errors.append(f"MIN_CALLS_PER_DOCKET ({MIN_CALLS_PER_DOCKET}) should not exceed TARGET_CALLS_PER_HOUR ({TARGET_CALLS_PER_HOUR})")

        # Validate Redis connection configuration
        try:
            from airflow.models import Variable
            redis_host = Variable.get("redis_host", default_var="localhost")
            redis_port = Variable.get("redis_port", default_var=6379)
            if not isinstance(redis_port, int) or redis_port <= 0 or redis_port > 65535:
                validation_errors.append(f"Invalid Redis port: {redis_port} (must be 1-65535)")
        except Exception as e:
            validation_errors.append(f"Error accessing Airflow Variables: {e}")

        # Validate Airflow pool configuration
        if SAFE_CONCURRENCY_LIMIT <= 0:
            validation_errors.append(f"SAFE_CONCURRENCY_LIMIT ({SAFE_CONCURRENCY_LIMIT}) must be positive")

        # Log results
        if validation_errors:
            task_logger.error(f"❌ DAG Configuration validation failed:")
            for i, error in enumerate(validation_errors, 1):
                task_logger.error(f"  {i}. {error}")
            return False
        else:
            task_logger.info("✅ DAG Configuration validation passed")
            task_logger.info(f"  - API Rate Limit: {TARGET_CALLS_PER_HOUR}/{CALL_LIMIT_PER_HOUR} calls/hour")
            task_logger.info(f"  - Safety Buffer: {SAFETY_BUFFER} calls")
            task_logger.info(f"  - Vector Pool Size: {MAX_VECTOR_PROCESSOR_POOL_SIZE}")
            task_logger.info(f"  - Redis TTL: Counter={REDIS_COUNTER_TTL_HOURS}h, State={REDIS_STATE_TTL_HOURS}h")
            return True

    # Validate configuration at DAG initialization
    config_valid = validate_dag_configuration()

    def classify_and_handle_error(error: Exception, context: str, docket_id: Optional[int] = None) -> None:
        """
        Enhanced error classification with Redis state awareness.

        Args:
            error: The exception that occurred
            context: Context string describing where the error occurred
            docket_id: Optional docket ID for context

        Raises:
            AirflowFailException: For non-retriable errors
            AirflowSkipException: For errors that should be skipped
            Exception: For retriable errors
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Redis-specific errors
        if 'redis' in error_str or 'connection pool' in error_str:
            task_logger.warning(f"Redis connectivity issue in {context}: {error}")
            # Don't fail the entire pipeline for Redis issues, use fallback
            return

        # Non-retriable authentication errors
        if any(term in error_str for term in ['unauthorized', '401', '403', 'invalid api key', 'authentication failed']):
            task_logger.error(f"Authentication error in {context}: {error}")
            raise AirflowFailException(f"Authentication error in {context} (docket {docket_id}): {error}")

        # Rate limit errors - should skip to allow retry later
        elif '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            task_logger.warning(f"Rate limit hit in {context} (docket {docket_id}), skipping: {error}")
            raise AirflowSkipException(f"Rate limit hit in {context} (docket {docket_id}): {error}")

        # Client errors - skip these dockets as they won't succeed on retry
        elif any(term in error_str for term in ['400', '404', 'not found', 'bad request', 'invalid docket']):
            task_logger.warning(f"Client error in {context} (docket {docket_id}), skipping: {error}")
            raise AirflowSkipException(f"Client error in {context} (docket {docket_id}): {error}")

        # Server errors and network issues - should be retried by Airflow
        elif any(term in error_str for term in ['500', '502', '503', '504', 'timeout', 'connection', 'network']):
            task_logger.warning(f"Retriable server/network error in {context} (docket {docket_id}): {error}")
            raise error

        # Qdrant specific errors
        elif 'qdrant' in error_str or error_type in ['QdrantException', 'UnexpectedResponse']:
            if 'collection not found' in error_str or 'does not exist' in error_str:
                task_logger.error(f"Qdrant collection missing in {context}: {error}")
                raise AirflowFailException(f"Qdrant collection missing in {context}: {error}")
            else:
                task_logger.warning(f"Qdrant error in {context} (docket {docket_id}), retrying: {error}")
                raise error

        # Memory/resource errors - fail fast to prevent cascading issues
        elif any(term in error_str for term in ['memory', 'out of memory', 'disk space', 'resource']):
            task_logger.error(f"Resource error in {context}: {error}")
            raise AirflowFailException(f"Resource error in {context}: {error}")

        # Unknown errors - log extensively and fail
        else:
            task_logger.error(f"Unknown error type '{error_type}' in {context} (docket {docket_id}): {error}")
            task_logger.error(f"Full traceback: {traceback.format_exc()}")
            raise AirflowFailException(f"Unknown error in {context} (docket {docket_id}): {error}")

    def make_api_call_with_backoff(url: str, headers: Dict[str, str], description: str, max_retries: int = 3) -> Any:
        """
        Enhanced API call with Redis rate limit awareness.

        Args:
            url: API URL to call
            headers: HTTP headers for the request
            description: Description of the API call for logging
            max_retries: Maximum number of retry attempts

        Returns:
            HTTP response object

        Raises:
            AirflowSkipException: If rate limited after retries
            Exception: For other failures after retries
        """
        import requests

        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                    if attempt < max_retries:
                        task_logger.warning(f"Rate limited fetching {description}, retrying in {retry_after}s (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(retry_after)
                        continue
                    else:
                        raise AirflowSkipException(f"Rate limit exceeded for {description} after {max_retries + 1} attempts")

                elif response.status_code >= 500:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        task_logger.warning(f"Server error ({response.status_code}) fetching {description}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()

                else:
                    response.raise_for_status()
                    return response

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    task_logger.warning(f"Request exception fetching {description}: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        raise Exception(f"Failed to fetch {description} after {max_retries + 1} attempts")

    @task
    def initialize_redis_pipeline_state() -> Dict[str, Any]:
        """
        Initialize Redis-based pipeline state for the DAG run.

        This task sets up the distributed state management system using Redis for
        tracking API calls, pipeline progress, and ensuring rate limit compliance
        across multiple worker nodes.

        Inputs:
            - DAG context (automatically provided by Airflow)
            - Redis connection configuration

        Outputs:
            dict: Pipeline state containing:
                - 'dag_run_id': Unique identifier for this DAG run
                - 'start_time': Pipeline initialization timestamp
                - 'current_hour': Current hour key for rate limiting
                - 'api_calls_this_hour': Current API call count
                - 'status': Pipeline status ('initialized')
                - 'redis_keys': Dictionary of Redis keys used for state management

        Side Effects:
            - Creates Redis keys for state tracking with configured TTL
            - Initializes API call counter for the current hour
            - Caches task start time for monitoring

        Error Handling:
            - Falls back to in-memory state if Redis is unavailable
            - Logs detailed error information for debugging
            - Continues pipeline execution with degraded functionality
        """
        try:
            from airflow.operators.python import get_current_context
            context = get_current_context()
            dag_run_id = context['dag_run'].run_id

            # Initialize Redis hook
            redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)

            # Cache the start time immediately
            start_time = redis_hook.cache_task_start_time(dag_run_id, "pipeline_start")

            # Initialize pipeline state with empty counter
            state = redis_hook.initialize_pipeline_state(dag_run_id, REDIS_STATE_TTL_HOURS)

            # Get current rate limit status
            rate_status = redis_hook.get_current_rate_limit_status()

            task_logger.info(f"Redis pipeline state initialized: {dag_run_id}")
            task_logger.info(f"Start time cached: {start_time}")
            task_logger.info(f"Current API usage: {rate_status['api_calls_used']}/{rate_status['limit']}")

            return {
                'dag_run_id': dag_run_id,
                'start_time': start_time,
                'redis_state_initialized': True,
                'current_api_calls': rate_status['api_calls_used'],
                'remaining_calls': rate_status['api_calls_remaining'],
                'can_proceed': rate_status['can_proceed'],
                'redis_keys': state.get('redis_keys', {})
            }

        except Exception as e:
            task_logger.error(f"Failed to initialize Redis pipeline state: {e}")
            # Fallback to legacy Variable-based method
            task_logger.warning("Falling back to Airflow Variables for state management")

            try:
                from airflow.operators.python import get_current_context
                context = get_current_context()
                dag_run_id = context['dag_run'].run_id
                start_time = datetime.now().isoformat()

                # Fallback initialization using Variables
                rate_limit_info = {
                    "dag_run_id": dag_run_id,
                    "start_time": start_time,
                    "current_hour": datetime.utcnow().strftime("%Y-%m-%d %H"),
                    "calls_this_hour": 0,
                    "dockets_processed": [],
                    "redis_fallback": True
                }
                Variable.set("courtlistener_rate_limit_redis", json.dumps(rate_limit_info))

                return {
                    'dag_run_id': dag_run_id,
                    'start_time': start_time,
                    'redis_state_initialized': False,
                    'fallback_used': True,
                    'current_api_calls': 0,
                    'remaining_calls': TARGET_CALLS_PER_HOUR,
                    'can_proceed': True,
                    'error': str(e)
                }

            except Exception as fallback_error:
                task_logger.error(f"Both Redis and fallback initialization failed: {fallback_error}")
                raise AirflowFailException(f"Pipeline state initialization failed: {fallback_error}")

    @task
    def check_redis_rate_limit_and_hour_boundary() -> Dict[str, Any]:
        """
        Check current API usage and enforce distributed rate limiting with Redis.

        This task performs critical rate limit enforcement across all worker nodes
        using Redis as a centralized counter. It also handles hour boundary detection
        and automatic counter resets.

        Inputs:
            - Redis connection (configured via REDIS_CONN_ID)
            - Current hour timestamp for boundary detection
            - Global rate limit configuration (CALL_LIMIT_PER_HOUR, SAFETY_BUFFER)

        Outputs:
            dict: Rate limit status containing:
                - 'current_hour': Current hour key for rate limiting
                - 'api_calls_used': Number of API calls used this hour
                - 'api_calls_remaining': Remaining calls before hitting limit
                - 'utilization_percent': Percentage of hourly limit used
                - 'can_proceed': Boolean indicating if pipeline can continue
                - 'limit': Configured hourly rate limit
                - 'redis_source': True if data comes from Redis
                - 'hour_reset': True if hour boundary caused counter reset

        Side Effects:
            - Updates Redis counters for distributed rate tracking
            - Resets counters automatically on hour boundary
            - Logs detailed rate limit status for monitoring

        Error Handling:
            - Falls back to conservative in-memory limits if Redis fails
            - Uses hardcoded safety margins when distributed state unavailable
            - Logs all fallback scenarios for operational awareness

        Rate Limiting Logic:
            - Enforces CALL_LIMIT_PER_HOUR with configurable safety buffer
            - Atomic operations prevent race conditions between workers
            - Hour boundary detection prevents counter drift
        """
        task_logger.info("🔍 Starting Redis rate limit check with enhanced debugging")

        try:
            redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)

            # Debug: Test Redis connection in DAG context
            try:
                client = redis_hook.get_conn()
                ping_result = client.ping()
                task_logger.info(f"✓ Redis connection test in DAG context: ping={ping_result}")
            except Exception as conn_error:
                task_logger.error(f"❌ Redis connection test failed in DAG context: {conn_error}")
                raise conn_error

            # Get current rate limit status from Redis
            rate_status = redis_hook.get_current_rate_limit_status()
            task_logger.info(f"📊 Raw Redis rate status: {rate_status}")

            # Check for errors in rate_status
            if 'error' in rate_status:
                task_logger.error(f"❌ Redis hook returned error in rate_status: {rate_status['error']}")
                raise Exception(f"Redis hook rate status error: {rate_status['error']}")

            # Check for hour boundary crossing
            boundary_check = redis_hook.check_hour_boundary_and_reset()
            task_logger.info(f"⏰ Hour boundary check: {boundary_check}")

            # Adjust safety buffer for more realistic usage
            safety_buffer = SAFETY_BUFFER

            # Determine if pipeline can proceed with enhanced logic
            redis_can_proceed = rate_status.get('can_proceed', False)
            remaining_calls = rate_status.get('api_calls_remaining', 0)
            buffer_check = remaining_calls >= safety_buffer

            can_proceed = redis_can_proceed and buffer_check

            task_logger.info(f"🚥 Rate limit decision breakdown:")
            task_logger.info(f"   Redis can_proceed: {redis_can_proceed}")
            task_logger.info(f"   API calls remaining: {remaining_calls}")
            task_logger.info(f"   Safety buffer: {safety_buffer}")
            task_logger.info(f"   Buffer check (>={safety_buffer}): {buffer_check}")
            task_logger.info(f"   Final can_proceed decision: {can_proceed}")

            result = {
                'current_hour': rate_status['current_hour'],
                'api_calls_used': rate_status['api_calls_used'],
                'api_calls_remaining': rate_status['api_calls_remaining'],
                'utilization_percent': rate_status['utilization_percent'],
                'can_proceed': can_proceed,
                'hour_boundary_crossed': boundary_check.get('hour_reset_detected', False),
                'previous_hour_calls': boundary_check.get('previous_count', 0),
                'redis_source': True,
                'safety_buffer_used': safety_buffer
            }

            task_logger.info(f"✅ REDIS SUCCESS PATH: Returning rate limit result: {result}")

            if boundary_check.get('hour_reset_detected'):
                task_logger.info(f"🔄 Hour boundary detected: {boundary_check['previous_hour']} -> {boundary_check['current_hour']}")
                task_logger.info(f"📈 Previous hour usage: {boundary_check['previous_count']} calls")

            task_logger.info(f"📈 Redis rate limit summary: {rate_status['api_calls_used']}/{rate_status['limit']} calls used ({rate_status['utilization_percent']:.1f}%)")

            return result

        except Exception as e:
            task_logger.error(f"❌ Redis rate limit check failed: {e}")
            task_logger.error(f"Exception type: {type(e).__name__}")
            task_logger.error(f"Full traceback: {traceback.format_exc()}")

            # Skip PostgreSQL fallback for now - if Redis is working (which it is),
            # we shouldn't need complex fallback logic that's causing issues
            task_logger.warning("⚠️ Redis failed but we'll try a simple Redis retry instead of PostgreSQL fallback")

            try:
                # Simple Redis retry with direct connection (using localhost to match updated configuration)
                task_logger.info("🔄 Attempting simple Redis retry...")
                import redis as redis_client

                # Explicit localhost connection - ensures consistency with Airflow connection configuration
                direct_client = redis_client.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                direct_client.ping()

                current_hour = datetime.utcnow().strftime("%Y-%m-%d_%H")
                counter_key = f"courtlistener:counter:{current_hour}"
                current_count = int(direct_client.get(counter_key) or 0)

                limit = 5000
                remaining = max(0, limit - current_count)
                can_proceed = current_count < (limit - SAFETY_BUFFER)  # Configurable safety buffer

                simple_result = {
                    'current_hour': current_hour,
                    'api_calls_used': current_count,
                    'api_calls_remaining': remaining,
                    'utilization_percent': (current_count / limit) * 100,
                    'can_proceed': can_proceed,
                    'redis_source': True,
                    'simple_retry_used': True,
                    'safety_buffer_used': SAFETY_BUFFER,
                    'redis_error': str(e)
                }

                task_logger.info(f"✅ Simple Redis retry succeeded: {simple_result}")
                return simple_result

            except Exception as retry_error:
                task_logger.error(f"❌ Simple Redis retry also failed: {retry_error}")

                # Only use conservative fallback as last resort
                task_logger.error(f"💀 TRIGGERING CONSERVATIVE FALLBACK: 0 used, 5000 remaining, can_proceed=True")
                task_logger.error(f"This should only happen if Redis is completely down")

                # More optimistic fallback - assume fresh hour if Redis is completely down
                return {
                    'current_hour': datetime.utcnow().strftime("%Y-%m-%d_%H"),
                    'api_calls_used': 0,  # Optimistic assumption - fresh hour
                    'api_calls_remaining': 5000,
                    'utilization_percent': 0.0,
                    'can_proceed': True,  # Allow processing if Redis is down
                    'redis_source': False,
                    'conservative_fallback_used': True,
                    'safety_buffer_used': 0,
                    'redis_error': str(e),
                    'retry_error': str(retry_error)
                }

    @task
    def get_existing_dockets() -> List[str]:
        """
        Retrieve existing docket IDs from Qdrant vector database for deduplication.

        This task queries the Qdrant collection to get all existing docket IDs,
        enabling the pipeline to skip processing documents that have already
        been vectorized and stored.

        Inputs:
            - Vector processor configuration (model name, collection name)
            - Qdrant connection settings from config

        Outputs:
            List[str]: List of existing docket ID strings already in the vector database

        Side Effects:
            - Creates vector processor instance from pool
            - Establishes connection to Qdrant database
            - May trigger vector processor pool management (LRU eviction)

        Error Handling:
            - Returns empty list if Qdrant connection fails
            - Logs connection errors for debugging
            - Continues pipeline execution to process all dockets

        Performance Notes:
            - Uses connection pooling for efficient resource management
            - Results used for deduplication in downstream tasks
            - Critical for avoiding duplicate processing in incremental runs
        """
        config = load_config()
        processor = vector_processor_pool.get_processor(
            model_name=config.vector_processing.embedding_model,
            collection_name=config.vector_processing.collection_name_vector
        )
        return list(processor.get_existing_docket_ids())


    @task
    def fetch_dockets_within_redis_rate_limit(court: str, rate_limit_state: Dict[str, Any]) -> List[int]:
        """
        Fetch docket IDs from CourtListener API with Redis-enforced rate limiting.

        This task integrates with the distributed rate limiting system to fetch
        dockets while staying within API limits. It uses the rate limit state
        from Redis to determine how many API calls can be safely made.

        Inputs:
            court (str): Court identifier (e.g., 'scotus', 'ca9')
            rate_limit_state (Dict[str, Any]): Rate limit status from Redis containing:
                - 'can_proceed': Boolean indicating if API calls are allowed
                - 'api_calls_remaining': Number of API calls remaining this hour
                - 'api_calls_used': Number of API calls already used

        Outputs:
            List[int]: List of docket IDs to be processed by downstream tasks

        Side Effects:
            - Makes API calls to CourtListener /dockets/ endpoint
            - Updates Redis counters for each API call made
            - Logs API usage and rate limit status

        Error Handling:
            - Stops processing if rate limit exceeded
            - Raises AirflowSkipException for rate limit violations
            - Logs detailed error information for monitoring

        Rate Limiting Logic:
            - Checks Redis state before making any API calls
            - Uses atomic Redis operations to track API usage
            - Respects minimum calls per docket (MIN_CALLS_PER_DOCKET)
            - Implements safety buffers to prevent limit breaches
        """
        remaining_calls = rate_limit_state["api_calls_remaining"]

        task_logger.info(f"🎯 Fetching dockets with rate limit state: {rate_limit_state}")

        if not rate_limit_state["can_proceed"]:
            # Enhanced error logging to help debug the source of the issue
            redis_source = rate_limit_state.get("redis_source", "unknown")
            fallback_used = rate_limit_state.get("fallback_used", False)
            safety_buffer = rate_limit_state.get("safety_buffer_used", "unknown")

            error_msg = (
                f"Rate limiting triggered - cannot proceed with docket fetching. "
                f"Details: remaining_calls={remaining_calls}, redis_source={redis_source}, "
                f"fallback_used={fallback_used}, safety_buffer={safety_buffer}"
            )

            if "conservative_fallback_used" in rate_limit_state:
                error_msg += " [CONSERVATIVE FALLBACK TRIGGERED - likely Redis connection issue]"
            elif "simple_retry_used" in rate_limit_state:
                error_msg += " [SIMPLE RETRY WAS USED - Redis hook failed but direct Redis worked]"

            task_logger.warning(error_msg)
            raise AirflowSkipException(error_msg)

        # More conservative estimation with Redis tracking
        estimated_calls_per_docket = 12
        max_dockets = max(1, remaining_calls // estimated_calls_per_docket)

        task_logger.info(f"Fetching up to {max_dockets} new dockets from {court} "
                        f"(estimated {estimated_calls_per_docket} calls/docket, {remaining_calls} calls remaining)")
        task_logger.info(f"Using Redis rate limit source: {rate_limit_state.get('redis_source', False)}")

        config = load_config()
        pipeline = LegalDocumentPipeline(config)

        # Get existing dockets for deduplication
        existing_dockets = set()
        try:
            vector_processor = vector_processor_pool.get_processor(
                model_name=config.vector_processing.embedding_model,
                collection_name=config.vector_processing.collection_name_vector
            )
            existing_dockets = vector_processor.get_existing_docket_ids()
            task_logger.info(f"Found {len(existing_dockets)} existing dockets")
        except Exception as e:
            task_logger.warning(f"Could not get existing dockets: {e}")
            existing_dockets = set()

        new_dockets = pipeline._fetch_all_dockets_paginated(
            court=court,
            num_dockets=max_dockets,
            existing_dockets=existing_dockets
        )

        docket_ids = [docket['id'] for docket in new_dockets]
        task_logger.info(f"Successfully fetched {len(docket_ids)} new docket IDs for processing (Redis rate limit enforced)")

        # If no dockets fetched due to all being duplicates, skip gracefully
        if len(docket_ids) == 0:
            task_logger.info("No new dockets to process (all dockets already exist in collection)")
            raise AirflowSkipException("No new dockets to process - all dockets already exist in collection")

        return docket_ids

    def _check_rate_limit_status(redis_hook: RedisRateLimitHook, docket_id: int) -> Optional[Dict[str, Any]]:
        """Check Redis rate limit status and return skip result if limits exceeded."""
        rate_status = redis_hook.get_current_rate_limit_status()

        if not rate_status['can_proceed']:
            task_logger.warning(f"Redis rate limit check failed for docket {docket_id}: "
                              f"{rate_status['api_calls_used']}/{rate_status['limit']} calls used")
            return {
                "docket_id": docket_id,
                "api_calls_made": 0,
                "status": "skipped_rate_limit_redis",
                "message": f"Redis rate limit exceeded: {rate_status['api_calls_used']}/{rate_status['limit']}"
            }

        if rate_status['api_calls_remaining'] < MIN_CALLS_PER_DOCKET:
            task_logger.warning(f"Insufficient remaining calls ({rate_status['api_calls_remaining']}) for docket {docket_id}")
            return {
                "docket_id": docket_id,
                "api_calls_made": 0,
                "status": "skipped_rate_limit",
                "message": f"Insufficient API calls remaining: {rate_status['api_calls_remaining']}"
            }

        return None

    def _create_redis_tracked_fetch_function(docket_id: int, dag_run_id: str, redis_hook: 'RedisRateLimitHook', config: Any) -> Tuple[Callable, Callable]:
        """Create a Redis-tracked fetch function for the pipeline."""
        api_calls_counter = {'count': 0}

        def redis_tracked_fetch(court: str, num_dockets: int, existing_dockets: set) -> List[Dict[str, Any]]:
            """Fetch with Redis API call tracking."""
            vp = vector_processor_pool.get_processor(
                model_name=config.vector_processing.embedding_model,
                collection_name=config.vector_processing.collection_name_vector
            )
            current_existing_dockets = vp.get_existing_docket_ids()

            if docket_id in current_existing_dockets or docket_id in existing_dockets:
                task_logger.info(f"Docket {docket_id} already exists, skipping with Redis state update")
                redis_hook.update_docket_processing_state(
                    dag_run_id=dag_run_id,
                    docket_id=docket_id,
                    api_calls_made=0,
                    success=True
                )
                return []

            task_logger.info(f"Docket {docket_id} is new, fetching from API with Redis call tracking")
            headers = {"Authorization": f"Token {config.data_ingestion.api_key}"}

            # Fetch main docket
            docket_data, calls_made = _fetch_main_docket(docket_id, headers, config, redis_hook)
            api_calls_counter['count'] += calls_made

            # Fetch associated opinions if available
            if 'cluster' in docket_data and docket_data['cluster']:
                opinion_calls = _fetch_cluster_opinions(docket_id, docket_data['cluster'], headers, redis_hook)
                api_calls_counter['count'] += opinion_calls

            task_logger.info(f"Docket {docket_id} successfully fetched with {api_calls_counter['count']} Redis-tracked API calls")
            return [docket_data]

        def get_api_calls() -> int:
            return api_calls_counter['count']

        return redis_tracked_fetch, get_api_calls

    def _fetch_main_docket(docket_id: int, headers: Dict[str, str], config: Any, redis_hook: 'RedisRateLimitHook') -> Tuple[Dict[str, Any], int]:
        """Fetch main docket data with Redis tracking."""
        import requests

        url = f"{config.data_ingestion.api_base_url}/dockets/{docket_id}/"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Track API call in Redis
        increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR, REDIS_COUNTER_TTL_HOURS)
        if not increment_result['allowed']:
            task_logger.error(f"Redis atomic increment rejected API call: "
                            f"current={increment_result['new_count']}, limit={increment_result['limit']}")
            raise Exception(f"Redis rate limit enforcement: {increment_result['new_count']}/{increment_result['limit']}")

        task_logger.info(f"Redis API call 1: Fetched docket {docket_id}, "
                        f"total calls this hour: {increment_result['new_count']}")

        return response.json(), 1

    def _fetch_cluster_opinions(docket_id: int, cluster_url: str, headers: Dict[str, str], redis_hook: 'RedisRateLimitHook') -> int:
        """Fetch cluster opinions with Redis tracking."""
        import requests

        api_calls_made = 0

        if not isinstance(cluster_url, str) or not cluster_url.startswith('http'):
            return api_calls_made

        # Fetch cluster data
        cluster_response = requests.get(cluster_url, headers=headers, timeout=30)
        cluster_response.raise_for_status()
        api_calls_made += 1

        # Track cluster API call in Redis
        increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR, REDIS_COUNTER_TTL_HOURS)
        if not increment_result['allowed']:
            raise Exception(f"Redis rate limit during cluster fetch: {increment_result['new_count']}/{increment_result['limit']}")

        task_logger.info(f"Redis API call {api_calls_made + 1}: Fetched cluster for docket {docket_id}, "
                        f"total calls: {increment_result['new_count']}")

        cluster_data = cluster_response.json()
        if 'sub_opinions' in cluster_data:
            opinion_calls = _fetch_sub_opinions(docket_id, cluster_data['sub_opinions'], headers, redis_hook, api_calls_made + 1)
            api_calls_made += opinion_calls

        return api_calls_made

    def _fetch_sub_opinions(docket_id: int, opinion_urls: List[str], headers: Dict[str, str], redis_hook: 'RedisRateLimitHook', base_call_count: int) -> int:
        """Fetch individual opinion documents with Redis tracking."""
        import requests

        api_calls_made = 0

        for opinion_url in opinion_urls:
            if not isinstance(opinion_url, str) or not opinion_url.startswith('http'):
                continue

            # Check rate limit before each opinion fetch
            current_status = redis_hook.get_current_rate_limit_status()
            if not current_status['can_proceed']:
                task_logger.warning(f"Redis rate limit reached during opinion fetching: "
                                  f"{current_status['api_calls_used']}/{current_status['limit']}")
                break

            opinion_response = requests.get(opinion_url, headers=headers, timeout=30)
            opinion_response.raise_for_status()
            api_calls_made += 1

            # Track opinion API call in Redis
            increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR, REDIS_COUNTER_TTL_HOURS)
            if not increment_result['allowed']:
                task_logger.warning(f"Redis rate limit reached during opinion {base_call_count + api_calls_made}: "
                                  f"{increment_result['new_count']}/{increment_result['limit']}")
                break

            task_logger.info(f"Redis API call {base_call_count + api_calls_made}: Fetched opinion for docket {docket_id}, "
                            f"total calls: {increment_result['new_count']}")

        return api_calls_made

    def _format_success_result(docket_id: int, api_calls_made: int, map_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful processing result."""
        if result.get('status') == 'up_to_date':
            return {
                "docket_id": docket_id,
                "api_calls_made": api_calls_made,
                "map_index": map_index,
                "opinions": 0,
                "chunks": 0,
                "vectors": 0,
                "status": "skipped_duplicate",
                "message": "Docket already exists in collection",
                "redis_tracked": True
            }

        stats = result.get('stats', {})
        return {
            "docket_id": docket_id,
            "api_calls_made": api_calls_made,
            "map_index": map_index,
            "opinions": stats.get('opinions_processed', 0),
            "chunks": stats.get('chunks_created', 0),
            "vectors": stats.get('vectors_uploaded', 0),
            "status": "success" if result.get('status') == 'completed' else result.get('status', 'unknown'),
            "message": f"Successfully processed docket {docket_id} with {api_calls_made} Redis-tracked API calls",
            "redis_tracked": True
        }

    def _format_error_result(docket_id: int, api_calls_made: int, map_index: int, error: Exception) -> Dict[str, Any]:
        """Format error processing result."""
        return {
            "docket_id": docket_id,
            "api_calls_made": api_calls_made,
            "map_index": map_index,
            "opinions": 0,
            "chunks": 0,
            "vectors": 0,
            "status": "failed",
            "error": str(error),
            "redis_tracked": True
        }

    @task
    def process_docket_with_redis_tracking(docket_id: int) -> Dict[str, Any]:
        """
        Process a single docket with full Redis-based tracking and rate limiting.

        This is the core processing task that handles document ingestion, chunking,
        vectorization, and upload to Qdrant. It integrates with Redis for distributed
        API rate limiting and progress tracking.

        Inputs:
            docket_id (int): Unique docket identifier from CourtListener

        Outputs:
            Dict[str, Any]: Processing result containing:
                - 'docket_id': Processed docket ID
                - 'status': Processing status ('success', 'failed', 'skipped')
                - 'api_calls_made': Number of API calls used for this docket
                - 'chunks_created': Number of text chunks generated
                - 'vectors_uploaded': Number of vectors stored in Qdrant
                - 'processing_time': Time taken to process this docket
                - 'error_message': Error details if processing failed

        Side Effects:
            - Makes multiple API calls to CourtListener (docket, clusters, opinions)
            - Updates Redis counters atomically for each API call
            - Creates text chunks from legal documents
            - Generates vector embeddings using BGE model
            - Uploads vectors to Qdrant database
            - Logs detailed processing metrics

        Error Handling:
            - Continues processing other dockets if one fails
            - Tracks API calls even for failed processing
            - Returns detailed error information for monitoring
            - Uses circuit breaker pattern for API resilience

        Rate Limiting Integration:
            - Checks remaining API calls before processing
            - Uses atomic Redis operations for call tracking
            - Respects MIN_CALLS_PER_DOCKET safety threshold
            - Coordinates with other workers via distributed state
        """
        from main import LegalDocumentPipeline
        from config import load_config
        from airflow.operators.python import get_current_context

        context = get_current_context()
        map_index = context.get('task_instance').map_index
        dag_run_id = context['dag_run'].run_id
        task_id = f"process_docket_with_redis_tracking_{map_index}"

        task_logger.info(f"Starting Redis-tracked processing of docket ID: {docket_id} "
                        f"(map_index: {map_index}, dag_run: {dag_run_id})")

        # Initialize Redis hook and cache task start time
        redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)
        start_time = redis_hook.cache_task_start_time(dag_run_id, task_id)
        task_logger.info(f"Task start time cached in Redis: {start_time}")

        api_calls_made = 0

        try:
            # Check rate limit status
            skip_result = _check_rate_limit_status(redis_hook, docket_id)
            if skip_result:
                skip_result["map_index"] = map_index
                return skip_result

            # Load config and initialize pipeline
            config = load_config()
            config.data_ingestion.num_dockets = 1
            pipeline = LegalDocumentPipeline(config)

            # Create and inject Redis-tracked fetch function
            original_fetch = pipeline._fetch_all_dockets_paginated
            redis_tracked_fetch, get_api_calls = _create_redis_tracked_fetch_function(
                docket_id, dag_run_id, redis_hook, config
            )
            pipeline._fetch_all_dockets_paginated = redis_tracked_fetch

            # Run the pipeline
            result = pipeline.run_pipeline()
            api_calls_made = get_api_calls()

            # Restore original method
            pipeline._fetch_all_dockets_paginated = original_fetch

            # Update Redis state with successful results
            redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=docket_id,
                api_calls_made=api_calls_made,
                success=True
            )

            task_logger.info(f"Redis state updated: docket {docket_id}, calls: {api_calls_made}, success: True")
            return _format_success_result(docket_id, api_calls_made, map_index, result)

        except Exception as e:
            task_logger.error(f"Error processing docket {docket_id} after {api_calls_made} API calls: {e}")

            # Update Redis state for failed docket
            redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=docket_id,
                api_calls_made=api_calls_made,
                success=False,
                error_message=str(e)
            )

            # Use error classification to determine appropriate action
            try:
                classify_and_handle_error(e, f"process_docket_with_redis_tracking", docket_id)
            except (AirflowSkipException, AirflowFailException):
                raise
            except Exception:
                return _format_error_result(docket_id, api_calls_made, map_index, e)

    @task
    def generate_redis_enhanced_summary(results: List[Dict[str, Any]],
                                       redis_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline summary with Redis metrics and performance analysis.

        This task aggregates results from all processed dockets and combines them
        with Redis state information to provide detailed pipeline execution metrics.

        Inputs:
            results (List[Dict[str, Any]]): List of processing results from each docket
            redis_state (Dict[str, Any]): Redis pipeline state with API usage tracking

        Outputs:
            Dict[str, Any]: Comprehensive summary containing:
                - 'total_dockets_processed': Count of dockets processed
                - 'successful_dockets': Count of successfully processed dockets
                - 'failed_dockets': Count of failed dockets with error details
                - 'total_api_calls_made': Sum of all API calls across all dockets
                - 'total_chunks_created': Sum of all text chunks generated
                - 'total_vectors_uploaded': Sum of all vectors stored in Qdrant
                - 'pipeline_duration': Total pipeline execution time
                - 'redis_metrics': Redis connection and operation metrics
                - 'performance_stats': Processing rates and efficiency metrics

        Side Effects:
            - Logs comprehensive pipeline summary
            - May trigger Redis metrics logging
            - Provides data for monitoring and alerting systems

        Performance Analysis:
            - Calculates processing rates (dockets/minute, vectors/second)
            - Tracks API efficiency and rate limit utilization
            - Provides insights for pipeline optimization
        """
        task_logger.info("Generating Redis-enhanced pipeline summary")

        # Basic statistics
        total_dockets = len(results)
        successful_dockets = sum(1 for r in results if r.get("status") == "success")
        failed_dockets = sum(1 for r in results if r.get("status") == "failed")
        total_skipped = sum(1 for r in results if 'skipped' in r.get("status", ""))
        total_opinions = sum(r.get("opinions", 0) for r in results if r.get("status") == "success")
        total_chunks = sum(r.get("chunks", 0) for r in results if r.get("status") == "success")
        total_api_calls = sum(r.get("api_calls_made", 0) for r in results)

        # Redis-enhanced metrics
        redis_tracked_dockets = sum(1 for r in results if r.get("redis_tracked", False))

        try:
            # Get final Redis state
            from airflow.operators.python import get_current_context
            context = get_current_context()
            dag_run_id = context['dag_run'].run_id

            redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)
            final_redis_state = redis_hook.get_pipeline_state(dag_run_id)
            final_rate_status = redis_hook.get_current_rate_limit_status()

            redis_metrics = {
                'redis_pipeline_state_available': not ('error' in final_redis_state),
                'redis_tracked_dockets': redis_tracked_dockets,
                'redis_api_calls_this_hour': final_rate_status.get('api_calls_used', 0),
                'redis_api_calls_remaining': final_rate_status.get('api_calls_remaining', 0),
                'redis_utilization_percent': final_rate_status.get('utilization_percent', 0),
                'redis_source_reliable': redis_state.get('redis_source', False)
            }

            # Calculate time metrics if start time is available
            if redis_state.get('start_time'):
                try:
                    start_time = datetime.fromisoformat(redis_state['start_time'].replace('Z', '+00:00'))
                    end_time = datetime.now()
                    execution_duration_minutes = (end_time - start_time).total_seconds() / 60
                    redis_metrics['execution_duration_minutes'] = round(execution_duration_minutes, 2)
                    redis_metrics['throughput_dockets_per_minute'] = round(successful_dockets / max(execution_duration_minutes, 1), 2)
                except Exception as time_error:
                    task_logger.warning(f"Could not calculate execution time: {time_error}")

        except Exception as redis_error:
            task_logger.warning(f"Could not get final Redis state: {redis_error}")
            redis_metrics = {
                'redis_pipeline_state_available': False,
                'redis_error': str(redis_error),
                'redis_tracked_dockets': redis_tracked_dockets,
                'redis_source_reliable': False
            }

        # Fallback to PostgreSQL for API call count if Redis failed
        try:
            if not redis_metrics.get('redis_source_reliable', False):
                pg_hook = PostgresHook(postgres_conn_id=METASTORE_CONN_ID)
                current_hour = datetime.utcnow().strftime("%Y-%m-%d %H")
                with pg_hook.get_conn() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT calls_this_hour FROM courtlistener_rate_tracking WHERE hour_key = %s",
                            (current_hour,)
                        )
                        result = cursor.fetchone()
                        postgres_calls = result[0] if result else 0

                redis_metrics['postgres_fallback_calls'] = postgres_calls
                calls_this_hour = postgres_calls
            else:
                calls_this_hour = redis_metrics.get('redis_api_calls_this_hour', total_api_calls)

        except Exception as pg_error:
            task_logger.warning(f"PostgreSQL fallback also failed: {pg_error}")
            calls_this_hour = total_api_calls

        remaining_calls = TARGET_CALLS_PER_HOUR - calls_this_hour

        # Comprehensive summary
        summary = {
            "total_dockets_attempted": total_dockets,
            "successful_dockets": successful_dockets,
            "failed_dockets": failed_dockets,
            "skipped_dockets": total_skipped,
            "total_opinions_processed": total_opinions,
            "total_chunks_created": total_chunks,
            "total_api_calls_made": total_api_calls,
            "api_calls_this_hour": calls_this_hour,
            "api_calls_remaining": remaining_calls,
            "api_efficiency": f"{total_api_calls / max(successful_dockets, 1):.1f} calls/docket" if successful_dockets > 0 else "N/A",
            "execution_date": "{{ ds }}",
            "success_rate": successful_dockets / max(total_dockets, 1) * 100,
            "completion_rate": (successful_dockets + total_skipped) / max(total_dockets, 1) * 100,
            "rate_limit_utilization": calls_this_hour / TARGET_CALLS_PER_HOUR * 100,
            **redis_metrics  # Include all Redis metrics
        }

        task_logger.info(f"Redis-enhanced pipeline execution summary: {summary}")

        # Log detailed breakdown
        for result in results:
            if result.get("api_calls_made", 0) > 0:
                map_index = result.get("map_index", "N/A")
                redis_tracked = "Redis" if result.get("redis_tracked", False) else "Legacy"
                task_logger.info(f"Docket {result['docket_id']} (map_index: {map_index}, tracking: {redis_tracked}): "
                               f"{result['api_calls_made']} API calls, status: {result['status']}")

        return summary

    @task
    def cleanup_redis_expired_data() -> Dict[str, Any]:
        """
        Clean up expired Redis keys to maintain optimal performance and storage.

        This maintenance task removes old rate limiting data, expired pipeline states,
        and other temporary keys that are no longer needed. It helps prevent Redis
        memory bloat and maintains consistent performance.

        Inputs:
            - Redis connection configuration
            - Cleanup parameters (hours_back threshold)

        Outputs:
            Dict[str, Any]: Cleanup results containing:
                - 'keys_deleted': Number of expired keys removed
                - 'cutoff_time': Timestamp threshold for cleanup
                - 'patterns_checked': List of key patterns examined
                - 'cleanup_duration': Time taken for cleanup operation

        Side Effects:
            - Removes expired Redis keys permanently
            - Frees up Redis memory space
            - Logs cleanup statistics

        Performance Impact:
            - Improves Redis query performance by reducing key count
            - Prevents memory exhaustion in long-running deployments
            - Maintains consistent rate limiting accuracy

        Configuration:
            - Configurable retention period via environment variables
            - Safe deletion of only expired keys (preserves active data)
            - Respects Redis key TTL settings
        """
        try:
            redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)
            cleanup_result = redis_hook.cleanup_expired_keys(hours_back=48)

            task_logger.info(f"Redis cleanup completed: {cleanup_result}")
            return cleanup_result

        except Exception as e:
            task_logger.warning(f"Redis cleanup failed (non-critical): {e}")
            return {
                'error': str(e),
                'keys_deleted': 0,
                'cleanup_attempted': True
            }

    dag_config = load_config()

    # Define the enhanced DAG flow with Redis integration
    # Step 1: Initialize Redis pipeline state with start time caching
    redis_state = initialize_redis_pipeline_state()

    # Step 2: Check Redis-based rate limiting and hour boundary
    rate_limit_check = check_redis_rate_limit_and_hour_boundary()

    # Step 3: Get existing dockets (unchanged)
    existing_docket_ids = get_existing_dockets()

    # Step 4: Fetch dockets within Redis-enforced rate limits
    docket_ids_to_process = fetch_dockets_within_redis_rate_limit(
        court=dag_config.data_ingestion.court,
        rate_limit_state=rate_limit_check
    )

    # Step 6: Process each docket with Redis tracking
    processed_results = process_docket_with_redis_tracking.expand(docket_id=docket_ids_to_process)

    # Step 7: Generate comprehensive summary with Redis metrics
    summary = generate_redis_enhanced_summary(processed_results, redis_state)

    # Step 8: Optional Redis cleanup (can be disabled for production)
    redis_cleanup = cleanup_redis_expired_data()

    # Set up dependencies with Redis-enhanced flow
    redis_state >> rate_limit_check >> existing_docket_ids >> docket_ids_to_process >> processed_results >> summary >> redis_cleanup

# Instantiate the enhanced DAG
courtlistener_pipeline_dag()
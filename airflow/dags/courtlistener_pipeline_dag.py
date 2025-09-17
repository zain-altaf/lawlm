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

import json
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException, AirflowFailException
from airflow.hooks.postgres_hook import PostgresHook

# Add the project root to Python path for imports
sys.path.append('/root/lawlm')
sys.path.append('/root/lawlm/airflow')

# Import custom Redis hook
from hooks.redis_rate_limit_hook import RedisRateLimitHook

# Configuration constants
API_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CALL_LIMIT_PER_HOUR = 5000
TARGET_CALLS_PER_HOUR = 4900
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
    tags=['legal', 'etl', 'vector_search', 'redis', 'enhanced'],
    description='Enhanced CourtListener pipeline with Redis-based rate limiting and state management'
)
def courtlistener_pipeline_dag():
    """
    Enhanced Airflow DAG with Redis-based caching for robust API rate limiting
    and persistent state management across pipeline executions.
    """

    from config import load_config
    from vector_processor import EnhancedVectorProcessor
    from main import LegalDocumentPipeline

    class VectorProcessorPool:
        """Singleton pattern for vector processor with connection pooling and resource management."""
        _instance = None
        _processor = None
        _embedding_model = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def get_processor(self, model_name: str, collection_name: str) -> EnhancedVectorProcessor:
            """Get or create a shared vector processor instance."""
            if (self._processor is None or
                self._embedding_model != model_name or
                self._processor.collection_name != collection_name):

                task_logger.info(f"Creating new VectorProcessor for model: {model_name}, collection: {collection_name}")

                if self._processor is not None:
                    self.cleanup()

                self._processor = EnhancedVectorProcessor(
                    model_name=model_name,
                    collection_name=collection_name
                )
                self._embedding_model = model_name

            return self._processor

        def cleanup(self):
            """Cleanup resources when done."""
            if self._processor:
                task_logger.info("Cleaning up vector processor resources")
                self._processor = None
                self._embedding_model = None

    vector_processor_pool = VectorProcessorPool()

    def classify_and_handle_error(error: Exception, context: str, docket_id: int = None) -> None:
        """Enhanced error classification with Redis state awareness."""
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

    def make_api_call_with_backoff(url: str, headers: Dict[str, str], description: str, max_retries: int = 3):
        """Enhanced API call with Redis rate limit awareness."""
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
        Initialize Redis-based pipeline state with empty API call counter and start time caching.
        This task runs immediately when the DAG starts and sets up the distributed state.
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
            state = redis_hook.initialize_pipeline_state(dag_run_id)

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
                    "current_hour": datetime.now().strftime("%Y-%m-%d %H"),
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
        Check current API usage against cached amounts in Redis and verify hour boundary.
        Uses Redis for high-performance distributed rate limit enforcement.
        """
        try:
            redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)

            # Get current rate limit status from Redis
            rate_status = redis_hook.get_current_rate_limit_status()

            # Check for hour boundary crossing
            boundary_check = redis_hook.check_hour_boundary_and_reset()

            # Determine if pipeline can proceed
            can_proceed = (
                rate_status['can_proceed'] and
                rate_status['api_calls_remaining'] >= 50  # Safety buffer
            )

            result = {
                'current_hour': rate_status['current_hour'],
                'api_calls_used': rate_status['api_calls_used'],
                'api_calls_remaining': rate_status['api_calls_remaining'],
                'utilization_percent': rate_status['utilization_percent'],
                'can_proceed': can_proceed,
                'hour_boundary_crossed': boundary_check.get('hour_reset_detected', False),
                'previous_hour_calls': boundary_check.get('previous_count', 0),
                'redis_source': True
            }

            if boundary_check.get('hour_reset_detected'):
                task_logger.info(f"Hour boundary detected: {boundary_check['previous_hour']} -> {boundary_check['current_hour']}")
                task_logger.info(f"Previous hour usage: {boundary_check['previous_count']} calls")

            task_logger.info(f"Redis rate limit check: {rate_status['api_calls_used']}/{rate_status['limit']} calls used ({rate_status['utilization_percent']:.1f}%)")

            return result

        except Exception as e:
            task_logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to PostgreSQL method
            task_logger.warning("Falling back to PostgreSQL rate limit check")

            try:
                pg_hook = PostgresHook(postgres_conn_id=METASTORE_CONN_ID)
                current_hour = datetime.now().strftime("%Y-%m-%d %H")

                with pg_hook.get_conn() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT calls_this_hour FROM courtlistener_rate_tracking WHERE hour_key = %s",
                            (current_hour,)
                        )
                        result = cursor.fetchone()
                        current_calls = result[0] if result else 0

                remaining_calls = TARGET_CALLS_PER_HOUR - current_calls
                can_proceed = remaining_calls >= 50

                return {
                    'current_hour': current_hour,
                    'api_calls_used': current_calls,
                    'api_calls_remaining': remaining_calls,
                    'utilization_percent': (current_calls / CALL_LIMIT_PER_HOUR) * 100,
                    'can_proceed': can_proceed,
                    'redis_source': False,
                    'fallback_used': True,
                    'redis_error': str(e)
                }

            except Exception as fallback_error:
                task_logger.error(f"Both Redis and PostgreSQL rate limit checks failed: {fallback_error}")
                # Conservative fallback
                return {
                    'current_hour': datetime.now().strftime("%Y-%m-%d %H"),
                    'api_calls_used': 4950,  # Conservative assumption
                    'api_calls_remaining': 50,
                    'utilization_percent': 99.0,
                    'can_proceed': False,
                    'redis_source': False,
                    'fallback_used': True,
                    'error': str(e),
                    'fallback_error': str(fallback_error)
                }

    @task
    def get_existing_dockets() -> List[str]:
        """Get a list of existing docket IDs from Qdrant (unchanged from original)."""
        config = load_config()
        processor = vector_processor_pool.get_processor(
            model_name=config.vector_processing.embedding_model,
            collection_name=config.vector_processing.collection_name_vector
        )
        return list(processor.get_existing_docket_ids())

    @task
    def remove_items_from_collection(docket_ids_to_remove: List[int] = None, remove_all: bool = False) -> Dict[str, Any]:
        """Remove specific docket items or all items from the Qdrant collection (unchanged from original)."""
        from config import load_config
        from vector_processor import EnhancedVectorProcessor

        task_logger.info("Starting collection cleanup task")

        config = load_config()
        processor = vector_processor_pool.get_processor(
            model_name=config.vector_processing.embedding_model,
            collection_name=config.vector_processing.collection_name_vector
        )

        try:
            initial_dockets = processor.get_existing_docket_ids()
            initial_count = len(initial_dockets)
            task_logger.info(f"Collection initially contains {initial_count} dockets")

            removed_count = 0
            removed_dockets = []

            if remove_all:
                task_logger.info("Removing ALL items from collection")
                client = processor._get_qdrant_client()

                try:
                    from qdrant_client.models import FilterSelector, Filter, VectorParams, Distance
                    import qdrant_client.models as models

                    delete_result = client.delete(
                        collection_name=config.vector_processing.collection_name_vector,
                        points_selector=FilterSelector(filter=Filter())
                    )

                    collection_info = client.get_collection(config.vector_processing.collection_name_vector)
                    remaining_points = collection_info.points_count

                    if remaining_points > 0:
                        task_logger.warning(f"Still {remaining_points} points remaining. Using collection recreation method.")

                        client.delete_collection(config.vector_processing.collection_name_vector)
                        client.create_collection(
                            collection_name=config.vector_processing.collection_name_vector,
                            vectors_config={
                                'bge-small': VectorParams(size=384, distance=Distance.COSINE),
                            },
                            sparse_vectors_config={
                                'bm25': models.SparseVectorParams(modifier=models.Modifier.IDF),
                            },
                        )

                    removed_count = initial_count
                    removed_dockets = list(initial_dockets)

                except Exception as delete_error:
                    task_logger.error(f"Error during deletion: {delete_error}")
                    raise delete_error

            elif docket_ids_to_remove:
                for docket_id in docket_ids_to_remove:
                    if docket_id in initial_dockets:
                        success = processor.remove_docket_from_collection(docket_id)
                        if success:
                            removed_count += 1
                            removed_dockets.append(docket_id)

            final_dockets = processor.get_existing_docket_ids()
            final_count = len(final_dockets)

            result = {
                "initial_docket_count": initial_count,
                "final_docket_count": final_count,
                "dockets_removed": removed_count,
                "removed_docket_ids": removed_dockets,
                "removal_type": "all" if remove_all else ("specific" if docket_ids_to_remove else "none"),
                "success": True,
                "message": f"Successfully removed {removed_count} dockets. Collection size: {initial_count} → {final_count}"
            }

            return result

        except Exception as e:
            error_message = f"Failed to remove items from collection: {str(e)}"
            task_logger.error(error_message)
            classify_and_handle_error(e, "remove_items_from_collection")

    @task
    def fetch_dockets_within_redis_rate_limit(court: str, rate_limit_state: Dict[str, Any]) -> List[int]:
        """Enhanced docket fetching with Redis-based rate limit awareness."""
        remaining_calls = rate_limit_state["api_calls_remaining"]

        if not rate_limit_state["can_proceed"]:
            task_logger.warning(f"Insufficient API calls remaining ({remaining_calls}). Skipping this run.")
            return []

        # More conservative estimation with Redis tracking
        estimated_calls_per_docket = 30  # Slightly higher due to Redis overhead tracking
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
        return docket_ids

    @task
    def process_docket_with_redis_tracking(docket_id: int) -> Dict[str, Any]:
        """
        Enhanced docket processing with Redis-based API call tracking and atomic state updates.
        Immediately caches task start time and maintains distributed state consistency.
        """
        import requests
        from main import LegalDocumentPipeline
        from config import load_config
        from airflow.operators.python import get_current_context

        context = get_current_context()
        map_index = context.get('task_instance').map_index
        dag_run_id = context['dag_run'].run_id
        task_id = f"process_docket_with_redis_tracking_{map_index}"

        task_logger.info(f"Starting Redis-tracked processing of docket ID: {docket_id} "
                        f"(map_index: {map_index}, dag_run: {dag_run_id})")

        # Initialize Redis hook
        redis_hook = RedisRateLimitHook(redis_conn_id=REDIS_CONN_ID)

        # Cache task start time immediately
        start_time = redis_hook.cache_task_start_time(dag_run_id, task_id)
        task_logger.info(f"Task start time cached in Redis: {start_time}")

        api_calls_made = 0

        try:
            # Check current rate limit state from Redis
            rate_status = redis_hook.get_current_rate_limit_status()

            if not rate_status['can_proceed']:
                task_logger.warning(f"Redis rate limit check failed for docket {docket_id}: "
                                  f"{rate_status['api_calls_used']}/{rate_status['limit']} calls used")
                return {
                    "docket_id": docket_id,
                    "api_calls_made": 0,
                    "map_index": map_index,
                    "status": "skipped_rate_limit_redis",
                    "message": f"Redis rate limit exceeded: {rate_status['api_calls_used']}/{rate_status['limit']}"
                }

            # Safety check: ensure we have minimum calls for this docket
            if rate_status['api_calls_remaining'] < 25:
                task_logger.warning(f"Insufficient remaining calls ({rate_status['api_calls_remaining']}) for docket {docket_id}")
                return {
                    "docket_id": docket_id,
                    "api_calls_made": 0,
                    "map_index": map_index,
                    "status": "skipped_rate_limit",
                    "message": f"Insufficient API calls remaining: {rate_status['api_calls_remaining']}"
                }

            # Load config and override for single docket processing
            config = load_config()
            config.data_ingestion.num_dockets = 1

            # Initialize the pipeline
            pipeline = LegalDocumentPipeline(config)

            # Store original fetch method
            original_fetch = pipeline._fetch_all_dockets_paginated

            def redis_tracked_fetch(court: str, num_dockets: int, existing_dockets: set):
                """Fetch with Redis API call tracking."""
                nonlocal api_calls_made

                vp = vector_processor_pool.get_processor(
                    model_name=config.vector_processing.embedding_model,
                    collection_name=config.vector_processing.collection_name_vector
                )
                current_existing_dockets = vp.get_existing_docket_ids()

                if docket_id in current_existing_dockets or docket_id in existing_dockets:
                    task_logger.info(f"Docket {docket_id} already exists, skipping with Redis state update")
                    # Update Redis state even for skipped dockets
                    redis_hook.update_docket_processing_state(
                        dag_run_id=dag_run_id,
                        docket_id=docket_id,
                        api_calls_made=0,
                        success=True
                    )
                    return []

                task_logger.info(f"Docket {docket_id} is new, fetching from API with Redis call tracking")
                headers = {"Authorization": f"Token {config.data_ingestion.api_key}"}

                # API Call 1: Fetch main docket
                url = f"{config.data_ingestion.api_base_url}/dockets/{docket_id}/"
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                api_calls_made += 1

                # Pre-check Redis rate limit after each API call
                increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR)
                if not increment_result['allowed']:
                    task_logger.error(f"Redis atomic increment rejected API call: "
                                    f"current={increment_result['new_count']}, limit={increment_result['limit']}")
                    raise Exception(f"Redis rate limit enforcement: {increment_result['new_count']}/{increment_result['limit']}")

                task_logger.info(f"Redis API call {api_calls_made}: Fetched docket {docket_id}, "
                                f"total calls this hour: {increment_result['new_count']}")

                docket = response.json()

                # API Calls 2+: Fetch associated opinions with Redis tracking
                if 'cluster' in docket and docket['cluster']:
                    cluster_url = docket['cluster']
                    if isinstance(cluster_url, str) and cluster_url.startswith('http'):
                        cluster_response = requests.get(cluster_url, headers=headers, timeout=30)
                        cluster_response.raise_for_status()
                        api_calls_made += 1

                        # Redis atomic increment for cluster call
                        increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR)
                        if not increment_result['allowed']:
                            raise Exception(f"Redis rate limit during cluster fetch: {increment_result['new_count']}/{increment_result['limit']}")

                        task_logger.info(f"Redis API call {api_calls_made}: Fetched cluster for docket {docket_id}, "
                                        f"total calls: {increment_result['new_count']}")

                        cluster_data = cluster_response.json()
                        if 'sub_opinions' in cluster_data:
                            for opinion_url in cluster_data['sub_opinions']:
                                if isinstance(opinion_url, str) and opinion_url.startswith('http'):
                                    # Check Redis rate limit before each opinion fetch
                                    current_status = redis_hook.get_current_rate_limit_status()
                                    if not current_status['can_proceed']:
                                        task_logger.warning(f"Redis rate limit reached during opinion fetching: "
                                                          f"{current_status['api_calls_used']}/{current_status['limit']}")
                                        break

                                    opinion_response = requests.get(opinion_url, headers=headers, timeout=30)
                                    opinion_response.raise_for_status()
                                    api_calls_made += 1

                                    # Redis atomic increment for opinion call
                                    increment_result = redis_hook.atomic_increment_api_calls(1, CALL_LIMIT_PER_HOUR)
                                    if not increment_result['allowed']:
                                        task_logger.warning(f"Redis rate limit reached during opinion {api_calls_made}: "
                                                          f"{increment_result['new_count']}/{increment_result['limit']}")
                                        break

                                    task_logger.info(f"Redis API call {api_calls_made}: Fetched opinion for docket {docket_id}, "
                                                    f"total calls: {increment_result['new_count']}")

                task_logger.info(f"Docket {docket_id} successfully fetched with {api_calls_made} Redis-tracked API calls")
                return [docket]

            # Replace fetch method with Redis-tracked version
            pipeline._fetch_all_dockets_paginated = redis_tracked_fetch

            # Run the pipeline
            result = pipeline.run_pipeline()

            # Restore original method
            pipeline._fetch_all_dockets_paginated = original_fetch

            # Update Redis state with final results
            redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=docket_id,
                api_calls_made=api_calls_made,
                success=True
            )

            task_logger.info(f"Redis state updated: docket {docket_id}, calls: {api_calls_made}, success: True")

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

            # Extract stats from result
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
                return {
                    "docket_id": docket_id,
                    "api_calls_made": api_calls_made,
                    "map_index": map_index,
                    "opinions": 0,
                    "chunks": 0,
                    "vectors": 0,
                    "status": "failed",
                    "error": str(e),
                    "redis_tracked": True
                }

    @task
    def generate_redis_enhanced_summary(results: List[Dict[str, Any]],
                                       redis_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline summary with Redis state information and performance metrics.
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
                current_hour = datetime.now().strftime("%Y-%m-%d %H")
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
        Optional cleanup task to remove expired Redis keys and maintain performance.
        This task can be run periodically to clean up old rate limiting data.
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

    # Step 4: Optional collection cleanup (unchanged but can be disabled)
    collection_cleanup = remove_items_from_collection(
        docket_ids_to_remove=None,
        remove_all=False  # Set to True to clear entire collection, False for normal operation
    )

    # Step 5: Fetch dockets within Redis-enforced rate limits
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
    redis_state >> rate_limit_check >> existing_docket_ids >> collection_cleanup >> docket_ids_to_process >> processed_results >> summary >> redis_cleanup

# Instantiate the enhanced DAG
courtlistener_pipeline_dag()
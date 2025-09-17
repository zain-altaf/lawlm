"""
Airflow DAG for Legal Document Processing Pipeline with Rate Limiting

This DAG orchestrates the complete legal document processing pipeline:
1. Rate limit enforcement (5000 calls/hour to CourtListener API)
2. Data ingestion from CourtListener API
3. Document chunking and processing
4. Vector embedding generation
5. Storage in Qdrant vector database
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

from airflow.decorators import dag, task
from airflow.models import Variable

# Add the project root to Python path for imports
sys.path.append('/root/lawlm')

# Configuration constants
API_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CALL_LIMIT_PER_HOUR = 5000
TARGET_CALLS_PER_HOUR = 4950
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
    schedule='@daily',
    catchup=False,
    tags=['legal', 'etl', 'vector_search'],
)
def courtlistener_pipeline_dag():
    """
    Airflow DAG to orchestrate the legal document processing pipeline.
    """

    # Imports must be inside the DAG function for Airflow's deferred import mechanism
    from config import load_config
    from vector_processor import EnhancedVectorProcessor
    from main import LegalDocumentPipeline

    @task
    def check_rate_limit() -> bool:
        """
        Check to ensure the CourtListener API hourly rate limit is not exceeded.
        """
        # Get the variable for remaining API calls and update it
        try:
            rate_limit_info = json.loads(Variable.get("courtlistener_rate_limit", default_var="{}"))
        except (KeyError, json.JSONDecodeError):
            rate_limit_info = {}

        current_hour = datetime.now().strftime("%Y-%m-%d %H")

        last_checked_hour = rate_limit_info.get("last_checked_hour", "")
        calls_this_hour = rate_limit_info.get("calls_this_hour", 0)

        # Reset count for a new hour
        if current_hour != last_checked_hour:
            calls_this_hour = 0
            rate_limit_info["last_checked_hour"] = current_hour

        remaining_calls = TARGET_CALLS_PER_HOUR - calls_this_hour
        task_logger.info(f"Checking API Rate Limit. Remaining calls this hour: {remaining_calls}")

        # Update the variable
        Variable.set("courtlistener_rate_limit", json.dumps(rate_limit_info))

        if remaining_calls < SAFE_CONCURRENCY_LIMIT:
            task_logger.warning("API call limit is approaching. Skipping this run.")
            return False

        task_logger.info("Rate limit check passed. Proceeding with pipeline.")
        return True


    @task
    def get_existing_dockets() -> List[str]:
        """Get a list of existing docket IDs from Qdrant."""
        config = load_config()
        processor = EnhancedVectorProcessor(collection_name=config.vector_processing.collection_name_vector)
        return list(processor.get_existing_docket_ids())

    @task
    def fetch_new_dockets_for_processing(court: str, batch_size: int) -> List[int]:
        """Fetches a batch of new dockets that haven't been processed yet."""
        task_logger.info(f"Fetching {batch_size} new dockets from {court}")

        config = load_config()
        pipeline = LegalDocumentPipeline(config)

        # Get existing dockets for deduplication
        existing_dockets = set()
        try:
            vector_processor = EnhancedVectorProcessor(
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
            num_dockets=batch_size,
            existing_dockets=existing_dockets
        )

        # Extract docket IDs for processing
        docket_ids = [docket['id'] for docket in new_dockets]
        task_logger.info(f"Successfully fetched {len(docket_ids)} new docket IDs for processing")
        return docket_ids

    @task
    def process_single_docket_with_main_pipeline(docket_id: int) -> Dict[str, Any]:
        """
        Process a single docket using the existing main.py pipeline logic.
        """
        import requests
        from main import LegalDocumentPipeline
        from config import load_config

        task_logger.info(f"Processing docket ID: {docket_id}")

        try:
            # Load config and override for single docket processing
            config = load_config()
            config.data_ingestion.num_dockets = 1

            # Initialize the pipeline
            pipeline = LegalDocumentPipeline(config)

            # Store original fetch method
            original_fetch = pipeline._fetch_all_dockets_paginated

            def mock_fetch_specific_docket(court: str, num_dockets: int, existing_dockets: set):
                """Mock fetch to return only the specific docket we want if it doesn't already exist."""
                from vector_processor import EnhancedVectorProcessor
                vp = EnhancedVectorProcessor(
                    model_name=config.vector_processing.embedding_model,
                    collection_name=config.vector_processing.collection_name_vector
                )
                current_existing_dockets = vp.get_existing_docket_ids()
                task_logger.info(f"Fresh check: Found {len(current_existing_dockets)} existing dockets")

                if docket_id in current_existing_dockets or docket_id in existing_dockets:
                    task_logger.info(f"Docket {docket_id} already exists, skipping")
                    return []

                task_logger.info(f"Docket {docket_id} is new, fetching from API")
                headers = {"Authorization": f"Token {config.data_ingestion.api_key}"}
                url = f"{config.data_ingestion.api_base_url}/dockets/{docket_id}/"
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                docket = response.json()
                task_logger.info(f"Docket {docket_id} successfully fetched")
                return [docket]

            # Temporarily replace the fetch method
            pipeline._fetch_all_dockets_paginated = mock_fetch_specific_docket

            # Run the pipeline
            result = pipeline.run_pipeline()

            # Restore original method
            pipeline._fetch_all_dockets_paginated = original_fetch

            task_logger.info(f"Pipeline result for docket {docket_id}: {result}")

            if result.get('status') == 'up_to_date':
                return {
                    "docket_id": docket_id,
                    "opinions": 0,
                    "chunks": 0,
                    "vectors": 0,
                    "status": "skipped_duplicate",
                    "message": "Docket already exists in collection"
                }

            # Extract stats from result
            stats = result.get('stats', {})
            return {
                "docket_id": docket_id,
                "opinions": stats.get('opinions_processed', 0),
                "chunks": stats.get('chunks_created', 0),
                "vectors": stats.get('vectors_uploaded', 0),
                "status": "success" if result.get('status') == 'completed' else result.get('status', 'unknown'),
                "message": f"Successfully processed docket {docket_id}"
            }

        except Exception as e:
            task_logger.error(f"Error processing docket {docket_id}: {e}")
            return {
                "docket_id": docket_id,
                "opinions": 0,
                "chunks": 0,
                "vectors": 0,
                "status": "failed",
                "error": str(e)
            }

    @task
    def pipeline_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates a summary of the pipeline run."""
        task_logger.info("Generating pipeline summary")
        
        total_dockets = len(results)
        successful_dockets = sum(1 for r in results if r.get("status") == "success")
        failed_dockets = sum(1 for r in results if r.get("status") == "failed")
        total_skipped = sum(1 for r in results if 'skipped' in r.get("status", ""))
        total_opinions = sum(r.get("opinions", 0) for r in results if r.get("status") == "success")
        total_chunks = sum(r.get("chunks", 0) for r in results if r.get("status") == "success")
        
        summary = {
            "total_dockets_attempted": total_dockets,
            "successful_dockets": successful_dockets,
            "failed_dockets": failed_dockets,
            "skipped_dockets": total_skipped,
            "total_opinions_processed": total_opinions,
            "total_chunks_created": total_chunks,
            "execution_date": "{{ ds }}",
            "success_rate": successful_dockets / max(total_dockets, 1) * 100,
            "completion_rate": (successful_dockets + total_skipped) / max(total_dockets, 1) * 100
        }

        task_logger.info(f"Pipeline execution summary: {summary}")
        return summary
    
    # Get configuration (load directly since we need static values for DAG definition)
    dag_config = load_config()

    # Define the DAG flow
    rate_limit_check = check_rate_limit()
    existing_docket_ids = get_existing_dockets()

    # Fetch new dockets for processing (with built-in deduplication)
    new_docket_ids = fetch_new_dockets_for_processing(
        court=dag_config.data_ingestion.court,
        batch_size=dag_config.airflow.batch_size
    )

    # Process each new docket with dynamic task mapping
    processed_results = process_single_docket_with_main_pipeline.expand(docket_id=new_docket_ids)
    
    # Generate final summary of the pipeline run
    summary = pipeline_summary(processed_results)

    # Set up dependencies
    rate_limit_check >> existing_docket_ids >> new_docket_ids >> processed_results >> summary

# Instantiate the DAG
courtlistener_pipeline_dag()
# A Framework for Robust Airflow Orchestration: Best Practices for API Rate Limiting and Observability

## Executive Summary

### The Challenge
The orchestration of data pipelines often involves consuming data from external APIs, which typically impose strict rate limits to protect their infrastructure from overload. This report addresses the specific challenge of building an Apache Airflow data pipeline that interacts with the CourtListener API, which enforces a cap of 5,000 calls per hour. The primary objective is to develop a robust and resilient workflow that not only successfully extracts data but also consistently operates within the defined API constraints. The project necessitates a solution that can accurately track API call volume, proactively prevent errors, and handle inevitable failures with grace, while providing clear, real-time and historical insight into pipeline health and performance.

### The Solution Framework
The proposed solution is a comprehensive, multi-layered framework built on three foundational pillars: resilient DAG design, a strategic approach to rate limit management, and a robust observability stack.  

1. **Resilient DAG Design** – Structuring DAGs and tasks to be idempotent and atomic.  
2. **Strategic Rate Limit Management** – Using Airflow Pools and exponential backoff for transient API errors.  
3. **Holistic Observability Stack** – Structured logging, custom metrics, and persistent counters for tracking.  

---

## Foundational Principles of Resilient DAG Design

### The Pillars of Reliability: Idempotency and Atomicity
- **Idempotency**: Rerunning a DAG with the same inputs should yield identical results.  
- **Atomicity**: Tasks should perform a single, indivisible operation.  

Examples:  
- Use `data_interval_start` instead of `datetime.now()` for deterministic time windows.  
- Use **UPSERT** instead of INSERT to avoid duplicates on retries.  

### Modular and Maintainable Workflows
- Treat the DAG file as configuration only.  
- Business logic should live in external Python modules.  
- Use **TaskFlow API** and **TaskGroups** for clarity and modularity.  
- Avoid heavy operations in top-level DAG scope (they run on every scheduler parse).  

### State Management and Incremental Loading
- Avoid full reprocessing on each run.  
- Use incremental loading with `date_modified` ranges or sequential IDs.  

---

## Conquering the API Rate Limit Challenge

### Understanding API Rate Limiting
CourtListener API enforces **5,000 calls/hour** using a fixed-window limit.  

### Implementing a Global Concurrency Governor with Airflow Pools
- Use **Pools** to control task concurrency across DAGs.  
- Example: `courtlistener_api_pool` with a safe slot configuration to enforce concurrency.  

### Dynamic Backoff and Retries for Transient Failures
- Use Airflow’s retries for general failures.  
- Use the **backoff** library to handle API-specific errors (e.g., 429 with `Retry-After` header).  

### The Time-Aware Sensor Pattern (Advanced)
- Custom sensor checks API call counters in a DB.  
- Enforces true hourly rate limits by rescheduling tasks if the limit is reached.  

---

## Building a Holistic Observability Stack

### Strategic Logging for Actionable Insight
- Use **structured logging** (JSON format) with key metadata: `timestamp`, `endpoint`, `status_code`, `duration`.  

### Custom Metrics for Real-Time Monitoring
- Use **StatsD** counters like `airflow.api.courtlistener.calls`.  
- Build dashboards in Datadog/Prometheus.  

### Persistent API Call Tracking in the Airflow Metastore
- Create a dedicated table to store hourly counters.  
- Use UPSERT logic to ensure idempotency.  

#### Comparison of Tracking Methods

| Criterion       | Airflow Variables | Dedicated DB Table (Metastore) | Custom StatsD Metrics |
|-----------------|------------------|--------------------------------|-----------------------|
| Reliability     | Good (encrypted) | Excellent (DB managed)         | Fair (fire-and-forget)|
| Queryability    | Poor             | Excellent (SQL support)        | Poor                  |
| Real-Time View  | Poor             | Poor                           | Excellent             |
| Complexity      | Low              | Moderate                       | High                  |
| Performance     | Scheduler impact | Low                            | Very low              |
| Use Case        | Simple counters  | Auditable, historical records  | Real-time dashboards  |

---

## Advanced Error Handling and Proactive Alerting

### Granular Failure Management with Airflow Exceptions
- Use `AirflowFailException` for **non-transient** errors (e.g., 401 Unauthorized).  
- Allow Airflow’s retries for **transient** server-side errors (500-series).  

### Automated Notifications with Callbacks and Notifiers
- Use `on_failure_callback` to send Slack/Email alerts.  
- Alerts should include DAG/task info, run ID, and log URL.  

---

## A Reference DAG for the CourtListener API Pipeline

The following DAG blueprint demonstrates best practices:

```python
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.slack.hooks.slack import SlackHook
from airflow.exceptions import AirflowFailException
from airflow.metrics import StatsD
from airflow.sensors.base import PokeReturnValue
import backoff

# --- Configuration & External Dependencies ---
API_ENDPOINT = "https://www.courtlistener.com/api/rest/v1/search/"
API_CONN_ID = "courtlistener_api_conn"
CALL_LIMIT_PER_HOUR = 5000
METASTORE_CONN_ID = "postgres_default"
ALERT_SLACK_CONN_ID = "slack_default"
API_POOL_NAME = "courtlistener_api_pool"

task_logger = logging.getLogger("airflow.task")

# --- Custom Functions & Callbacks ---
def alert_on_failure(context: Dict[str, Any]):
    slack_hook = SlackHook(slack_conn_id=ALERT_SLACK_CONN_ID)
    message = (
        f"🚨 Task Failure\n"
        f"DAG: {context['task_instance'].dag_id}\n"
        f"Task: {context['task_instance'].task_id}\n"
        f"Log URL: {context['task_instance'].log_url}"
    )
    slack_hook.call(api_method="chat.postMessage", json={"channel": "#airflow-alerts", "text": message})

@task
def upsert_api_call_count():
    pg_hook = PostgresHook(postgres_conn_id=METASTORE_CONN_ID)
    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    upsert_sql = """
        INSERT INTO api_call_metrics (api_name, timestamp_hour, call_count)
        VALUES ('courtlistener', %s, 1)
        ON CONFLICT (api_name, timestamp_hour) DO UPDATE
        SET call_count = api_call_metrics.call_count + 1;
    """
    pg_hook.run(upsert_sql, parameters=[current_hour])

@task.sensor(poke_interval=300, timeout=3540, mode="reschedule", pool=API_POOL_NAME)
def enforce_rate_limit():
    pg_hook = PostgresHook(postgres_conn_id=METASTORE_CONN_ID)
    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    result = pg_hook.get_first("""
        SELECT COALESCE(call_count, 0) FROM api_call_metrics
        WHERE api_name = 'courtlistener' AND timestamp_hour = %s;
    """, parameters=[current_hour])
    call_count = result or 0
    return PokeReturnValue(is_done=(call_count < CALL_LIMIT_PER_HOUR))

@task(pool=API_POOL_NAME)
@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=8)
def fetch_and_process_data(ti) -> Dict:
    params = {"date_modified__gte": ti.data_interval_start.isoformat(),
              "date_modified__lt": ti.data_interval_end.isoformat()}
    response = requests.get(API_ENDPOINT, params=params)
    response.raise_for_status()
    StatsD.incr(f"api_calls.{API_POOL_NAME}")
    return response.json()

@task
def upsert_to_database(data: Dict):
    if not data or not data.get("results"):
        return
    pg_hook = PostgresHook(postgres_conn_id=METASTORE_CONN_ID)
    for record in data["results"]:
        upsert_sql = """
            INSERT INTO courtlistener_data (id, data, date_modified)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET data = EXCLUDED.data, date_modified = EXCLUDED.date_modified;
        """
        pg_hook.run(upsert_sql, parameters=[record["id"], record, record["date_modified"]])

@dag(
    dag_id="courtlistener_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(hours=1),
    catchup=False,
    default_args={"retries": 3, "on_failure_callback": alert_on_failure}
)
def courtlistener_dag():
    rate_limit_check = enforce_rate_limit()
    call_and_process = fetch_and_process_data()
    store_call_count = upsert_api_call_count()
    load_data = upsert_to_database(call_and_process)
    rate_limit_check >> call_and_process >> store_call_count >> load_data

courtlistener_dag()

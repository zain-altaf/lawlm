# Airflow Pool Management Commands

## Pool Operations

### List All Pools
```bash
export PATH="/root/anaconda3/envs/lawlm/bin:$PATH" && export AIRFLOW_HOME="/root/lawlm/airflow"
airflow pools list
```

### Create/Update Pool
```bash
# Create the CourtListener API pool with 10 slots
airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting"

# Adjust pool size if needed (5-50 range recommended)
airflow pools set courtlistener_api_pool 15 "CourtListener API pool for rate limiting"
```

### Delete Pool
```bash
airflow pools delete courtlistener_api_pool
```

## DAG Management

### Check DAG Status
```bash
airflow dags list | grep courtlistener
airflow dags unpause courtlistener_legal_pipeline
```

### Trigger DAG Manually
```bash
airflow dags trigger courtlistener_legal_pipeline
```

### Check Task States
```bash
# Get the latest run ID first
airflow dags list-runs courtlistener_legal_pipeline

# Check task states for a specific run
airflow tasks states-for-dag-run courtlistener_legal_pipeline EXECUTION_DATE
```

## Scheduler Management

### Check Scheduler Status
```bash
pgrep -f airflow
```

### Restart Scheduler
```bash
# Stop scheduler
pkill -f "airflow scheduler"

# Start scheduler in daemon mode
airflow scheduler --daemon
```

### Check Scheduler Logs
```bash
tail -f /root/lawlm/airflow/airflow-scheduler.log
```

## Pool Size Recommendations

- **Conservative (5 slots)**: For careful API rate limiting during development
- **Balanced (10 slots)**: Current setting, good for testing with safety margin
- **Aggressive (20-50 slots)**: For maximum throughput while staying under 5000/hour limit

The pool size controls how many tasks can run concurrently. With the CourtListener API limit of 5000 calls/hour, calculate based on:
- Each docket requires ~2 API calls (docket + opinions)
- Target ~4950 calls/hour for safety margin
- Pool size should align with your processing capacity

## Troubleshooting

### If Pool Warnings Return
1. Check pool exists: `airflow pools list`
2. Recreate if missing: `airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting"`
3. Restart scheduler: `pkill -f "airflow scheduler" && airflow scheduler --daemon`

### If Tasks Don't Schedule
1. Check DAG is unpaused: `airflow dags unpause courtlistener_legal_pipeline`
2. Check scheduler is running: `pgrep -f airflow`
3. Check pool slots available: `airflow pools list`
4. Review scheduler logs: `tail -f /root/lawlm/airflow/airflow-scheduler.log`
# Redis-Enhanced Pipeline Setup Guide

## Quick Start

This guide provides step-by-step instructions for implementing Redis-based rate limiting and state management in the CourtListener pipeline.

## Prerequisites

- Existing CourtListener pipeline deployment
- Redis server (local or cloud)
- Updated requirements.txt dependencies installed

## Step 1: Install Dependencies

```bash
# Activate your conda environment
conda activate lawlm

# Install Redis dependencies
pip install redis==5.0.1 hiredis==2.2.3 apache-airflow-providers-redis==3.4.0
```

## Step 2: Redis Server Setup

### Option A: Local Redis (Development)
```bash
# Using Docker
docker run -d --name redis-courtlistener \
  -p 6379:6379 \
  redis:7-alpine

# Verify Redis is running
docker exec redis-courtlistener redis-cli ping
# Expected output: PONG
```

### Option B: Redis Cloud (Production)
1. Sign up for Redis Cloud (recommended) or AWS ElastiCache
2. Create a Redis instance with the following specifications:
   - Redis version: 7.x
   - Memory: 1GB minimum (2GB recommended)
   - Persistence: Enabled (RDB + AOF)
   - High Availability: Enabled for production

## Step 3: Configure Airflow Connection

### Via Airflow UI
1. Go to Admin → Connections
2. Click "Add a new record"
3. Fill in the details:
   ```
   Connection Id: redis_default
   Connection Type: Redis
   Host: localhost (or your Redis host)
   Port: 6379
   Schema: 0
   Password: (if required)
   Extra: {"ssl": false}  # Set to true for Redis Cloud
   ```

### Via Environment Variable
```bash
export AIRFLOW_CONN_REDIS_DEFAULT='redis://localhost:6379/0'

# For Redis with password:
export AIRFLOW_CONN_REDIS_DEFAULT='redis://:password@host:6379/0'

# For Redis Cloud with SSL:
export AIRFLOW_CONN_REDIS_DEFAULT='rediss://:password@host:port/0'
```

## Step 4: Test Redis Connectivity

```python
# Test script: test_redis_connection.py
from airflow.hooks.redis_hook import RedisHook

try:
    redis_hook = RedisHook(redis_conn_id='redis_default')
    client = redis_hook.get_conn()

    # Test basic operations
    client.set('test_key', 'test_value')
    value = client.get('test_key')
    client.delete('test_key')

    print(f"✅ Redis connection successful! Test value: {value}")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

## Step 5: Deploy Enhanced DAG

### Copy Enhanced DAG
```bash
# Copy the enhanced DAG to your Airflow DAGs folder
cp /root/lawlm/airflow/dags/courtlistener_pipeline_redis_enhanced.py \
   /path/to/your/airflow/dags/

# Copy the Redis hook
mkdir -p /path/to/your/airflow/hooks/
cp /root/lawlm/airflow/hooks/redis_rate_limit_hook.py \
   /path/to/your/airflow/hooks/
```

### Verify DAG Parsing
```bash
# Test DAG parsing
python -c "
import sys
sys.path.append('/path/to/your/airflow/dags')
from courtlistener_pipeline_redis_enhanced import courtlistener_pipeline_redis_enhanced
print('✅ Enhanced DAG parsed successfully')
"
```

## Step 6: Configuration Updates

### Update Environment Variables
```bash
# Add Redis configuration to your .env file
echo "
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false
" >> .env
```

### Test Configuration Loading
```python
# Test script: test_config.py
import sys
sys.path.append('/root/lawlm')
from config import load_config

config = load_config()
print(f"✅ Redis host: {config.redis.host}")
print(f"✅ Redis port: {config.redis.port}")
print(f"✅ Redis TTL: {config.redis.rate_limit_key_ttl_hours} hours")
```

## Step 7: Parallel Testing

### Run Both DAGs Simultaneously
1. Enable the original DAG: `courtlistener_pipeline`
2. Enable the enhanced DAG: `courtlistener_pipeline_redis_enhanced`
3. Trigger both DAGs manually
4. Compare execution results and performance

### Monitoring During Testing
```bash
# Monitor Redis operations
redis-cli monitor

# Check Redis keys
redis-cli keys "courtlistener:*"

# View pipeline state
redis-cli hgetall "courtlistener:pipeline:manual__2024-01-01T00:00:00+00:00"
```

## Step 8: Validation Checklist

### Functional Validation
- [ ] Enhanced DAG runs without errors
- [ ] API rate limiting respects 5000 calls/hour limit
- [ ] Pipeline state persists across task failures
- [ ] Fallback to PostgreSQL works when Redis is unavailable

### Performance Validation
- [ ] Redis operations complete within 5ms
- [ ] No significant overhead compared to original DAG
- [ ] Memory usage remains stable over multiple runs
- [ ] Connection pooling prevents resource exhaustion

### Data Validation
- [ ] API call counts match between Redis and PostgreSQL
- [ ] Docket processing results are identical
- [ ] Vector upload counts are consistent
- [ ] No data loss during Redis failures

## Step 9: Production Migration

### Phase 1: Shadow Mode
```bash
# Run enhanced DAG in shadow mode (no collection cleanup)
# Set remove_all=False in collection_cleanup task
# Monitor for 1 week to ensure stability
```

### Phase 2: Gradual Transition
```bash
# Gradually increase traffic to enhanced DAG
# Reduce frequency of original DAG
# Monitor error rates and performance metrics
```

### Phase 3: Full Migration
```bash
# Disable original DAG
# Remove PostgreSQL-only rate limiting code
# Enable Redis cleanup tasks
```

## Troubleshooting

### Common Issues

#### DAG Import Errors
```bash
# Symptom: ImportError for Redis modules
# Solution: Ensure all dependencies are installed
pip install redis==5.0.1 apache-airflow-providers-redis==3.4.0
```

#### Redis Connection Timeout
```bash
# Symptom: Connection timeout errors
# Solution: Check Redis server status and network connectivity
redis-cli -h your-host -p 6379 ping
```

#### Rate Limiting Not Working
```bash
# Symptom: API calls exceed 5000/hour
# Solution: Verify atomic operations are enabled
redis-cli eval "return redis.call('GET', 'test')" 0
```

#### State Persistence Issues
```bash
# Symptom: Pipeline state lost after restart
# Solution: Check Redis TTL configuration
redis-cli ttl "courtlistener:pipeline:your-dag-run-id"
```

### Debug Commands
```bash
# View all CourtListener keys
redis-cli keys "courtlistener:*"

# Check current API count
redis-cli get "courtlistener:counter:$(date +'%Y-%m-%d_%H')"

# View pipeline state
redis-cli hgetall "courtlistener:pipeline:manual__$(date -Iseconds)"

# Monitor Redis operations live
redis-cli monitor | grep courtlistener
```

### Performance Monitoring
```python
# Create monitoring script: monitor_performance.py
import time
import redis
from datetime import datetime

client = redis.Redis(host='localhost', port=6379, db=0)

def monitor_performance():
    while True:
        start_time = time.time()

        # Test basic operations
        client.set('perf_test', 'value')
        client.get('perf_test')
        client.delete('perf_test')

        elapsed = (time.time() - start_time) * 1000
        print(f"{datetime.now()}: Redis operation took {elapsed:.2f}ms")

        time.sleep(60)

if __name__ == "__main__":
    monitor_performance()
```

## Success Criteria

Your Redis enhancement is successful when:

1. **Reliability**: Pipeline survives Redis restarts and worker failures
2. **Performance**: No significant latency increase (< 10ms overhead)
3. **Accuracy**: 100% API rate limit compliance
4. **Observability**: Real-time visibility into pipeline state
5. **Scalability**: Supports multiple concurrent workers

## Next Steps

After successful deployment:

1. **Monitoring Setup**: Configure Grafana dashboards for Redis metrics
2. **Alerting**: Set up alerts for Redis connectivity and rate limit breaches
3. **Backup Strategy**: Implement Redis backup and recovery procedures
4. **Documentation**: Update operational runbooks with Redis procedures
5. **Training**: Train team on Redis troubleshooting and monitoring

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Redis logs and Airflow task logs
3. Test individual components in isolation
4. Verify configuration against the architecture documentation

## Additional Resources

- [Redis-Enhanced Architecture Documentation](redis_enhanced_architecture.md)
- [Original Airflow Design Principles](../context/airflow-design-principles.md)
- [CourtListener API Documentation](https://www.courtlistener.com/api/)
- [Apache Airflow Redis Provider Documentation](https://airflow.apache.org/docs/apache-airflow-providers-redis/)
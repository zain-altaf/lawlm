#!/bin/bash

# Airflow Management Script
# Supports three modes:
#   Default (no flags): Start/ensure services are running
#   --reset: Complete database wipe and reinitialization
#   --stop: Stop all Airflow services and Redis

# Parse command line arguments
RESET_MODE=false
STOP_MODE=false
if [[ "$1" == "--reset" ]]; then
    RESET_MODE=true
    echo "🔄 Starting Airflow complete database reset..."
elif [[ "$1" == "--stop" ]]; then
    STOP_MODE=true
    echo "🛑 Stopping all Airflow services and Redis..."
else
    echo "🚀 Starting/Ensuring Airflow services are running..."
fi

# Set environment variables - Critical for proper operation
export PATH="/root/anaconda3/envs/lawlm/bin:$PATH"
export AIRFLOW_HOME="/root/lawlm/airflow"

# Activate conda environment and ensure AIRFLOW_HOME is set for all operations
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lawlm

echo "📍 Using AIRFLOW_HOME: $AIRFLOW_HOME"

# Function to check if a service is running
check_service_running() {
    local service_name="$1"
    local pattern="$2"
    if pgrep -f "$pattern" > /dev/null 2>&1; then
        local pid=$(pgrep -f "$pattern")
        echo "  ✅ $service_name already running (PID: $pid)"
        return 0
    else
        echo "  ❌ $service_name not running"
        return 1
    fi
}

# Function to check if Redis container is running
check_redis_running() {
    if docker ps | grep -q "redis-courtlistener"; then
        echo "  ✅ Redis container already running"
        return 0
    else
        echo "  ❌ Redis container not running"
        return 1
    fi
}

# Function to stop all services
stop_all_services() {
    # Step 1: Stop All Airflow Services
    echo "🛑 Stopping all Airflow services..."

    # Use more specific patterns to avoid killing the script itself
    pids_to_kill=""

    # Check and stop scheduler
    scheduler_pid=$(pgrep -f "airflow scheduler")
    if [ ! -z "$scheduler_pid" ]; then
        pids_to_kill="$pids_to_kill $scheduler_pid"
        echo "  Found scheduler (PID: $scheduler_pid)"
    fi

    # Check and stop webserver
    webserver_pid=$(pgrep -f "airflow webserver")
    if [ ! -z "$webserver_pid" ]; then
        pids_to_kill="$pids_to_kill $webserver_pid"
        echo "  Found webserver (PID: $webserver_pid)"
    fi

    # Check and stop other non-daemon Airflow processes
    other_airflow_pids=$(pgrep -f "airflow.*(run|task|worker)")
    if [ ! -z "$other_airflow_pids" ]; then
        pids_to_kill="$pids_to_kill $other_airflow_pids"
        echo "  Found other airflow processes"
    fi

    # Kill the collected PIDs
    if [ ! -z "$pids_to_kill" ]; then
        echo "  Killing processes: $pids_to_kill"
        kill $pids_to_kill 2>/dev/null && echo "✅ Airflow services stopped" || echo "⚠️  Failed to stop some services"
    else
        echo "  No Airflow processes found to stop"
    fi

    # Wait and verify
    echo "  Waiting for processes to terminate..."
    sleep 5
    if pgrep -f "airflow" > /dev/null 2>&1; then
        echo "⚠️  Some Airflow processes still running, force killing..."
        pkill -9 -f "airflow.*(scheduler|webserver|run|task|worker)" 2>/dev/null || true
        sleep 2
        if pgrep -f "airflow" > /dev/null 2>&1; then
            echo "❌ Failed to stop all processes, continuing anyway..."
        else
            echo "✅ All processes force killed"
        fi
    else
        echo "✅ All Airflow services stopped"
    fi


    # Stop Redis container
    echo "🛑 Stopping Redis container..."
    if docker ps | grep -q "redis-courtlistener"; then
        if docker stop redis-courtlistener && docker rm redis-courtlistener; then
            echo "✅ Redis container stopped and removed"
        else
            echo "⚠️ Failed to stop Redis container"
        fi
    else
        echo "ℹ️ Redis container not running"
    fi
}

# Handle stop mode
if [ "$STOP_MODE" = true ]; then
    stop_all_services
    echo "🎉 All services stopped successfully!"
    exit 0
fi

# Only perform destructive operations in reset mode
if [ "$RESET_MODE" = true ]; then
    stop_all_services
    echo "✅ Process stopping completed, continuing to database cleanup..."

    # Step 2: Remove Database File and PID Files
    echo "🗑️  Removing SQLite database file and PID files..."
    if [ -f "/root/lawlm/airflow/airflow.db" ]; then
        rm /root/lawlm/airflow/airflow.db
        echo "✅ Database file removed"
    else
        echo "ℹ️  Database file not found (already clean)"
    fi

    # Remove all PID files to prevent daemon startup conflicts
    echo "🗑️  Removing stale PID files..."
    rm -f /root/lawlm/airflow/*.pid 2>/dev/null || true
    echo "✅ PID files cleaned"

    # Step 3: Clean Up Cache and Logs
    echo "🧹 Cleaning up cache and logs..."
    rm -rf /root/lawlm/airflow/__pycache__/ 2>/dev/null || true
    rm -rf /root/lawlm/airflow/dags/__pycache__/ 2>/dev/null || true
    rm -rf /root/lawlm/airflow/logs/* 2>/dev/null || true
    echo "✅ Cache and logs cleaned"

    # Step 3.5: Clean Up Qdrant Collection
    echo "🗑️ Cleaning up Qdrant collection..."
    if python -c "
import sys
sys.path.append('/root/lawlm')
try:
    from config import load_config
    from vector_processor import EnhancedVectorProcessor
    from qdrant_client.models import FilterSelector, Filter, VectorParams, Distance
    import qdrant_client.models as models

    print('Loading configuration...')
    config = load_config()

    print('Creating vector processor...')
    processor = EnhancedVectorProcessor(
        model_name=config.vector_processing.embedding_model,
        collection_name=config.vector_processing.collection_name_vector,
        qdrant_url=config.qdrant.url
    )

    print('Getting Qdrant client...')
    client = processor._get_qdrant_client()

    try:
        print('Checking collection exists...')
        collection_info = client.get_collection(config.vector_processing.collection_name_vector)
        initial_points = collection_info.points_count
        print(f'Collection has {initial_points} points')

        if initial_points > 0:
            print('Deleting all points...')
            delete_result = client.delete(
                collection_name=config.vector_processing.collection_name_vector,
                points_selector=FilterSelector(filter=Filter())
            )

            # Verify deletion
            collection_info = client.get_collection(config.vector_processing.collection_name_vector)
            remaining_points = collection_info.points_count

            if remaining_points > 0:
                print(f'Still {remaining_points} points remaining. Recreating collection...')
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
                print('Collection recreated successfully')
            else:
                print('All points deleted successfully')
        else:
            print('Collection is already empty')

    except Exception as e:
        if 'not found' in str(e).lower():
            print('Collection does not exist, nothing to clean')
        else:
            print(f'Error during Qdrant cleanup: {e}')
            raise e

except Exception as e:
    print(f'Failed to clean Qdrant collection: {e}')
    exit(1)

print('Qdrant collection cleanup completed')
" 2>/dev/null; then
        echo "✅ Qdrant collection cleaned"
    else
        echo "⚠️ Qdrant cleanup failed or skipped (collection may not exist)"
    fi

    # Step 4: Initialize Fresh Database
    echo "🔧 Initializing fresh database..."
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow db init 2>/dev/null; then
        echo "✅ Fresh database initialized"
    else
        echo "❌ Database initialization failed, trying again..."
        AIRFLOW_HOME="$AIRFLOW_HOME" airflow db init 2>&1 || echo "Database init had issues but continuing..."
    fi

    # Step 5: Create Admin User
    echo "👤 Creating admin user..."
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin 2>/dev/null; then
        echo "✅ Admin user created (username: admin, password: admin)"
    else
        echo "❌ User creation failed, trying again..."
        AIRFLOW_HOME="$AIRFLOW_HOME" airflow users create \
            --username admin \
            --firstname Admin \
            --lastname User \
            --role Admin \
            --email admin@example.com \
            --password admin 2>&1 || echo "User creation had issues but continuing..."
    fi
else
    echo "🔍 Checking current service status..."
    check_service_running "Scheduler" "airflow scheduler"
    check_service_running "Webserver" "airflow webserver"
    check_redis_running
    echo ""
fi

# Step 6: Create/Ensure API Pools
echo "🏊 Creating/ensuring API pools..."
# Check if pool already exists
if AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools list 2>/dev/null | grep -q "courtlistener_api_pool"; then
    echo "  ℹ️  courtlistener_api_pool already exists"
else
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting" 2>/dev/null; then
        echo "✅ API pools created"
    else
        echo "❌ Pool creation failed, trying again..."
        AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting" 2>&1 || echo "Pool creation had issues but continuing..."
    fi
fi

# Step 7: Start/Ensure Services
echo "🚀 Starting/ensuring Airflow services..."

# Only kill processes in reset mode
if [ "$RESET_MODE" = true ]; then
    # Make sure no processes are still running
    pkill -f "airflow.*(scheduler|webserver|run|task|worker)" 2>/dev/null || true
    sleep 3
fi

echo "  Starting/ensuring scheduler..."
if ! check_service_running "Scheduler" "airflow scheduler" > /dev/null 2>&1; then
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow scheduler --daemon 2>/dev/null; then
        sleep 3
        if pgrep -f "airflow scheduler" > /dev/null 2>&1; then
            echo "✅ Scheduler started successfully"
        else
            echo "❌ Scheduler process not found after start, retrying..."
            AIRFLOW_HOME="$AIRFLOW_HOME" airflow scheduler --daemon 2>/dev/null || true
            sleep 3
        fi
    else
        echo "❌ Scheduler command failed, trying again..."
        AIRFLOW_HOME="$AIRFLOW_HOME" airflow scheduler --daemon 2>/dev/null || true
        sleep 3
    fi
else
    echo "  ℹ️  Scheduler already running, skipping start"
fi

echo "  Starting/ensuring webserver..."
if ! check_service_running "Webserver" "airflow webserver" > /dev/null 2>&1; then
    # Only clean up port in reset mode or if no webserver is running
    if [ "$RESET_MODE" = true ] || ! pgrep -f "airflow webserver" > /dev/null 2>&1; then
        if ss -tuln | grep -q ":8080 "; then
            echo "    ⚠️  Port 8080 is already in use, killing existing process..."
            lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
    fi
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow webserver --port 8080 --daemon 2>/dev/null; then
        sleep 5
        if pgrep -f "airflow webserver" > /dev/null 2>&1; then
            echo "✅ Webserver started successfully on port 8080"
        else
            echo "❌ Webserver process not found after start, retrying..."
            AIRFLOW_HOME="$AIRFLOW_HOME" airflow webserver --port 8080 --daemon 2>/dev/null || true
            sleep 5
        fi
    else
        echo "❌ Webserver command failed, trying again..."
        AIRFLOW_HOME="$AIRFLOW_HOME" airflow webserver --port 8080 --daemon 2>/dev/null || true
        sleep 5
    fi
else
    echo "  ℹ️  Webserver already running, skipping start"
fi

# Wait for services to fully initialize
echo "  Waiting for services to initialize..."
sleep 5

# Step 7.5: Start/Ensure Redis Docker Container
echo "🔧 Starting/ensuring Redis Docker container for caching..."
if ! check_redis_running > /dev/null 2>&1; then
    # Only stop/remove in reset mode
    if [ "$RESET_MODE" = true ]; then
        # Stop any existing container with the same name
        docker stop redis-courtlistener 2>/dev/null || true
        docker rm redis-courtlistener 2>/dev/null || true
    fi

    # Start fresh Redis container
    if docker run -d --name redis-courtlistener \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine redis-server --appendonly yes; then
        echo "✅ Redis container started successfully on port 6379"

        # Wait for Redis to be ready
        echo "  Waiting for Redis to be ready..."
        sleep 3

        # Test Redis connection
        if docker exec redis-courtlistener redis-cli ping | grep -q "PONG"; then
            echo "✅ Redis is responding to ping"
        else
            echo "⚠️  Redis may not be fully ready yet"
        fi
    else
        echo "❌ Failed to start Redis container"
    fi
else
    echo "  ℹ️  Redis container already running, skipping start"
fi

# Configure Airflow Redis connection with enhanced validation
echo "🔗 Configuring Airflow Redis connection with validation..."
export AIRFLOW_CONN_REDIS_DEFAULT='redis://localhost:6379/0'

# Validate Redis container is accessible on localhost:6379
echo "  🔍 Validating Redis accessibility on localhost:6379..."
if timeout 5 bash -c 'echo > /dev/tcp/localhost/6379' 2>/dev/null; then
    echo "  ✅ Redis is accessible on localhost:6379"
else
    echo "  ❌ Redis is not accessible on localhost:6379"
    echo "  🔄 Attempting to restart Redis container..."
    docker stop redis-courtlistener 2>/dev/null || true
    docker rm redis-courtlistener 2>/dev/null || true
    sleep 2
    docker run -d --name redis-courtlistener \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7-alpine redis-server --appendonly yes
    sleep 3
fi

# Function to validate Redis connection format
validate_redis_connection() {
    local connection_info="$1"
    # Check for localhost in the connection info (should be in the host column)
    if echo "$connection_info" | grep -E "(host.*localhost|localhost.*6379)" > /dev/null; then
        return 0  # Valid
    else
        return 1  # Invalid
    fi
}

# Always validate and fix Redis connection in database
echo "  Checking current Redis connection in Airflow database..."
current_connection=$(AIRFLOW_HOME="$AIRFLOW_HOME" airflow connections get redis_default 2>/dev/null || echo "not_found")

if [[ "$current_connection" == "not_found" ]]; then
    echo "  📝 No Redis connection found, creating new one..."
    connection_needs_update=true
elif validate_redis_connection "$current_connection"; then
    echo "  ✅ Redis connection already configured correctly with localhost"
    echo "     Connection details: $(echo "$current_connection" | grep -E '(host|port)' | tr '\n' ' ')"
    connection_needs_update=false
else
    echo "  ⚠️  Redis connection found but using incorrect host, will update..."
    echo "     Current connection: $(echo "$current_connection" | grep -E '(host|port)' | tr '\n' ' ')"
    connection_needs_update=true
fi

if [[ "$connection_needs_update" == "true" ]]; then
    echo "  🔄 Updating Redis connection in Airflow database..."

    # Always delete existing connection to prevent caching issues
    AIRFLOW_HOME="$AIRFLOW_HOME" airflow connections delete redis_default 2>/dev/null || true

    # Wait a moment for the deletion to complete
    sleep 1

    # Create new connection with correct localhost host
    if AIRFLOW_HOME="$AIRFLOW_HOME" airflow connections add redis_default \
        --conn-type redis \
        --conn-host localhost \
        --conn-port 6379 \
        --conn-schema 0 2>/dev/null; then
        echo "  ✅ Redis connection created successfully with localhost"

        # Verify the new connection
        verification_connection=$(AIRFLOW_HOME="$AIRFLOW_HOME" airflow connections get redis_default 2>/dev/null)
        if validate_redis_connection "$verification_connection"; then
            echo "  🔍 Verification passed: Redis connection using localhost:6379"
        else
            echo "  ❌ Verification failed: Redis connection still incorrect after update!"
            echo "     Verification details: $(echo "$verification_connection" | grep -E '(host|port)' | tr '\n' ' ')"
        fi
    else
        echo "  ❌ Failed to create Redis connection, but environment variable is set as fallback"
    fi
fi

echo "✅ Redis connection configuration completed: redis://localhost:6379/0"

# Final validation: Test Redis connection from Airflow context
echo "  🧪 Final validation: Testing Redis connection from Airflow context..."
if AIRFLOW_HOME="$AIRFLOW_HOME" python -c "
import sys
sys.path.append('/root/lawlm/airflow')
from airflow.hooks.base import BaseHook
try:
    conn = BaseHook.get_connection('redis_default')
    assert conn.host == 'localhost', f'Expected localhost, got {conn.host}'
    print('✅ Airflow Redis connection validation passed')
except Exception as e:
    print(f'❌ Airflow Redis connection validation failed: {e}')
    exit(1)
" 2>/dev/null; then
    echo "  ✅ Final Redis connection validation successful"
else
    echo "  ⚠️  Final Redis connection validation had issues (but continuing...)"
fi

# Step 8: Verify Service State
if [ "$RESET_MODE" = true ]; then
    echo "🔍 Verifying clean state..."
else
    echo "🔍 Verifying service state..."
fi

echo "📋 Available DAGs:"
AIRFLOW_HOME="$AIRFLOW_HOME" airflow dags list || echo "  No DAGs found (expected for clean state)"

echo "🏊 Available pools:"
AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools list

echo "🔧 Running services:"
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "  ✅ Scheduler running (PID: $(pgrep -f 'airflow scheduler'))"
else
    echo "  ❌ Scheduler not running"
fi

if pgrep -f "airflow webserver" > /dev/null; then
    echo "  ✅ Webserver running (PID: $(pgrep -f 'airflow webserver'))"
else
    echo "  ❌ Webserver not running"
fi

if docker ps | grep -q "redis-courtlistener"; then
    echo "  ✅ Redis container running"
else
    echo "  ❌ Redis container not running"
fi

echo ""
if [ "$RESET_MODE" = true ]; then
    echo "🎉 Airflow database reset complete!"
else
    echo "🎉 Airflow services are running!"
fi
echo "📡 Web UI available at: http://localhost:8080"
echo "🔑 Login credentials: admin / admin"
echo ""
echo "Next steps:"
echo "  1. Access the web UI at http://localhost:8080"
echo "  2. Add your DAG files to /root/lawlm/airflow/dags/"
echo "  3. Unpause and trigger your DAGs"
echo ""
echo "Usage:"
echo "  ./run_airflow.sh         # Start/ensure services are running (default)"
echo "  ./run_airflow.sh --reset # Complete database reset and restart"
echo "  ./run_airflow.sh --stop  # Stop all Airflow services and Redis"
#!/bin/bash

# Airflow Complete Database Reset Script
# This script completely wipes and reinitializes the Airflow database

echo "🔄 Starting Airflow complete database reset..."

# Set environment variables - Critical for proper operation
export PATH="/root/anaconda3/envs/lawlm/bin:$PATH"
export AIRFLOW_HOME="/root/lawlm/airflow"

# Activate conda environment and ensure AIRFLOW_HOME is set for all operations
source /root/anaconda3/etc/profile.d/conda.sh
conda activate lawlm

echo "📍 Using AIRFLOW_HOME: $AIRFLOW_HOME"

# Step 1: Stop All Airflow Services
echo "🛑 Stopping all Airflow services..."

# Use more specific patterns to avoid killing the script itself
pids_to_kill=""

# Check and stop scheduler
scheduler_pid=$(pgrep -f "airflow scheduler")
if [ ! -z "$scheduler_pid" ]; then
    pids_to_kill="$pids_to_kill $scheduler_pid"
    echo "  Found scheduler (PID: $scheduler_pid)"
fi

# Check and stop webserver
webserver_pid=$(pgrep -f "airflow webserver")
if [ ! -z "$webserver_pid" ]; then
    pids_to_kill="$pids_to_kill $webserver_pid"
    echo "  Found webserver (PID: $webserver_pid)"
fi

# Check and stop other non-daemon Airflow processes
other_airflow_pids=$(pgrep -f "airflow.*(run|task|worker)")
if [ ! -z "$other_airflow_pids" ]; then
    pids_to_kill="$pids_to_kill $other_airflow_pids"
    echo "  Found other airflow processes"
fi

# Kill the collected PIDs
if [ ! -z "$pids_to_kill" ]; then
    echo "  Killing processes: $pids_to_kill"
    kill $pids_to_kill 2>/dev/null && echo "✅ Airflow services stopped" || echo "⚠️  Failed to stop some services"
else
    echo "  No Airflow processes found to stop"
fi

# Wait and verify
echo "  Waiting for processes to terminate..."
sleep 5
if pgrep -f "airflow" > /dev/null 2>&1; then
    echo "⚠️  Some Airflow processes still running, force killing..."
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

# Step 6: Recreate Pools
echo "🏊 Creating API pools..."
if AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting" 2>/dev/null; then
    echo "✅ API pools created"
else
    echo "❌ Pool creation failed, trying again..."
    AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools set courtlistener_api_pool 10 "CourtListener API pool for rate limiting" 2>&1 || echo "Pool creation had issues but continuing..."
fi

# Step 7: Start Services
echo "🚀 Starting Airflow services..."

# Make sure no processes are still running
pkill -f "airflow.*(scheduler|webserver|run|task|worker)" 2>/dev/null || true
sleep 3

echo "  Starting scheduler..."
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

echo "  Starting webserver..."
# Check if port 8080 is available and clean up if needed
if ss -tuln | grep -q ":8080 "; then
    echo "    ⚠️  Port 8080 is already in use, killing existing process..."
    lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
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

# Wait for services to fully initialize
echo "  Waiting for services to initialize..."
sleep 5

# Step 7.5: Start Redis Docker Container
echo "🔧 Starting Redis Docker container for caching..."
if docker ps | grep -q "redis-courtlistener"; then
    echo "  ℹ️  Redis container already running"
else
    # Stop any existing container with the same name
    docker stop redis-courtlistener 2>/dev/null || true
    docker rm redis-courtlistener 2>/dev/null || true

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
fi

# Configure Airflow Redis connection
echo "🔗 Configuring Airflow Redis connection..."
export AIRFLOW_CONN_REDIS_DEFAULT='redis://localhost:6379/0'
echo "✅ Redis connection configured: redis://localhost:6379/0"

# Step 8: Verify Clean State
echo "🔍 Verifying clean state..."

echo "📋 Available DAGs:"
AIRFLOW_HOME="$AIRFLOW_HOME" airflow dags list || echo "  No DAGs found (expected for clean state)"

echo "🏊 Available pools:"
AIRFLOW_HOME="$AIRFLOW_HOME" airflow pools list

echo "🔧 Running services:"
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "  ✅ Scheduler running (PID: $(pgrep -f 'airflow scheduler'))"
else
    echo "  ❌ Scheduler not running"
fi

if pgrep -f "airflow webserver" > /dev/null; then
    echo "  ✅ Webserver running (PID: $(pgrep -f 'airflow webserver'))"
else
    echo "  ❌ Webserver not running"
fi

echo ""
echo "🎉 Airflow database reset complete!"
echo "📡 Web UI available at: http://localhost:8080"
echo "🔑 Login credentials: admin / admin"
echo ""
echo "Next steps:"
echo "  1. Access the web UI at http://localhost:8080"
echo "  2. Add your DAG files to /root/lawlm/airflow/dags/"
echo "  3. Unpause and trigger your DAGs"
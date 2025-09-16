#!/bin/bash

# Simple Airflow Setup for Legal Document Pipeline
# Avoids version conflicts by using constraint files

set -e

echo "🚀 Setting up Airflow for Legal Document Pipeline"
echo "================================================="

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "lawlm" ]]; then
    echo "⚠️  Please activate the lawlm conda environment first:"
    echo "   conda activate lawlm"
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "❌ .env file not found. Please create it with your API keys."
    echo "   Copy .env.template to .env and add your CASELAW_API_KEY"
    exit 1
fi

# Source environment variables
source .env

# Check for required API key
if [[ -z "$CASELAW_API_KEY" ]]; then
    echo "❌ CASELAW_API_KEY not found in .env file"
    exit 1
fi

echo "✅ Environment checks passed"

# Set Airflow home
export AIRFLOW_HOME="${PWD}/airflow"
mkdir -p "$AIRFLOW_HOME"

echo "📦 Installing Airflow with constraints..."

# Use Airflow's official constraint file for Python 3.10
AIRFLOW_VERSION=2.7.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install core Airflow
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Install our additional dependencies
pip install backoff==2.2.1

echo "🔧 Initializing Airflow..."

# Initialize Airflow database
airflow db init

echo "👤 Creating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "⚙️  Setting up pools and variables..."

# Set up API pool for rate limiting
airflow pools set courtlistener_api_pool 50 'CourtListener API rate limiting pool - high throughput'

# Set pipeline variables
airflow variables set CASELAW_API_KEY "$CASELAW_API_KEY"
airflow variables set COURT "scotus"
airflow variables set BATCH_SIZE "495"
airflow variables set TOTAL_BATCHES "1"

echo "✅ Airflow setup complete!"
echo ""
echo "🎯 Pipeline configured for maximum throughput:"
echo "   📊 4950 API calls/hour (99% of rate limit)"
echo "   📋 2,475 dockets/hour"
echo "   🛡️  Soft rate limiting prevents partial processing"
echo ""

# Check if Qdrant is running
echo "🔍 Checking Qdrant status..."
if ./manage_qdrant.sh status | grep -q "running"; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant is not running. Starting it now..."
    ./manage_qdrant.sh start
fi

echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Start Airflow Scheduler (Terminal 1):"
echo "   export AIRFLOW_HOME=${PWD}/airflow"
echo "   airflow scheduler"
echo ""
echo "2. Start Airflow Webserver (Terminal 2):"
echo "   export AIRFLOW_HOME=${PWD}/airflow"
echo "   airflow webserver --port 8080"
echo ""
echo "3. Access Airflow UI: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "4. Enable the 'courtlistener_legal_pipeline' DAG"
echo ""
echo "📊 Monitor with: airflow variables list | grep courtlistener_calls"
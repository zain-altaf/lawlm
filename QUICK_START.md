# Quick Start Guide - Maximum Throughput Pipeline

**4950 API calls/hour | 2,475 dockets/hour | Soft rate limiting**

## 🚀 Simple Setup

### 1. Run Setup Script
```bash
./setup_airflow_simple.sh
```

### 2. Start Airflow (2 terminals)

**Terminal 1:**
```bash
export AIRFLOW_HOME=/root/lawlm/airflow
airflow scheduler
```

**Terminal 2:**
```bash
export AIRFLOW_HOME=/root/lawlm/airflow
airflow webserver --port 8080
```

### 3. Enable Pipeline
- Go to http://localhost:8080
- Login: `admin` / `admin`
- Enable `courtlistener_legal_pipeline`
- **Done!** Runs every 12 minutes automatically

## 📊 What You Get
- **4950 API calls/hour** (99% of rate limit)
- **2,475 dockets/hour** (59,400/day, 1.78M/month)
- **Smart rate limiting** - no partial docket processing
- **Complete orchestration** with monitoring

## 🔍 Monitor
```bash
# Check API usage
airflow variables list | grep courtlistener_calls

# View logs in Airflow UI → DAGs → Graph View
```

---

**Need help?** See `MANUAL_SETUP.md` for detailed troubleshooting.
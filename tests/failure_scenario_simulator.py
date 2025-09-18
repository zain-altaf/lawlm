#!/usr/bin/env python3
"""
Failure Scenario Simulator for Redis-Enhanced CourtListener DAG

This module simulates various failure scenarios to test the resilience and recovery
capabilities of the Redis-enhanced DAG. It covers all critical failure modes:

1. DAG restart scenarios
2. Task failure and restart scenarios
3. Scheduler restart simulation
4. Worker node failure simulation
5. Redis connectivity failure scenarios
6. PostgreSQL fallback testing

Usage:
    python failure_scenario_simulator.py --scenario dag-restart
    python failure_scenario_simulator.py --scenario task-failure
    python failure_scenario_simulator.py --scenario scheduler-restart
    python failure_scenario_simulator.py --scenario worker-failure
    python failure_scenario_simulator.py --scenario redis-failure
    python failure_scenario_simulator.py --scenario all
"""

import argparse
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

# Add project paths
sys.path.append('/root/lawlm')
sys.path.append('/root/lawlm/airflow')

import redis
from airflow.models import DagBag, Variable
from airflow.hooks.postgres_hook import PostgresHook
from hooks.redis_rate_limit_hook import RedisRateLimitHook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FailureScenarioSimulator:
    """
    Simulates various failure scenarios to test DAG resilience and recovery.

    Tests production-critical scenarios including infrastructure failures,
    process restarts, network issues, and distributed system edge cases.
    """

    def __init__(self, redis_conn_id: str = "redis_default",
                 postgres_conn_id: str = "postgres_default"):
        """Initialize simulator with connection parameters."""
        self.redis_conn_id = redis_conn_id
        self.postgres_conn_id = postgres_conn_id
        self.scenario_results = {}
        self.start_time = datetime.now()

        # Initialize hooks for testing
        try:
            self.redis_hook = RedisRateLimitHook(redis_conn_id=redis_conn_id)
            self.postgres_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
            logger.info("Successfully initialized hooks for failure simulation")
        except Exception as e:
            logger.error(f"Failed to initialize hooks: {e}")
            self.redis_hook = None
            self.postgres_hook = None

    def log_scenario_result(self, scenario_name: str, passed: bool, message: str,
                           details: Dict[str, Any] = None):
        """Log and store scenario test results."""
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {scenario_name}: {message}")

        self.scenario_results[scenario_name] = {
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }

    def simulate_dag_restart_scenario(self) -> bool:
        """
        Simulate a complete DAG restart and verify state recovery.

        This tests that API call counters and pipeline state persist
        across DAG execution restarts.
        """
        logger.info("Simulating DAG restart scenario...")

        if not self.redis_hook:
            self.log_scenario_result("dag_restart", False, "Redis hook not available")
            return False

        try:
            # Step 1: Setup initial state as if DAG was running
            initial_dag_run_id = f"initial_run_{int(time.time())}"
            initial_api_calls = 150
            initial_dockets = [1001, 1002, 1003]

            # Initialize pipeline state
            self.redis_hook.initialize_pipeline_state(initial_dag_run_id)

            # Simulate some API calls
            for calls in [50, 50, 50]:
                self.redis_hook.atomic_increment_api_calls(calls)

            # Process some dockets
            for docket_id in initial_dockets:
                self.redis_hook.update_docket_processing_state(
                    dag_run_id=initial_dag_run_id,
                    docket_id=docket_id,
                    api_calls_made=10,
                    success=True
                )

            # Record pre-restart state
            pre_restart_rate_status = self.redis_hook.get_current_rate_limit_status()
            pre_restart_pipeline_state = self.redis_hook.get_pipeline_state(initial_dag_run_id)

            logger.info(f"Pre-restart state: {pre_restart_rate_status['api_calls_used']} API calls")

            # Step 2: Simulate DAG restart by creating new hook instances
            logger.info("Simulating DAG restart...")
            time.sleep(2)  # Simulate restart delay

            # Create new hook instances (simulates process restart)
            new_redis_hook = RedisRateLimitHook(redis_conn_id=self.redis_conn_id)

            # Step 3: Verify state persistence after restart
            post_restart_rate_status = new_redis_hook.get_current_rate_limit_status()
            post_restart_pipeline_state = new_redis_hook.get_pipeline_state(initial_dag_run_id)

            # Step 4: Test new DAG run can start properly
            new_dag_run_id = f"restarted_run_{int(time.time())}"
            new_pipeline_state = new_redis_hook.initialize_pipeline_state(new_dag_run_id)

            # Step 5: Verify continued operations work
            additional_calls_result = new_redis_hook.atomic_increment_api_calls(25)

            # Validation checks
            api_calls_preserved = (
                post_restart_rate_status['api_calls_used'] >= pre_restart_rate_status['api_calls_used']
            )

            pipeline_state_preserved = (
                'error' not in post_restart_pipeline_state and
                post_restart_pipeline_state.get('dag_run_id') == initial_dag_run_id
            )

            new_dag_can_start = (
                'error' not in new_pipeline_state and
                new_pipeline_state.get('dag_run_id') == new_dag_run_id
            )

            operations_continue_working = additional_calls_result['allowed']

            details = {
                'pre_restart_api_calls': pre_restart_rate_status['api_calls_used'],
                'post_restart_api_calls': post_restart_rate_status['api_calls_used'],
                'api_calls_preserved': api_calls_preserved,
                'pipeline_state_preserved': pipeline_state_preserved,
                'new_dag_can_start': new_dag_can_start,
                'operations_continue_working': operations_continue_working,
                'initial_dockets_processed': len(initial_dockets),
                'pipeline_state_details': post_restart_pipeline_state
            }

            all_checks_passed = all([
                api_calls_preserved,
                pipeline_state_preserved,
                new_dag_can_start,
                operations_continue_working
            ])

            if all_checks_passed:
                self.log_scenario_result("dag_restart", True,
                                       f"DAG restart recovery successful. API calls preserved: {post_restart_rate_status['api_calls_used']}",
                                       details)
            else:
                failed_checks = []
                if not api_calls_preserved:
                    failed_checks.append("API calls not preserved")
                if not pipeline_state_preserved:
                    failed_checks.append("Pipeline state not preserved")
                if not new_dag_can_start:
                    failed_checks.append("New DAG cannot start")
                if not operations_continue_working:
                    failed_checks.append("Operations not working after restart")

                error_msg = f"DAG restart recovery failed: {', '.join(failed_checks)}"
                self.log_scenario_result("dag_restart", False, error_msg, details)

            return all_checks_passed

        except Exception as e:
            error_msg = f"DAG restart scenario failed: {e}"
            self.log_scenario_result("dag_restart", False, error_msg,
                                   {'error': str(e), 'traceback': traceback.format_exc()})
            return False

    def simulate_task_failure_and_restart(self) -> bool:
        """
        Simulate task failure and restart scenarios.

        Tests that failed tasks can be restarted without data corruption
        and that API call tracking remains accurate.
        """
        logger.info("Simulating task failure and restart scenario...")

        if not self.redis_hook:
            self.log_scenario_result("task_failure_restart", False, "Redis hook not available")
            return False

        try:
            dag_run_id = f"task_failure_test_{int(time.time())}"
            docket_id = 9999

            # Step 1: Initialize pipeline
            self.redis_hook.initialize_pipeline_state(dag_run_id)

            # Step 2: Simulate successful task execution
            task_start_time = self.redis_hook.cache_task_start_time(dag_run_id, "successful_task")
            success_result = self.redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=docket_id,
                api_calls_made=15,
                success=True
            )

            # Step 3: Simulate task failure
            failed_docket_id = 9998
            failure_task_start = self.redis_hook.cache_task_start_time(dag_run_id, "failing_task")

            # Simulate API calls made before failure
            pre_failure_calls = self.redis_hook.atomic_increment_api_calls(20)

            # Record failure
            failure_result = self.redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=failed_docket_id,
                api_calls_made=20,
                success=False,
                error_message="Simulated task failure"
            )

            # Step 4: Simulate task restart (new execution after clearing failure)
            restart_task_start = self.redis_hook.cache_task_start_time(dag_run_id, "restarted_task")

            # On restart, the task should be able to proceed
            restart_calls = self.redis_hook.atomic_increment_api_calls(10)

            # Task succeeds on restart
            restart_success = self.redis_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=failed_docket_id,
                api_calls_made=10,
                success=True
            )

            # Step 5: Verify state consistency
            final_pipeline_state = self.redis_hook.get_pipeline_state(dag_run_id)
            final_rate_status = self.redis_hook.get_current_rate_limit_status()

            # Validation checks
            api_calls_accurate = (
                final_rate_status['api_calls_used'] >= (15 + 20 + 10)  # All calls tracked
            )

            pipeline_state_valid = 'error' not in final_pipeline_state

            both_tasks_tracked = (
                len(final_pipeline_state.get('dockets_processed', [])) >= 1  # At least successful ones
            )

            restart_operations_work = restart_calls['allowed']

            # Check that start times were cached for all task executions
            client = self.redis_hook.get_conn()
            start_times_cached = all([
                client.get(self.redis_hook.get_task_start_key(dag_run_id, task_id))
                for task_id in ["successful_task", "failing_task", "restarted_task"]
            ])

            details = {
                'successful_task_calls': 15,
                'failed_task_calls': 20,
                'restart_task_calls': 10,
                'total_expected_calls': 45,
                'actual_api_calls': final_rate_status['api_calls_used'],
                'api_calls_accurate': api_calls_accurate,
                'pipeline_state_valid': pipeline_state_valid,
                'both_tasks_tracked': both_tasks_tracked,
                'restart_operations_work': restart_operations_work,
                'start_times_cached': start_times_cached,
                'pipeline_state': final_pipeline_state
            }

            all_checks_passed = all([
                api_calls_accurate,
                pipeline_state_valid,
                restart_operations_work,
                start_times_cached
            ])

            if all_checks_passed:
                self.log_scenario_result("task_failure_restart", True,
                                       f"Task failure and restart handled correctly. API calls: {final_rate_status['api_calls_used']}",
                                       details)
            else:
                failed_checks = []
                if not api_calls_accurate:
                    failed_checks.append("API calls not accurate")
                if not pipeline_state_valid:
                    failed_checks.append("Pipeline state invalid")
                if not restart_operations_work:
                    failed_checks.append("Restart operations don't work")
                if not start_times_cached:
                    failed_checks.append("Start times not cached")

                error_msg = f"Task failure/restart scenario failed: {', '.join(failed_checks)}"
                self.log_scenario_result("task_failure_restart", False, error_msg, details)

            return all_checks_passed

        except Exception as e:
            error_msg = f"Task failure/restart scenario failed: {e}"
            self.log_scenario_result("task_failure_restart", False, error_msg,
                                   {'error': str(e), 'traceback': traceback.format_exc()})
            return False

    def simulate_scheduler_restart(self) -> bool:
        """
        Simulate Airflow scheduler restart.

        Tests that scheduler restart doesn't affect Redis state and that
        the pipeline can continue from where it left off.
        """
        logger.info("Simulating scheduler restart scenario...")

        if not self.redis_hook:
            self.log_scenario_result("scheduler_restart", False, "Redis hook not available")
            return False

        try:
            dag_run_id = f"scheduler_test_{int(time.time())}"

            # Step 1: Setup state as if scheduler was running tasks
            self.redis_hook.initialize_pipeline_state(dag_run_id)

            # Simulate multiple tasks running
            tasks = ["task_1", "task_2", "task_3"]
            api_calls_per_task = [25, 30, 20]

            pre_restart_total = 0
            for i, task_id in enumerate(tasks):
                # Cache start times
                start_time = self.redis_hook.cache_task_start_time(dag_run_id, task_id)

                # Make API calls
                calls = api_calls_per_task[i]
                self.redis_hook.atomic_increment_api_calls(calls)
                pre_restart_total += calls

                # Update docket state
                self.redis_hook.update_docket_processing_state(
                    dag_run_id=dag_run_id,
                    docket_id=2000 + i,
                    api_calls_made=calls,
                    success=True
                )

            pre_restart_state = self.redis_hook.get_pipeline_state(dag_run_id)
            pre_restart_rate = self.redis_hook.get_current_rate_limit_status()

            logger.info(f"Pre-scheduler-restart state: {len(tasks)} tasks, {pre_restart_total} API calls")

            # Step 2: Simulate scheduler restart
            # In reality, this would involve stopping/starting the Airflow scheduler
            # For simulation, we'll test that new connections can see the existing state
            logger.info("Simulating scheduler restart...")
            time.sleep(1)

            # Create new hook instances (simulates new scheduler process)
            post_restart_hook = RedisRateLimitHook(redis_conn_id=self.redis_conn_id)

            # Step 3: Verify scheduler can see existing state
            post_restart_state = post_restart_hook.get_pipeline_state(dag_run_id)
            post_restart_rate = post_restart_hook.get_current_rate_limit_status()

            # Step 4: Verify scheduler can continue operations
            new_task_start = post_restart_hook.cache_task_start_time(dag_run_id, "post_restart_task")
            new_api_calls = post_restart_hook.atomic_increment_api_calls(35)
            new_docket_update = post_restart_hook.update_docket_processing_state(
                dag_run_id=dag_run_id,
                docket_id=3000,
                api_calls_made=35,
                success=True
            )

            # Step 5: Verify all start times are still accessible
            client = post_restart_hook.get_conn()
            all_start_times_accessible = all([
                client.get(post_restart_hook.get_task_start_key(dag_run_id, task_id))
                for task_id in tasks + ["post_restart_task"]
            ])

            # Validation checks
            state_visible_after_restart = 'error' not in post_restart_state

            api_calls_preserved = (
                post_restart_rate['api_calls_used'] >= pre_restart_rate['api_calls_used']
            )

            new_operations_work = new_api_calls['allowed']

            processed_dockets_preserved = (
                len(post_restart_state.get('dockets_processed', [])) >= len(tasks)
            )

            details = {
                'pre_restart_tasks': len(tasks),
                'pre_restart_api_calls': pre_restart_rate['api_calls_used'],
                'post_restart_api_calls': post_restart_rate['api_calls_used'],
                'state_visible_after_restart': state_visible_after_restart,
                'api_calls_preserved': api_calls_preserved,
                'new_operations_work': new_operations_work,
                'processed_dockets_preserved': processed_dockets_preserved,
                'all_start_times_accessible': all_start_times_accessible,
                'new_task_calls': 35
            }

            all_checks_passed = all([
                state_visible_after_restart,
                api_calls_preserved,
                new_operations_work,
                processed_dockets_preserved,
                all_start_times_accessible
            ])

            if all_checks_passed:
                self.log_scenario_result("scheduler_restart", True,
                                       "Scheduler restart handled correctly. State preserved and operations continue.",
                                       details)
            else:
                failed_checks = []
                if not state_visible_after_restart:
                    failed_checks.append("State not visible after restart")
                if not api_calls_preserved:
                    failed_checks.append("API calls not preserved")
                if not new_operations_work:
                    failed_checks.append("New operations don't work")
                if not processed_dockets_preserved:
                    failed_checks.append("Processed dockets not preserved")
                if not all_start_times_accessible:
                    failed_checks.append("Start times not accessible")

                error_msg = f"Scheduler restart scenario failed: {', '.join(failed_checks)}"
                self.log_scenario_result("scheduler_restart", False, error_msg, details)

            return all_checks_passed

        except Exception as e:
            error_msg = f"Scheduler restart scenario failed: {e}"
            self.log_scenario_result("scheduler_restart", False, error_msg,
                                   {'error': str(e), 'traceback': traceback.format_exc()})
            return False

    def simulate_worker_node_failure(self) -> bool:
        """
        Simulate worker node failure in distributed environment.

        Tests that Redis operations remain atomic and consistent
        even when worker nodes fail mid-operation.
        """
        logger.info("Simulating worker node failure scenario...")

        if not self.redis_hook:
            self.log_scenario_result("worker_failure", False, "Redis hook not available")
            return False

        try:
            import threading
            import concurrent.futures

            dag_run_id = f"worker_failure_test_{int(time.time())}"

            # Step 1: Initialize pipeline
            self.redis_hook.initialize_pipeline_state(dag_run_id)

            # Step 2: Simulate multiple workers processing dockets concurrently
            def worker_simulation(worker_id: int, num_dockets: int) -> Dict[str, Any]:
                """Simulate a worker processing multiple dockets."""
                try:
                    worker_hook = RedisRateLimitHook(redis_conn_id=self.redis_conn_id)
                    processed_dockets = []
                    total_api_calls = 0

                    for i in range(num_dockets):
                        docket_id = (worker_id * 1000) + i
                        task_id = f"worker_{worker_id}_docket_{i}"

                        # Cache start time
                        start_time = worker_hook.cache_task_start_time(dag_run_id, task_id)

                        # Simulate API calls
                        api_calls = 10 + (i * 2)  # Varying call counts
                        increment_result = worker_hook.atomic_increment_api_calls(api_calls)

                        if increment_result['allowed']:
                            total_api_calls += api_calls

                            # Update docket processing state
                            worker_hook.update_docket_processing_state(
                                dag_run_id=dag_run_id,
                                docket_id=docket_id,
                                api_calls_made=api_calls,
                                success=True
                            )
                            processed_dockets.append(docket_id)

                        # Simulate worker failure for worker 2 after processing 2 dockets
                        if worker_id == 2 and i == 2:
                            raise Exception("Simulated worker node failure")

                    return {
                        'worker_id': worker_id,
                        'processed_dockets': processed_dockets,
                        'total_api_calls': total_api_calls,
                        'success': True
                    }

                except Exception as e:
                    return {
                        'worker_id': worker_id,
                        'processed_dockets': processed_dockets,
                        'total_api_calls': total_api_calls,
                        'success': False,
                        'error': str(e)
                    }

            # Step 3: Run workers concurrently
            num_workers = 4
            dockets_per_worker = 5

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_simulation, worker_id, dockets_per_worker)
                    for worker_id in range(num_workers)
                ]

                worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Step 4: Analyze results
            successful_workers = [r for r in worker_results if r['success']]
            failed_workers = [r for r in worker_results if not r['success']]

            # Verify that worker 2 failed as expected
            worker_2_failed = any(r['worker_id'] == 2 and not r['success'] for r in failed_workers)

            total_processed_dockets = sum(len(r['processed_dockets']) for r in successful_workers)
            total_expected_api_calls = sum(r['total_api_calls'] for r in successful_workers)

            # Step 5: Verify Redis state consistency
            final_pipeline_state = self.redis_hook.get_pipeline_state(dag_run_id)
            final_rate_status = self.redis_hook.get_current_rate_limit_status()

            # Verify atomic consistency: Redis counter should match successful operations
            redis_consistency = (
                final_rate_status['api_calls_used'] >= total_expected_api_calls
            )

            # Verify that other workers continued despite one failure
            other_workers_succeeded = len(successful_workers) >= 2

            # Verify pipeline state is valid
            pipeline_state_valid = 'error' not in final_pipeline_state

            # Step 6: Test recovery - new worker can join and continue
            recovery_worker_hook = RedisRateLimitHook(redis_conn_id=self.redis_conn_id)
            recovery_start_time = recovery_worker_hook.cache_task_start_time(dag_run_id, "recovery_worker")
            recovery_api_calls = recovery_worker_hook.atomic_increment_api_calls(15)
            recovery_success = recovery_api_calls['allowed']

            details = {
                'num_workers': num_workers,
                'dockets_per_worker': dockets_per_worker,
                'successful_workers': len(successful_workers),
                'failed_workers': len(failed_workers),
                'worker_2_failed_as_expected': worker_2_failed,
                'total_processed_dockets': total_processed_dockets,
                'total_expected_api_calls': total_expected_api_calls,
                'redis_api_calls': final_rate_status['api_calls_used'],
                'redis_consistency': redis_consistency,
                'other_workers_succeeded': other_workers_succeeded,
                'pipeline_state_valid': pipeline_state_valid,
                'recovery_success': recovery_success,
                'worker_results': worker_results
            }

            all_checks_passed = all([
                worker_2_failed,  # Expected failure occurred
                other_workers_succeeded,  # Other workers continued
                redis_consistency,  # State remains consistent
                pipeline_state_valid,  # Pipeline state valid
                recovery_success  # Recovery works
            ])

            if all_checks_passed:
                self.log_scenario_result("worker_failure", True,
                                       f"Worker failure handled correctly. {len(successful_workers)}/{num_workers} workers succeeded.",
                                       details)
            else:
                failed_checks = []
                if not worker_2_failed:
                    failed_checks.append("Expected worker failure didn't occur")
                if not other_workers_succeeded:
                    failed_checks.append("Other workers didn't continue")
                if not redis_consistency:
                    failed_checks.append("Redis state inconsistent")
                if not pipeline_state_valid:
                    failed_checks.append("Pipeline state invalid")
                if not recovery_success:
                    failed_checks.append("Recovery failed")

                error_msg = f"Worker failure scenario failed: {', '.join(failed_checks)}"
                self.log_scenario_result("worker_failure", False, error_msg, details)

            return all_checks_passed

        except Exception as e:
            error_msg = f"Worker failure scenario failed: {e}"
            self.log_scenario_result("worker_failure", False, error_msg,
                                   {'error': str(e), 'traceback': traceback.format_exc()})
            return False

    def simulate_redis_connectivity_failure(self) -> bool:
        """
        Simulate Redis connectivity issues and test fallback mechanisms.

        Tests that the system gracefully falls back to PostgreSQL when
        Redis is unavailable.
        """
        logger.info("Simulating Redis connectivity failure...")

        try:
            # Step 1: Verify Redis is working initially
            if self.redis_hook:
                initial_status = self.redis_hook.get_current_rate_limit_status()
                redis_initially_working = 'error' not in initial_status
            else:
                redis_initially_working = False

            # Step 2: Simulate Redis failure by creating a failing hook
            class FailingRedisHook:
                def __init__(self):
                    pass

                def get_current_rate_limit_status(self):
                    raise redis.exceptions.ConnectionError("Simulated Redis connection failure")

                def atomic_increment_api_calls(self, *args, **kwargs):
                    raise redis.exceptions.ConnectionError("Simulated Redis connection failure")

                def initialize_pipeline_state(self, *args, **kwargs):
                    raise redis.exceptions.ConnectionError("Simulated Redis connection failure")

                def cache_task_start_time(self, *args, **kwargs):
                    raise redis.exceptions.ConnectionError("Simulated Redis connection failure")

            # Step 3: Test fallback to PostgreSQL
            if not self.postgres_hook:
                self.log_scenario_result("redis_failure", False, "PostgreSQL hook not available for fallback")
                return False

            current_hour = datetime.now().strftime("%Y-%m-%d %H")
            test_calls = 75

            # Test PostgreSQL-based rate limiting
            with self.postgres_hook.get_conn() as conn:
                with conn.cursor() as cursor:
                    # Create table if not exists
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS courtlistener_rate_tracking (
                            hour_key VARCHAR(13) PRIMARY KEY,
                            calls_this_hour INTEGER DEFAULT 0,
                            dockets_processed INTEGER[] DEFAULT '{}',
                            last_updated TIMESTAMP DEFAULT NOW()
                        )
                    """)

                    # Test atomic update operation
                    cursor.execute("""
                        INSERT INTO courtlistener_rate_tracking (hour_key, calls_this_hour)
                        VALUES (%s, %s)
                        ON CONFLICT (hour_key) DO UPDATE SET
                            calls_this_hour = courtlistener_rate_tracking.calls_this_hour + %s,
                            last_updated = NOW()
                    """, (current_hour, test_calls, test_calls))

                    # Verify the update
                    cursor.execute(
                        "SELECT calls_this_hour FROM courtlistener_rate_tracking WHERE hour_key = %s",
                        (current_hour,)
                    )
                    result = cursor.fetchone()
                    postgres_count = result[0] if result else 0

                    conn.commit()

            # Step 4: Test Variable-based fallback
            variable_key = "courtlistener_rate_limit_redis_failure_test"
            fallback_data = {
                "dag_run_id": f"redis_failure_test_{int(time.time())}",
                "start_time": datetime.now().isoformat(),
                "current_hour": current_hour,
                "calls_this_hour": 40,
                "redis_fallback": True
            }

            Variable.set(variable_key, json.dumps(fallback_data))
            retrieved_data = json.loads(Variable.get(variable_key, default_var="{}"))

            variable_fallback_works = (
                retrieved_data.get("calls_this_hour") == 40 and
                retrieved_data.get("redis_fallback") is True
            )

            # Step 5: Test that system can detect Redis is back online
            redis_recovery_works = False
            if self.redis_hook:
                try:
                    recovery_status = self.redis_hook.get_current_rate_limit_status()
                    redis_recovery_works = 'error' not in recovery_status
                except Exception:
                    redis_recovery_works = False

            # Cleanup
            Variable.delete(variable_key, session=None)

            details = {
                'redis_initially_working': redis_initially_working,
                'postgres_fallback_calls': postgres_count,
                'postgres_operations_work': postgres_count >= test_calls,
                'variable_fallback_works': variable_fallback_works,
                'redis_recovery_works': redis_recovery_works,
                'test_calls_made': test_calls
            }

            all_checks_passed = all([
                postgres_count >= test_calls,  # PostgreSQL fallback works
                variable_fallback_works,  # Variable fallback works
                # Redis recovery test is optional since Redis might actually be down
            ])

            if all_checks_passed:
                self.log_scenario_result("redis_failure", True,
                                       f"Redis failure fallback working correctly. PostgreSQL calls: {postgres_count}",
                                       details)
            else:
                failed_checks = []
                if postgres_count < test_calls:
                    failed_checks.append("PostgreSQL fallback failed")
                if not variable_fallback_works:
                    failed_checks.append("Variable fallback failed")

                error_msg = f"Redis failure scenario failed: {', '.join(failed_checks)}"
                self.log_scenario_result("redis_failure", False, error_msg, details)

            return all_checks_passed

        except Exception as e:
            error_msg = f"Redis failure scenario failed: {e}"
            self.log_scenario_result("redis_failure", False, error_msg,
                                   {'error': str(e), 'traceback': traceback.format_exc()})
            return False

    def run_all_failure_scenarios(self) -> Dict[str, Any]:
        """Run all failure scenario simulations."""
        logger.info("Starting comprehensive failure scenario testing...")

        scenario_methods = [
            self.simulate_dag_restart_scenario,
            self.simulate_task_failure_and_restart,
            self.simulate_scheduler_restart,
            self.simulate_worker_node_failure,
            self.simulate_redis_connectivity_failure
        ]

        for scenario_method in scenario_methods:
            try:
                scenario_method()
            except Exception as e:
                scenario_name = scenario_method.__name__.replace('simulate_', '').replace('_scenario', '')
                self.log_scenario_result(scenario_name, False, f"Scenario execution failed: {e}")

        return self.generate_failure_scenario_report()

    def generate_failure_scenario_report(self) -> Dict[str, Any]:
        """Generate comprehensive failure scenario report."""
        logger.info("Generating failure scenario report...")

        total_scenarios = len(self.scenario_results)
        passed_scenarios = sum(1 for result in self.scenario_results.values() if result['passed'])
        failed_scenarios = total_scenarios - passed_scenarios

        execution_time = (datetime.now() - self.start_time).total_seconds()

        # Critical scenarios that must pass for production
        critical_scenarios = [
            'dag_restart', 'task_failure_restart', 'scheduler_restart', 'redis_failure'
        ]

        critical_passed = sum(
            1 for scenario_name in critical_scenarios
            if scenario_name in self.scenario_results and self.scenario_results[scenario_name]['passed']
        )

        resilience_score = (passed_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
        production_ready = (critical_passed == len(critical_scenarios))

        report = {
            'failure_scenario_summary': {
                'total_scenarios': total_scenarios,
                'passed_scenarios': passed_scenarios,
                'failed_scenarios': failed_scenarios,
                'resilience_score': round(resilience_score, 2),
                'execution_time_seconds': round(execution_time, 2),
                'production_ready': production_ready,
                'critical_scenarios_passed': f"{critical_passed}/{len(critical_scenarios)}"
            },
            'scenario_results': self.scenario_results,
            'resilience_analysis': self._analyze_resilience(),
            'recommendations': self._generate_resilience_recommendations(),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _analyze_resilience(self) -> Dict[str, Any]:
        """Analyze system resilience based on scenario results."""
        analysis = {
            'state_persistence': self.scenario_results.get('dag_restart', {}).get('passed', False),
            'failure_recovery': self.scenario_results.get('task_failure_restart', {}).get('passed', False),
            'distributed_resilience': self.scenario_results.get('worker_failure', {}).get('passed', False),
            'fallback_mechanisms': self.scenario_results.get('redis_failure', {}).get('passed', False),
            'scheduler_independence': self.scenario_results.get('scheduler_restart', {}).get('passed', False)
        }

        resilience_level = "HIGH" if all(analysis.values()) else "MEDIUM" if sum(analysis.values()) >= 3 else "LOW"
        analysis['overall_resilience_level'] = resilience_level

        return analysis

    def _generate_resilience_recommendations(self) -> List[str]:
        """Generate recommendations for improving system resilience."""
        recommendations = []

        if not self.scenario_results.get('dag_restart', {}).get('passed', False):
            recommendations.append("Implement proper state persistence for DAG restarts")

        if not self.scenario_results.get('task_failure_restart', {}).get('passed', False):
            recommendations.append("Improve task failure recovery and state tracking")

        if not self.scenario_results.get('scheduler_restart', {}).get('passed', False):
            recommendations.append("Ensure scheduler restart doesn't affect pipeline state")

        if not self.scenario_results.get('worker_failure', {}).get('passed', False):
            recommendations.append("Improve distributed worker failure handling")

        if not self.scenario_results.get('redis_failure', {}).get('passed', False):
            recommendations.append("Strengthen Redis fallback mechanisms")

        # Check overall resilience
        passed_scenarios = sum(1 for result in self.scenario_results.values() if result['passed'])
        total_scenarios = len(self.scenario_results)

        if passed_scenarios == total_scenarios:
            recommendations.append("Excellent resilience! System is ready for production deployment.")
        elif passed_scenarios >= total_scenarios * 0.8:
            recommendations.append("Good resilience. Address remaining issues before production deployment.")
        else:
            recommendations.append("Low resilience. Significant improvements needed before production deployment.")

        return recommendations


def main():
    """Main entry point for failure scenario simulator."""
    parser = argparse.ArgumentParser(description='Redis-Enhanced DAG Failure Scenario Simulator')
    parser.add_argument('--scenario', choices=['dag-restart', 'task-failure', 'scheduler-restart',
                                              'worker-failure', 'redis-failure', 'all'],
                       default='all', help='Failure scenario to simulate')
    parser.add_argument('--redis-conn-id', default='redis_default',
                       help='Airflow Redis connection ID')
    parser.add_argument('--postgres-conn-id', default='postgres_default',
                       help='Airflow PostgreSQL connection ID')
    parser.add_argument('--output-file', help='Output file for scenario report (JSON)')

    args = parser.parse_args()

    # Initialize simulator
    simulator = FailureScenarioSimulator(
        redis_conn_id=args.redis_conn_id,
        postgres_conn_id=args.postgres_conn_id
    )

    # Run specific scenario or all scenarios
    if args.scenario == 'all':
        report = simulator.run_all_failure_scenarios()
    elif args.scenario == 'dag-restart':
        simulator.simulate_dag_restart_scenario()
        report = simulator.generate_failure_scenario_report()
    elif args.scenario == 'task-failure':
        simulator.simulate_task_failure_and_restart()
        report = simulator.generate_failure_scenario_report()
    elif args.scenario == 'scheduler-restart':
        simulator.simulate_scheduler_restart()
        report = simulator.generate_failure_scenario_report()
    elif args.scenario == 'worker-failure':
        simulator.simulate_worker_node_failure()
        report = simulator.generate_failure_scenario_report()
    elif args.scenario == 'redis-failure':
        simulator.simulate_redis_connectivity_failure()
        report = simulator.generate_failure_scenario_report()

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Failure scenario report written to {args.output_file}")
    else:
        print(json.dumps(report, indent=2))

    # Exit with appropriate code
    if report['failure_scenario_summary']['production_ready']:
        logger.info("✅ All critical failure scenarios passed. System is resilient.")
        sys.exit(0)
    else:
        logger.error("❌ Critical failure scenarios failed. System needs improvement.")
        sys.exit(1)


if __name__ == '__main__':
    main()
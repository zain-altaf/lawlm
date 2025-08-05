#!/usr/bin/env python3
"""
Pipeline Monitoring and Health Check Utilities

Provides monitoring capabilities for the legal document processing pipeline,
including memory usage, system health, and performance metrics.
"""

import os
import json
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from config import load_config, PipelineConfig

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources and pipeline health."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize system monitor."""
        self.config = config or load_config()
        self.start_time = time.time()
        self.monitoring_enabled = self.config.monitoring.enable_memory_monitoring
        
        # Track metrics over time
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000  # Keep last 1000 measurements
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        # Basic system info
        info = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'platform': os.name,
        }
        
        # CPU information
        info['cpu'] = {
            'count': psutil.cpu_count(),
            'usage_percent': psutil.cpu_percent(interval=1),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
        
        # Process-specific memory
        process = psutil.Process()
        process_memory = process.memory_info()
        info['process_memory'] = {
            'rss_mb': process_memory.rss / (1024**2),
            'vms_mb': process_memory.vms / (1024**2),
            'percent': process.memory_percent()
        }
        
        # GPU information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            info['gpu'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'memory_cached_mb': torch.cuda.memory_cached() / (1024**2) if hasattr(torch.cuda, 'memory_cached') else 0
            }
        else:
            info['gpu'] = {'available': False}
        
        # Disk usage for working directory
        if Path(self.config.processing.working_directory).exists():
            disk_usage = psutil.disk_usage(self.config.processing.working_directory)
            info['disk'] = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100
            }
        
        return info
    
    def check_qdrant_health(self) -> Dict[str, Any]:
        """Check Qdrant server health and collections."""
        if not QDRANT_AVAILABLE:
            return {'available': False, 'error': 'qdrant-client not installed'}
        
        try:
            client = QdrantClient(self.config.qdrant.url, timeout=5)
            
            # Get collections info
            collections = client.get_collections()
            collection_info = []
            
            for collection in collections.collections:
                try:
                    collection_details = client.get_collection(collection.name)
                    collection_info.append({
                        'name': collection.name,
                        'vectors_count': collection.vectors_count,
                        'points_count': collection.points_count,
                        'status': collection_details.status,
                        'indexed_vectors_count': getattr(collection_details, 'indexed_vectors_count', None)
                    })
                except Exception as e:
                    collection_info.append({
                        'name': collection.name,
                        'error': str(e)
                    })
            
            return {
                'available': True,
                'url': self.config.qdrant.url,
                'collections': collection_info,
                'total_collections': len(collections.collections)
            }
            
        except Exception as e:
            return {
                'available': False,
                'url': self.config.qdrant.url,
                'error': str(e)
            }
    
    def check_pipeline_files(self) -> Dict[str, Any]:
        """Check status of pipeline files and directories."""
        working_dir = Path(self.config.processing.working_directory)
        
        files_info = {
            'working_directory': {
                'path': str(working_dir),
                'exists': working_dir.exists(),
                'is_directory': working_dir.is_dir() if working_dir.exists() else False,
                'files': []
            }
        }
        
        if working_dir.exists():
            # List relevant files
            for file_pattern in ['*.json', '*.txt', '*.log']:
                for file_path in working_dir.glob(file_pattern):
                    try:
                        stat = file_path.stat()
                        files_info['working_directory']['files'].append({
                            'name': file_path.name,
                            'size_mb': stat.st_size / (1024**2),
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'type': file_path.suffix.lower()
                        })
                    except Exception as e:
                        files_info['working_directory']['files'].append({
                            'name': file_path.name,
                            'error': str(e)
                        })
        
        return files_info
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'issues': [],
            'warnings': []
        }\n        \n        # Check system resources\n        system_info = self.get_system_info()\n        \n        # Memory checks\n        if system_info['memory']['usage_percent'] > 90:\n            health['issues'].append('High system memory usage (>90%)')\n            health['overall_status'] = 'degraded'\n        elif system_info['memory']['usage_percent'] > 80:\n            health['warnings'].append('High system memory usage (>80%)')\n        \n        if system_info['process_memory']['rss_mb'] > self.config.monitoring.alert_memory_threshold_mb:\n            health['issues'].append(f\"Process memory usage exceeds threshold ({system_info['process_memory']['rss_mb']:.1f}MB)\")\n            health['overall_status'] = 'degraded'\n        \n        # GPU memory checks\n        if system_info['gpu']['available']:\n            gpu_usage = system_info['gpu']['memory_allocated_mb']\n            if gpu_usage > self.config.monitoring.alert_gpu_memory_threshold_mb:\n                health['issues'].append(f\"GPU memory usage exceeds threshold ({gpu_usage:.1f}MB)\")\n                health['overall_status'] = 'degraded'\n        \n        # CPU checks\n        if system_info['cpu']['usage_percent'] > 95:\n            health['warnings'].append('High CPU usage (>95%)')\n        \n        # Disk space checks\n        if 'disk' in system_info and system_info['disk']['usage_percent'] > 90:\n            health['issues'].append('Low disk space (<10% free)')\n            health['overall_status'] = 'degraded'\n        elif 'disk' in system_info and system_info['disk']['usage_percent'] > 80:\n            health['warnings'].append('Disk space running low (<20% free)')\n        \n        # Qdrant health check\n        qdrant_health = self.check_qdrant_health()\n        if not qdrant_health['available']:\n            health['issues'].append(f\"Qdrant unavailable: {qdrant_health.get('error', 'Unknown error')}\")\n            health['overall_status'] = 'degraded'\n        \n        # Set overall status\n        if health['issues']:\n            health['overall_status'] = 'unhealthy' if len(health['issues']) > 2 else 'degraded'\n        \n        return health\n    \n    def record_metrics(self) -> Dict[str, Any]:\n        \"\"\"Record current metrics and add to history.\"\"\"\n        if not self.monitoring_enabled:\n            return {}\n        \n        metrics = {\n            'timestamp': datetime.now().isoformat(),\n            'system': self.get_system_info(),\n            'qdrant': self.check_qdrant_health(),\n            'files': self.check_pipeline_files(),\n            'health': self.get_health_status()\n        }\n        \n        # Add to history\n        self.metrics_history.append(metrics)\n        \n        # Trim history if too large\n        if len(self.metrics_history) > self.max_history_size:\n            self.metrics_history = self.metrics_history[-self.max_history_size:]\n        \n        return metrics\n    \n    def get_metrics_summary(self, last_minutes: int = 60) -> Dict[str, Any]:\n        \"\"\"Get summary of metrics from the last N minutes.\"\"\"\n        if not self.metrics_history:\n            return {'error': 'No metrics recorded'}\n        \n        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)\n        recent_metrics = [\n            m for m in self.metrics_history \n            if datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00').replace('+00:00', '')) >= cutoff_time\n        ]\n        \n        if not recent_metrics:\n            return {'error': 'No recent metrics found'}\n        \n        # Calculate averages and trends\n        memory_usage = [m['system']['memory']['usage_percent'] for m in recent_metrics]\n        cpu_usage = [m['system']['cpu']['usage_percent'] for m in recent_metrics]\n        process_memory = [m['system']['process_memory']['rss_mb'] for m in recent_metrics]\n        \n        summary = {\n            'time_range_minutes': last_minutes,\n            'samples_count': len(recent_metrics),\n            'memory': {\n                'avg_usage_percent': sum(memory_usage) / len(memory_usage),\n                'max_usage_percent': max(memory_usage),\n                'min_usage_percent': min(memory_usage)\n            },\n            'cpu': {\n                'avg_usage_percent': sum(cpu_usage) / len(cpu_usage),\n                'max_usage_percent': max(cpu_usage),\n                'min_usage_percent': min(cpu_usage)\n            },\n            'process_memory': {\n                'avg_mb': sum(process_memory) / len(process_memory),\n                'max_mb': max(process_memory),\n                'min_mb': min(process_memory)\n            }\n        }\n        \n        return summary\n    \n    def save_metrics_report(self, output_file: str) -> None:\n        \"\"\"Save comprehensive metrics report to file.\"\"\"\n        report = {\n            'generated_at': datetime.now().isoformat(),\n            'config_summary': self.config.get_summary(),\n            'current_metrics': self.record_metrics(),\n            'metrics_summary_1h': self.get_metrics_summary(60),\n            'metrics_summary_24h': self.get_metrics_summary(1440),\n            'metrics_history_count': len(self.metrics_history)\n        }\n        \n        with open(output_file, 'w', encoding='utf-8') as f:\n            json.dump(report, f, indent=2)\n        \n        logger.info(f\"📊 Metrics report saved to: {output_file}\")\n\n\ndef main():\n    \"\"\"Command line interface for monitoring.\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description='Legal Document Pipeline Monitor')\n    parser.add_argument('--config', help='Configuration file path')\n    parser.add_argument('--health', action='store_true', help='Show health status')\n    parser.add_argument('--system', action='store_true', help='Show system info')\n    parser.add_argument('--qdrant', action='store_true', help='Show Qdrant status')\n    parser.add_argument('--files', action='store_true', help='Show file status')\n    parser.add_argument('--report', help='Generate comprehensive report to file')\n    parser.add_argument('--watch', type=int, metavar='SECONDS', \n                       help='Watch mode - continuously monitor every N seconds')\n    \n    args = parser.parse_args()\n    \n    # Load configuration\n    config = load_config(args.config)\n    monitor = SystemMonitor(config)\n    \n    if args.watch:\n        # Watch mode\n        print(f\"🔍 Monitoring every {args.watch} seconds (Ctrl+C to stop)...\")\n        try:\n            while True:\n                health = monitor.get_health_status()\n                timestamp = datetime.now().strftime('%H:%M:%S')\n                status_emoji = '✅' if health['overall_status'] == 'healthy' else '⚠️' if health['overall_status'] == 'degraded' else '❌'\n                print(f\"[{timestamp}] {status_emoji} Status: {health['overall_status']} - Issues: {len(health['issues'])}, Warnings: {len(health['warnings'])}\")\n                \n                if health['issues']:\n                    for issue in health['issues']:\n                        print(f\"  ❌ {issue}\")\n                if health['warnings']:\n                    for warning in health['warnings']:\n                        print(f\"  ⚠️ {warning}\")\n                \n                time.sleep(args.watch)\n        except KeyboardInterrupt:\n            print(\"\\n👋 Monitoring stopped\")\n            return\n    \n    if args.report:\n        monitor.save_metrics_report(args.report)\n        print(f\"📊 Report saved to: {args.report}\")\n        return\n    \n    # Show specific information\n    if args.health or not any([args.system, args.qdrant, args.files]):\n        health = monitor.get_health_status()\n        print(json.dumps(health, indent=2))\n    \n    if args.system:\n        system_info = monitor.get_system_info()\n        print(json.dumps(system_info, indent=2))\n    \n    if args.qdrant:\n        qdrant_info = monitor.check_qdrant_health()\n        print(json.dumps(qdrant_info, indent=2))\n    \n    if args.files:\n        files_info = monitor.check_pipeline_files()\n        print(json.dumps(files_info, indent=2))\n\n\nif __name__ == \"__main__\":\n    main()"
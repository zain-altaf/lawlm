"""
Batch Processing Utilities for Legal Document Pipeline

Provides utilities for robust batch processing with validation,
smart batch division, and progress tracking.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """Information about a processing batch."""
    batch_id: int
    size: int
    start_idx: int
    end_idx: int
    status: str = "pending"  # pending, processing, completed, failed
    created_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class BatchProcessor:
    """
    Manages batch processing with validation, progress tracking, and resume capability.
    """
    
    def __init__(self, working_dir: str = "data"):
        """Initialize batch processor."""
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.batch_state_file = self.working_dir / "batch_state.json"
        
    def validate_batch_parameters(self, num_items: int, batch_size: int) -> None:
        """
        Validate batch processing parameters.
        
        Args:
            num_items: Total number of items to process
            batch_size: Desired batch size
            
        Raises:
            ValueError: If parameters are invalid
        """
        if num_items < 1:
            raise ValueError(f"Number of items must be at least 1, got {num_items}")
        
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if batch_size > 10:
            raise ValueError("Maximum batch size is 10 to prevent memory issues")
        
        if batch_size > num_items:
            raise ValueError(f"Batch size ({batch_size}) cannot exceed total items ({num_items})")
    
    def calculate_batches(self, num_items: int, batch_size: int) -> List[BatchInfo]:
        """
        Calculate optimal batch divisions with smart handling of remainders.
        
        Args:
            num_items: Total number of items to process
            batch_size: Desired batch size
            
        Returns:
            List of BatchInfo objects describing each batch
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.validate_batch_parameters(num_items, batch_size)
        
        batches = []
        current_idx = 0
        batch_id = 0
        
        while current_idx < num_items:
            # Calculate size for this batch
            remaining_items = num_items - current_idx
            
            if remaining_items >= batch_size:
                # Full batch
                current_batch_size = batch_size
            else:
                # Remainder batch
                current_batch_size = remaining_items
            
            # Create batch info
            batch = BatchInfo(
                batch_id=batch_id,
                size=current_batch_size,
                start_idx=current_idx,
                end_idx=current_idx + current_batch_size
            )
            
            batches.append(batch)
            
            current_idx += current_batch_size
            batch_id += 1
        
        logger.info(f"📊 Created {len(batches)} batches for {num_items} items (batch_size={batch_size})")
        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i}: {batch.size} items (indices {batch.start_idx}-{batch.end_idx-1})")
        
        return batches
    
    def save_batch_state(self, 
                        job_id: str, 
                        batches: List[BatchInfo], 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save batch processing state to enable resume capability.
        
        Args:
            job_id: Unique identifier for this processing job
            batches: List of batch information
            metadata: Additional metadata about the job
        """
        state = {
            'job_id': job_id,
            'created_at': datetime.now().isoformat(),
            'total_batches': len(batches),
            'metadata': metadata or {},
            'batches': [
                {
                    'batch_id': b.batch_id,
                    'size': b.size,
                    'start_idx': b.start_idx,
                    'end_idx': b.end_idx,
                    'status': b.status,
                    'created_at': b.created_at,
                    'completed_at': b.completed_at,
                    'error_message': b.error_message
                }
                for b in batches
            ]
        }
        
        with open(self.batch_state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Saved batch state for job '{job_id}' to {self.batch_state_file}")
    
    def load_batch_state(self, job_id: str) -> Tuple[List[BatchInfo], Dict[str, Any]]:
        """
        Load batch processing state for resume capability.
        
        Args:
            job_id: Job identifier to load
            
        Returns:
            Tuple of (batches, metadata)
            
        Raises:
            FileNotFoundError: If no state file exists
            ValueError: If job_id doesn't match
        """
        if not self.batch_state_file.exists():
            raise FileNotFoundError(f"No batch state file found at {self.batch_state_file}")
        
        with open(self.batch_state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        if state['job_id'] != job_id:
            raise ValueError(f"Job ID mismatch: expected '{job_id}', found '{state['job_id']}'")
        
        # Reconstruct BatchInfo objects
        batches = []
        for b_data in state['batches']:
            batch = BatchInfo(
                batch_id=b_data['batch_id'],
                size=b_data['size'],
                start_idx=b_data['start_idx'],
                end_idx=b_data['end_idx'],
                status=b_data['status'],
                created_at=b_data['created_at'],
                completed_at=b_data['completed_at'],
                error_message=b_data['error_message']
            )
            batches.append(batch)
        
        logger.info(f"📂 Loaded batch state for job '{job_id}' with {len(batches)} batches")
        return batches, state['metadata']
    
    def update_batch_status(self, 
                           job_id: str, 
                           batch_id: int, 
                           status: str, 
                           error_message: str = "") -> None:
        """
        Update the status of a specific batch.
        
        Args:
            job_id: Job identifier
            batch_id: Batch identifier to update
            status: New status (pending, processing, completed, failed)
            error_message: Error message if status is 'failed'
        """
        try:
            batches, metadata = self.load_batch_state(job_id)
            
            # Find and update the batch
            for batch in batches:
                if batch.batch_id == batch_id:
                    batch.status = status
                    batch.error_message = error_message
                    if status in ['completed', 'failed']:
                        batch.completed_at = datetime.now().isoformat()
                    break
            else:
                logger.warning(f"Batch {batch_id} not found in job '{job_id}'")
                return
            
            # Save updated state
            self.save_batch_state(job_id, batches, metadata)
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to update batch status: {e}")
    
    def get_batch_progress(self, job_id: str) -> Dict[str, Any]:
        """
        Get progress information for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Progress information dictionary
        """
        try:
            batches, metadata = self.load_batch_state(job_id)
            
            status_counts = {}
            for batch in batches:
                status_counts[batch.status] = status_counts.get(batch.status, 0) + 1
            
            total_batches = len(batches)
            completed = status_counts.get('completed', 0)
            failed = status_counts.get('failed', 0)
            processing = status_counts.get('processing', 0)
            pending = status_counts.get('pending', 0)
            
            progress = {
                'job_id': job_id,
                'total_batches': total_batches,
                'completed': completed,
                'failed': failed,
                'processing': processing,
                'pending': pending,
                'completion_percentage': (completed / total_batches) * 100 if total_batches > 0 else 0,
                'status_counts': status_counts,
                'metadata': metadata
            }
            
            return progress
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to get batch progress: {e}")
            return {'error': str(e)}
    
    def get_pending_batches(self, job_id: str) -> List[BatchInfo]:
        """
        Get list of batches that still need processing.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of pending BatchInfo objects
        """
        try:
            batches, _ = self.load_batch_state(job_id)
            pending_batches = [b for b in batches if b.status in ['pending', 'failed']]
            return pending_batches
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to get pending batches: {e}")
            return []
    
    def cleanup_batch_files(self, job_id: str, keep_final: bool = True) -> None:
        """
        Clean up intermediate batch files.
        
        Args:
            job_id: Job identifier
            keep_final: Whether to keep the final merged files
        """
        # Look for files matching batch patterns
        batch_patterns = [
            f"batch_{job_id}_*.json",
            f"*_batch_{job_id}_*.json",
            f"{job_id}_batch_*.json"
        ]
        
        removed_count = 0
        for pattern in batch_patterns:
            for file_path in self.working_dir.glob(pattern):
                if keep_final and 'final' in file_path.name.lower():
                    continue
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} batch files for job '{job_id}'")


def create_job_id(court: str, num_dockets: int, timestamp: Optional[str] = None) -> str:
    """
    Create a unique job identifier for batch processing.
    
    Args:
        court: Court identifier
        num_dockets: Number of dockets
        timestamp: Optional timestamp (uses current time if not provided)
        
    Returns:
        Unique job identifier
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{court}_{num_dockets}dockets_{timestamp}"


def merge_json_files(input_files: List[str], output_file: str) -> Dict[str, Any]:
    """
    Merge multiple JSON files containing lists into a single file.
    
    Args:
        input_files: List of JSON file paths to merge
        output_file: Output file path
        
    Returns:
        Summary statistics about the merge operation
    """
    all_data = []
    stats = {
        'input_files': len(input_files),
        'total_items': 0,
        'files_processed': 0,
        'files_failed': 0,
        'output_file': output_file
    }
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_data.extend(data)
                stats['total_items'] += len(data)
            else:
                logger.warning(f"File {file_path} does not contain a list, skipping")
                
            stats['files_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            stats['files_failed'] += 1
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📋 Merged {stats['files_processed']} files into {output_file}")
    logger.info(f"📊 Total items: {stats['total_items']}")
    
    return stats


if __name__ == "__main__":
    # Test the batch utilities
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Processing Utilities")
    parser.add_argument('--test-batches', type=int, nargs=2, metavar=('ITEMS', 'BATCH_SIZE'),
                       help='Test batch calculation with given items and batch size')
    parser.add_argument('--job-progress', help='Show progress for job ID')
    parser.add_argument('--cleanup-job', help='Clean up files for job ID')
    
    args = parser.parse_args()
    
    processor = BatchProcessor()
    
    if args.test_batches:
        items, batch_size = args.test_batches
        try:
            batches = processor.calculate_batches(items, batch_size)
            print(f"✅ Successfully calculated {len(batches)} batches")
            for batch in batches:
                print(f"  Batch {batch.batch_id}: {batch.size} items")
        except ValueError as e:
            print(f"❌ Error: {e}")
    
    if args.job_progress:
        progress = processor.get_batch_progress(args.job_progress)
        print(json.dumps(progress, indent=2))
    
    if args.cleanup_job:
        processor.cleanup_batch_files(args.cleanup_job)
        print(f"✅ Cleaned up files for job: {args.cleanup_job}")
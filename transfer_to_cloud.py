#!/usr/bin/env python3
"""
Transfer vectors from local Qdrant to cloud Qdrant with duplicate detection and validation.
"""
import os
from typing import Set, List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from datetime import datetime
import logging
import json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantTransfer:
    def __init__(self):
        # Local Qdrant
        self.local_client = QdrantClient(url="http://localhost:6333")
        
        # Cloud Qdrant
        self.cloud_url = os.getenv("QDRANT_URL_CLOUD")
        self.cloud_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.cloud_url or not self.cloud_api_key:
            raise ValueError("Missing QDRANT_URL_CLOUD or QDRANT_API_KEY in .env file")
            
        self.cloud_client = QdrantClient(
            url=self.cloud_url,
            api_key=self.cloud_api_key
        )
        
        self.collection_name = "caselaw-chunks-hybrid"
        
    def get_existing_document_ids_from_cloud(self) -> Set[str]:
        """Get existing document IDs from cloud collection to prevent duplicates."""
        try:
            if not self.cloud_client.collection_exists(collection_name=self.collection_name):
                logger.info("📋 Cloud collection doesn't exist - all vectors will be new")
                return set()
                
            existing_ids = set()
            offset = None
            
            logger.info("🔍 Scanning cloud collection for existing document IDs...")
            
            while True:
                points, next_offset = self.cloud_client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                    
                for point in points:
                    doc_id = point.payload.get('document_id')
                    if doc_id:
                        existing_ids.add(doc_id)
                        
                offset = next_offset
                if next_offset is None:
                    break
                    
            logger.info(f"📊 Found {len(existing_ids)} existing document IDs in cloud")
            return existing_ids
            
        except Exception as e:
            logger.error(f"Error getting existing IDs from cloud: {e}")
            return set()
    
    def validate_collections(self) -> Dict[str, Any]:
        """Validate both local and cloud collections before transfer."""
        validation = {
            'local_exists': False,
            'local_count': 0,
            'cloud_exists': False,
            'cloud_count': 0,
            'ready_for_transfer': False
        }
        
        try:
            # Check local collection
            if self.local_client.collection_exists(collection_name=self.collection_name):
                validation['local_exists'] = True
                local_info = self.local_client.get_collection(collection_name=self.collection_name)
                validation['local_count'] = local_info.points_count
                logger.info(f"✅ Local collection: {validation['local_count']} vectors")
            else:
                logger.error("❌ Local collection doesn't exist")
                return validation
                
            # Check cloud collection (create if doesn't exist)
            if self.cloud_client.collection_exists(collection_name=self.collection_name):
                validation['cloud_exists'] = True
                cloud_info = self.cloud_client.get_collection(collection_name=self.collection_name)
                validation['cloud_count'] = cloud_info.points_count
                logger.info(f"✅ Cloud collection: {validation['cloud_count']} vectors")
            else:
                logger.info("📋 Cloud collection doesn't exist - will create it")
                # Create collection in cloud with same config as local
                local_config = self.local_client.get_collection(collection_name=self.collection_name)
                self.cloud_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=local_config.config.params.vectors.size,
                        distance=local_config.config.params.vectors.distance
                    )
                )
                validation['cloud_exists'] = True
                validation['cloud_count'] = 0
                logger.info("✅ Cloud collection created")
                
            validation['ready_for_transfer'] = True
            return validation
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return validation
    
    def transfer_vectors(self, batch_size: int = 100, dry_run: bool = False) -> Dict[str, Any]:
        """
        Transfer vectors from local to cloud with duplicate detection.
        
        Args:
            batch_size: Number of vectors to transfer per batch
            dry_run: If True, only simulate the transfer
        """
        logger.info(f"🚀 Starting vector transfer (dry_run={dry_run})")
        start_time = datetime.now()
        
        # Validate collections
        validation = self.validate_collections()
        if not validation['ready_for_transfer']:
            raise RuntimeError("Collections not ready for transfer")
            
        # Get existing document IDs from cloud to prevent duplicates
        existing_cloud_ids = self.get_existing_document_ids_from_cloud()
        
        # Get all vectors from local collection
        logger.info("📥 Fetching vectors from local collection...")
        local_vectors = []
        offset = None
        
        while True:
            points, next_offset = self.local_client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                break
                
            local_vectors.extend(points)
            offset = next_offset
            if next_offset is None:
                break
                
        logger.info(f"📊 Retrieved {len(local_vectors)} vectors from local")
        
        # Filter out duplicates
        new_vectors = []
        duplicates_skipped = 0
        
        for vector in local_vectors:
            doc_id = vector.payload.get('document_id')
            if doc_id and doc_id in existing_cloud_ids:
                duplicates_skipped += 1
            else:
                new_vectors.append(vector)
                
        logger.info(f"📊 Transfer analysis:")
        logger.info(f"   Total local vectors: {len(local_vectors)}")
        logger.info(f"   New vectors to transfer: {len(new_vectors)}")
        logger.info(f"   Duplicates skipped: {duplicates_skipped}")
        
        if dry_run:
            logger.info("🧪 DRY RUN - No actual transfer performed")
            return {
                'transferred': 0,
                'duplicates_skipped': duplicates_skipped,
                'total_local': len(local_vectors),
                'would_transfer': len(new_vectors),
                'dry_run': True
            }
            
        if not new_vectors:
            logger.info("✅ No new vectors to transfer - cloud is up to date")
            return {
                'transferred': 0,
                'duplicates_skipped': duplicates_skipped,
                'total_local': len(local_vectors),
                'message': 'Already up to date'
            }
            
        # Transfer in batches
        transferred = 0
        failed_batches = 0
        
        logger.info(f"⬆️ Starting transfer of {len(new_vectors)} vectors...")
        
        for i in range(0, len(new_vectors), batch_size):
            batch = new_vectors[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(new_vectors) + batch_size - 1) // batch_size
            
            try:
                logger.info(f"📤 Transferring batch {batch_num}/{total_batches} ({len(batch)} vectors)")
                
                self.cloud_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
                transferred += len(batch)
                
                if batch_num % 5 == 0:  # Progress update every 5 batches
                    logger.info(f"✅ Progress: {transferred}/{len(new_vectors)} vectors transferred")
                    
            except Exception as e:
                logger.error(f"❌ Failed to transfer batch {batch_num}: {e}")
                failed_batches += 1
                
        duration = (datetime.now() - start_time).total_seconds()
        
        # Final validation
        final_cloud_info = self.cloud_client.get_collection(collection_name=self.collection_name)
        final_cloud_count = final_cloud_info.points_count
        
        result = {
            'transferred': transferred,
            'duplicates_skipped': duplicates_skipped,
            'total_local': len(local_vectors),
            'failed_batches': failed_batches,
            'duration_seconds': duration,
            'final_cloud_count': final_cloud_count,
            'success': failed_batches == 0
        }
        
        logger.info(f"🎉 Transfer completed!")
        logger.info(f"   Vectors transferred: {transferred}")
        logger.info(f"   Duplicates skipped: {duplicates_skipped}")
        logger.info(f"   Failed batches: {failed_batches}")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   Final cloud count: {final_cloud_count}")
        
        return result
    
    def compare_collections(self) -> Dict[str, Any]:
        """Compare local and cloud collections for verification."""
        try:
            local_info = self.local_client.get_collection(collection_name=self.collection_name)
            cloud_info = self.cloud_client.get_collection(collection_name=self.collection_name)
            
            comparison = {
                'local_count': local_info.points_count,
                'cloud_count': cloud_info.points_count,
                'difference': abs(local_info.points_count - cloud_info.points_count),
                'in_sync': local_info.points_count <= cloud_info.points_count  # Cloud can have more
            }
            
            logger.info(f"📊 Collection comparison:")
            logger.info(f"   Local: {comparison['local_count']} vectors")
            logger.info(f"   Cloud: {comparison['cloud_count']} vectors")
            logger.info(f"   Status: {'✅ In sync' if comparison['in_sync'] else '⚠️ Out of sync'}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing collections: {e}")
            return {}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer vectors from local to cloud Qdrant")
    parser.add_argument('--dry-run', action='store_true', help='Simulate transfer without actually doing it')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for transfer')
    parser.add_argument('--compare', action='store_true', help='Compare collections without transfer')
    
    args = parser.parse_args()
    
    try:
        transfer = QdrantTransfer()
        
        if args.compare:
            transfer.compare_collections()
        else:
            result = transfer.transfer_vectors(
                batch_size=args.batch_size,
                dry_run=args.dry_run
            )
            
            # Save result to file
            result_file = f"transfer_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"📄 Transfer result saved to {result_file}")
            
    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        raise

if __name__ == "__main__":
    main()
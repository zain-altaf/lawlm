#!/usr/bin/env python3
"""
Qdrant Collection Migration Utility

This script helps migrate collections from local Qdrant to Qdrant Cloud
or between different Qdrant instances. It exports collections to JSON files
and can import them to a target Qdrant instance.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantMigrationTool:
    """Tool for migrating Qdrant collections between instances."""
    
    def __init__(self, 
                 source_url: str = "http://localhost:6333",
                 source_api_key: Optional[str] = None,
                 target_url: Optional[str] = None,
                 target_api_key: Optional[str] = None):
        """
        Initialize migration tool.
        
        Args:
            source_url: Source Qdrant URL
            source_api_key: Source API key (if needed)
            target_url: Target Qdrant URL (defaults to cloud from env)
            target_api_key: Target API key (defaults to env)
        """
        self.source_url = source_url
        self.source_api_key = source_api_key
        
        # Default target to cloud from environment
        self.target_url = target_url or os.getenv("QDRANT_URL", source_url)
        self.target_api_key = target_api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize clients
        self.source_client = self._create_client(self.source_url, self.source_api_key)
        self.target_client = self._create_client(self.target_url, self.target_api_key)
        
        logger.info(f"📤 Source: {self.source_url}")
        logger.info(f"📥 Target: {self.target_url}")
    
    def _create_client(self, url: str, api_key: Optional[str]) -> QdrantClient:
        """Create Qdrant client with optional API key."""
        try:
            if api_key:
                client = QdrantClient(url=url, api_key=api_key, timeout=60)
                client_type = "cloud" if "cloud.qdrant.io" in url else "authenticated"
            else:
                client = QdrantClient(url, timeout=60)
                client_type = "local"
            
            # Test connection
            collections = client.get_collections()
            logger.info(f"✅ Connected to {client_type} Qdrant at {url}")
            logger.info(f"   Found {len(collections.collections)} collections")
            
            return client
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant at {url}: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error("   Authentication failed. Check your API key.")
            raise
    
    def list_collections(self, client: QdrantClient, client_name: str) -> List[str]:
        """List all collections in a Qdrant instance."""
        try:
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            logger.info(f"📋 Collections in {client_name}:")
            for i, name in enumerate(collection_names, 1):
                collection_info = client.get_collection(name)
                logger.info(f"   {i}. {name} ({collection_info.points_count} points)")
            
            return collection_names
            
        except Exception as e:
            logger.error(f"❌ Failed to list collections from {client_name}: {e}")
            return []
    
    def export_collection(self, 
                         collection_name: str,
                         output_file: Optional[str] = None,
                         batch_size: int = 100) -> str:
        """
        Export a collection to JSON file.
        
        Args:
            collection_name: Name of collection to export
            output_file: Output filename (auto-generated if None)
            batch_size: Number of points to fetch per batch
            
        Returns:
            Path to exported file
        """
        logger.info(f"📤 Exporting collection '{collection_name}'")
        
        try:
            # Get collection info
            collection_info = self.source_client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            if total_points == 0:
                logger.warning(f"⚠️ Collection '{collection_name}' is empty")
                return ""
            
            logger.info(f"   Total points to export: {total_points}")
            
            # Generate output filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"collection_{collection_name}_{timestamp}.json"
            
            output_path = Path(output_file)
            
            # Export collection metadata
            export_data = {
                "collection_name": collection_name,
                "exported_at": datetime.now().isoformat(),
                "source_url": self.source_url,
                "total_points": total_points,
                "collection_config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value
                },
                "points": []
            }
            
            # Export points in batches
            exported_count = 0
            offset = None
            
            while exported_count < total_points:
                # Fetch batch of points
                points_batch, next_offset = self.source_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not points_batch:
                    break
                
                # Convert points to exportable format
                for point in points_batch:
                    point_data = {
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload or {}
                    }
                    export_data["points"].append(point_data)
                
                exported_count += len(points_batch)
                offset = next_offset
                
                # Progress update
                progress = (exported_count / total_points) * 100
                logger.info(f"   Exported {exported_count}/{total_points} points ({progress:.1f}%)")
                
                # Break if no more points
                if next_offset is None:
                    break
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Exported {exported_count} points to {output_path}")
            logger.info(f"   File size: {file_size_mb:.2f}MB")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to export collection '{collection_name}': {e}")
            raise
    
    def import_collection(self, 
                         export_file: str,
                         target_collection_name: Optional[str] = None,
                         batch_size: int = 100,
                         overwrite: bool = False) -> str:
        """
        Import a collection from JSON file.
        
        Args:
            export_file: Path to exported JSON file
            target_collection_name: Name for imported collection (defaults to original)
            batch_size: Number of points to upload per batch
            overwrite: Whether to overwrite existing collection
            
        Returns:
            Name of created collection
        """
        logger.info(f"📥 Importing collection from {export_file}")
        
        try:
            # Load export data
            with open(export_file, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            
            collection_name = target_collection_name or export_data["collection_name"]
            points_data = export_data["points"]
            total_points = len(points_data)
            
            logger.info(f"   Collection: {collection_name}")
            logger.info(f"   Points to import: {total_points}")
            
            # Check if collection exists
            try:
                existing_collection = self.target_client.get_collection(collection_name)
                if overwrite:
                    logger.info(f"🗑️ Deleting existing collection '{collection_name}'")
                    self.target_client.delete_collection(collection_name)
                else:
                    logger.error(f"❌ Collection '{collection_name}' already exists. Use --overwrite to replace.")
                    return ""
            except Exception:
                # Collection doesn't exist, which is fine
                pass
            
            # Create collection
            config = export_data["collection_config"]
            distance_map = {
                "Cosine": Distance.COSINE,
                "Dot": Distance.DOT,
                "Euclid": Distance.EUCLID
            }
            
            self.target_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config["vector_size"],
                    distance=distance_map.get(config["distance"], Distance.COSINE)
                )
            )
            logger.info(f"✅ Created collection '{collection_name}'")
            
            # Import points in batches
            imported_count = 0
            
            for i in range(0, total_points, batch_size):
                batch = points_data[i:i + batch_size]
                
                # Convert to Qdrant points format
                points = []
                for point_data in batch:
                    points.append(models.PointStruct(
                        id=point_data["id"],
                        vector=point_data["vector"],
                        payload=point_data["payload"]
                    ))
                
                # Upload batch
                self.target_client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                
                imported_count += len(batch)
                progress = (imported_count / total_points) * 100
                logger.info(f"   Imported {imported_count}/{total_points} points ({progress:.1f}%)")
            
            logger.info(f"✅ Successfully imported {imported_count} points to '{collection_name}'")
            
            # Verify import
            final_collection = self.target_client.get_collection(collection_name)
            logger.info(f"   Final collection size: {final_collection.points_count} points")
            
            return collection_name
            
        except Exception as e:
            logger.error(f"❌ Failed to import collection: {e}")
            raise
    
    def migrate_collection(self, 
                          collection_name: str,
                          target_name: Optional[str] = None,
                          batch_size: int = 100,
                          keep_export: bool = True) -> str:
        """
        Migrate a collection directly from source to target.
        
        Args:
            collection_name: Name of collection to migrate
            target_name: Name for migrated collection (defaults to original)
            batch_size: Batch size for transfer
            keep_export: Whether to keep the temporary export file
            
        Returns:
            Name of migrated collection
        """
        logger.info(f"🚀 Migrating collection '{collection_name}' from source to target")
        
        try:
            # Export collection
            export_file = self.export_collection(collection_name, batch_size=batch_size)
            if not export_file:
                return ""
            
            # Import to target
            result_name = self.import_collection(
                export_file, 
                target_name, 
                batch_size=batch_size,
                overwrite=True
            )
            
            # Clean up export file if requested
            if not keep_export and export_file:
                Path(export_file).unlink()
                logger.info(f"🗑️ Cleaned up export file: {export_file}")
            
            logger.info(f"🎉 Successfully migrated '{collection_name}' to '{result_name}'")
            return result_name
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise


def main():
    """Command line interface for collection migration."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collections between instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List collections in local instance
  python migrate_to_cloud.py --list-source
  
  # List collections in cloud instance  
  python migrate_to_cloud.py --list-target
  
  # Export a collection to file
  python migrate_to_cloud.py --export caselaw-cases --output backup.json
  
  # Import collection from file
  python migrate_to_cloud.py --import backup.json --overwrite
  
  # Migrate collection directly from local to cloud
  python migrate_to_cloud.py --migrate caselaw-cases
  
  # Migrate all collections
  python migrate_to_cloud.py --migrate-all
        """
    )
    
    # Connection settings
    parser.add_argument('--source-url', default='http://localhost:6333',
                       help='Source Qdrant URL')
    parser.add_argument('--source-api-key', 
                       help='Source API key (if needed)')
    parser.add_argument('--target-url',
                       help='Target Qdrant URL (defaults to QDRANT_URL env var)')
    parser.add_argument('--target-api-key',
                       help='Target API key (defaults to QDRANT_API_KEY env var)')
    
    # Operations
    parser.add_argument('--list-source', action='store_true',
                       help='List collections in source instance')
    parser.add_argument('--list-target', action='store_true',
                       help='List collections in target instance')
    parser.add_argument('--export',
                       help='Export collection to file')
    parser.add_argument('--import',
                       help='Import collection from file')
    parser.add_argument('--migrate',
                       help='Migrate single collection from source to target')
    parser.add_argument('--migrate-all', action='store_true',
                       help='Migrate all collections from source to target')
    
    # Options
    parser.add_argument('--output',
                       help='Output filename for export')
    parser.add_argument('--target-name',
                       help='Target collection name (for import/migrate)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for operations')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing collections')
    parser.add_argument('--keep-exports', action='store_true',
                       help='Keep temporary export files')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize migration tool
        migrator = QdrantMigrationTool(
            source_url=args.source_url,
            source_api_key=args.source_api_key,
            target_url=args.target_url,
            target_api_key=args.target_api_key
        )
        
        # Execute operations
        if args.list_source:
            migrator.list_collections(migrator.source_client, "source")
            
        elif args.list_target:
            migrator.list_collections(migrator.target_client, "target")
            
        elif args.export:
            export_file = migrator.export_collection(
                args.export, 
                args.output, 
                args.batch_size
            )
            print(f"Exported to: {export_file}")
            
        elif getattr(args, 'import'):
            collection_name = migrator.import_collection(
                getattr(args, 'import'),
                args.target_name,
                args.batch_size,
                args.overwrite
            )
            print(f"Imported as: {collection_name}")
            
        elif args.migrate:
            collection_name = migrator.migrate_collection(
                args.migrate,
                args.target_name,
                args.batch_size,
                args.keep_exports
            )
            print(f"Migrated as: {collection_name}")
            
        elif args.migrate_all:
            source_collections = migrator.list_collections(migrator.source_client, "source")
            
            if not source_collections:
                print("No collections found in source instance")
                return
            
            print(f"\n🚀 Migrating {len(source_collections)} collections...")
            
            for collection_name in source_collections:
                try:
                    migrated_name = migrator.migrate_collection(
                        collection_name,
                        batch_size=args.batch_size,
                        keep_export=args.keep_exports
                    )
                    print(f"✅ Migrated: {collection_name} -> {migrated_name}")
                except Exception as e:
                    print(f"❌ Failed to migrate {collection_name}: {e}")
                    
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n❌ Migration interrupted by user")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
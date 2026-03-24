import os
from datetime import datetime
from typing import Optional, Set, Tuple
from qdrant_client import QdrantClient, models
import logging

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, collection_name: str, vector_size: int = 768, url=None):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.client = QdrantClient(url=self.url)


    def get_or_create_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    'bge-small': models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    'bm25': models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            logger.info(f"Qdrant collection exists: {self.collection_name}") 


    def upsert(self, collection_name: str, points: list):
        """Delegates the upsert call to the underlying QdrantClient."""
        return self.client.upsert(
            collection_name=collection_name,
            points=points
        )


    def get_existing_ids_and_cursor(self) -> Tuple[Set[str], Optional[str]]:
        """
        Retrieve all unique docket_ids and the most recent cursor from a Qdrant collection.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the collection

        Returns:
            tuple[set, str | None]: (set of unique docket_ids, most recent cursor string)
        """
        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Could not access collection '{self.collection_name}': {e}")
            return set(), None

        if collection.points_count == 0:
            logger.info("Collection empty — no docket IDs yet.")
            return set(), None

        unique_docket_ids = set()
        most_recent_point = None
        most_recent_time = None
        next_page = None

        while True:
            try:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    with_payload=True,
                    limit=1000,
                    offset=next_page
                )
            except Exception as e:
                logger.error(f"Error while scrolling: {e}")
                break

            for point in points:
                payload = point.payload or {}

                docket_id = payload.get("docket_id")
                if docket_id:
                    unique_docket_ids.add(docket_id)

                # Parse `time_processed`
                time_str = payload.get("time_processed")
                if time_str:
                    try:
                        dt = datetime.strptime(time_str, "%d-%m-%y %H:%M:%S")
                        if most_recent_time is None or dt > most_recent_time:
                            most_recent_time = dt
                            most_recent_point = payload
                    except ValueError:
                        pass

            if not next_page:
                break

        latest_cursor = most_recent_point.get("cursor") if most_recent_point else None

        return unique_docket_ids, latest_cursor
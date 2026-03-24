class Opinion:
    def __init__(
        self,
        docket_id: str,
        cluster_id: str,
        opinion_id: str,
        court: str,
        date_filed: str,
        raw_text: str,
        cluster_data: dict,
        opinion_data: dict,
        source_field: str,
        processed: dict,
        next_cursor: str | None,
    ):
        # Basic metadata
        self.docket_id = docket_id
        self.cluster_id = cluster_id
        self.opinion_id = opinion_id
        self.court = court
        self.date_filed = date_filed
        self.raw_text = raw_text

        # Cluster-level metadata
        self.judges = cluster_data.get("judges", "")
        self.date_filed_cluster = cluster_data.get("date_filed")
        self.precedential_status = cluster_data.get("precedential_status")

        # Opinion-level metadata
        self.author = opinion_data.get("author_id", "")
        self.opinion_type = opinion_data.get("type", "unknown")
        self.sha1 = opinion_data.get("sha1")
        self.download_url = opinion_data.get("download_url")
        self.date_created = opinion_data.get("date_created")
        self.date_modified = opinion_data.get("date_modified")

        # Processing results
        self.opinion_text = processed.get("cleaned_text", "")
        self.citations = processed.get("citations", [])
        self.legal_entities = processed.get("legal_entities", [])
        self.text_stats = processed.get("text_stats", {})

        # Other fields
        self.source_field = source_field
        self.cursor = next_cursor
    
    def chunk_metadata(self) -> dict:
        """Returns only the essential metadata, excluding large text blocks."""
        return {
            "docket_id": self.docket_id,
            "cluster_id": self.cluster_id,
            "opinion_id": self.opinion_id,
            "court": self.court,
            "date_filed": self.date_filed,
            "judges": self.judges,
            "precedential_status": self.precedential_status,
            "author": self.author,
            "opinion_type": self.opinion_type,
            "sha1": self.sha1,
            "download_url": self.download_url,
            "date_created": self.date_created,
            "date_modified": self.date_modified,
            "citations": self.citations,
            "legal_entities": self.legal_entities,
            "text_stats": self.text_stats,
            "source_field": self.source_field,
        }
from opinion import Opinion


class Chunk:
    def __init__(self, opinion: Opinion, chunk_index: int, text: str):
        self.opinion = opinion
        self.chunk_index = chunk_index
        self.text = text

    @property
    def id(self) -> str:
        return f"{self.opinion.docket_id}_{self.opinion.opinion_id}_{self.chunk_index}"


    def to_dict(self) -> dict:
        # This now only spreads the filtered metadata from the updated method above
        return {
            "id": self.id,
            "chunk_id": f"{self.opinion.opinion_id}_{self.chunk_index}",
            "chunk_index": self.chunk_index,
            "text": self.text,  # Only the small text segment for THIS chunk
            **self.opinion.chunk_metadata(),
        }

"""
Recursive Character Text Splitter for Legal Documents.

This module provides chunking of legal documents using recursive character splitting
with configurable overlap, optimized for legal text processing.
"""

import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecursiveCharacterTextSplitter:
    """
    Recursive character text splitter that creates chunks with overlap
    optimized for legal documents.
    """
    
    def __init__(self, 
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 separators: Optional[List[str]] = None,
                 quality_threshold: float = 0.3):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Maximum size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
            separators: List of separators to use for splitting (legal-optimized if None)
            quality_threshold: Minimum quality score for chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.quality_threshold = quality_threshold
        
        # Legal document separators in priority order
        if separators is None:
            self.separators = [
                "\n\n\n",  # Triple line breaks (major sections)
                "\n\n",    # Double line breaks (paragraphs)
                "\n",      # Single line breaks
                ". ",      # Sentence endings
                "; ",      # Clause separators
                ", ",      # Phrase separators
                " "        # Word separators
            ]
        else:
            self.separators = separators
        
        logger.info(f"Initialized RecursiveCharacterTextSplitter")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        logger.info(f"Separators: {len(self.separators)} configured")
    
    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the provided separators."""
        if not text or len(text.strip()) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                
                # Reconstruct chunks while preserving separator
                chunks = []
                current_chunk = ""
                
                for i, split in enumerate(splits):
                    # Add separator back (except for last split)
                    if i < len(splits) - 1:
                        split_with_sep = split + separator
                    else:
                        split_with_sep = split
                    
                    # Check if adding this split would exceed chunk size
                    if len(current_chunk + split_with_sep) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = split_with_sep
                    else:
                        current_chunk += split_with_sep
                
                # Add the last chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Recursively split chunks that are still too large
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size:
                        # Use remaining separators for recursive splitting
                        remaining_separators = separators[separators.index(separator) + 1:]
                        if remaining_separators:
                            final_chunks.extend(self._split_text_recursively(chunk, remaining_separators))
                        else:
                            # Force split if no separators left
                            final_chunks.extend(self._force_split(chunk))
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks
        
        # If no separators found, force split
        return self._force_split(text)
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text into chunks of specified size."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks
    
    def _create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """Create overlapping chunks from the split text."""
        if not chunks or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no prefix overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                
                # Clean up overlap boundary (try to break at word boundary)
                if ' ' in overlap_text:
                    words = overlap_text.split(' ')
                    if len(words) > 1:
                        overlap_text = ' '.join(words[1:])  # Remove partial first word
                
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive character splitting with overlap."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # First, split text recursively using separators
        chunks = self._split_text_recursively(text, self.separators)
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= self.min_chunk_size]
        
        # Create overlapping chunks
        overlapped_chunks = self._create_overlapping_chunks(chunks)
        
        return overlapped_chunks
    
    def _validate_chunk_quality(self, chunk: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate chunk quality based on various criteria.
        
        Args:
            chunk: Chunk dictionary with text and metadata
            
        Returns:
            Dictionary of validation results
        """
        text = chunk.get('text', '')
        
        validation = {
            'has_sufficient_content': len(text.strip()) >= self.min_chunk_size,
            'has_complete_sentences': text.strip().endswith(('.', '!', '?', ';')),
            'not_just_citations': len(re.findall(r'[A-Za-z]{3,}', text)) > 5,  # Has substantial text beyond citations
            'reasonable_length': len(text.split()) >= 20,  # At least 20 words
            'no_excessive_repetition': self._check_repetition(text),
            'contains_legal_content': self._has_legal_indicators(text)
        }
        
        # Overall quality score
        validation['quality_score'] = sum(validation.values()) / len(validation)
        validation['passes_quality_check'] = validation['quality_score'] >= self.quality_threshold
        
        return validation
    
    def _check_repetition(self, text: str) -> bool:
        """Check if text has excessive repetition."""
        words = text.lower().split()
        if len(words) < 10:
            return True  # Too short to assess repetition
        
        # Check for excessive word repetition
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only count substantial words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Flag if any word appears more than 30% of the time
        max_repetition = max(word_counts.values()) if word_counts else 0
        repetition_ratio = max_repetition / len(words)
        
        return repetition_ratio < 0.3
    
    def _has_legal_indicators(self, text: str) -> bool:
        """Check if text contains legal language indicators."""
        legal_indicators = [
            # Common legal words
            'court', 'judge', 'plaintiff', 'defendant', 'case', 'law', 'legal',
            'statute', 'constitutional', 'jurisdiction', 'appeal', 'ruling',
            'decision', 'opinion', 'brief', 'motion', 'evidence', 'testimony',
            # Legal concepts
            'due process', 'equal protection', 'probable cause', 'reasonable',
            'standard', 'burden', 'proof', 'liability', 'damages', 'remedy',
            # Legal actions/verbs
            'held', 'ruled', 'decided', 'concluded', 'found', 'determined',
            'granted', 'denied', 'affirmed', 'reversed', 'remanded'
        ]
        
        text_lower = text.lower()
        legal_word_count = sum(1 for indicator in legal_indicators if indicator in text_lower)
        
        # Should have at least 2 legal indicators in a chunk
        return legal_word_count >= 2
    
    def _calculate_chunk_metadata(self, chunk: Dict[str, Any], document: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metadata for a chunk including legal importance scoring."""
        text = chunk['text']
        
        # Legal importance indicators
        legal_keywords = [
            'holding', 'rule', 'standard', 'test', 'doctrine', 'precedent', 'overrule',
            'distinguish', 'apply', 'interpret', 'construe', 'conclude', 'find', 'held',
            'established', 'principle', 'factor', 'element', 'requirement', 'burden',
            'constitutional', 'statutory', 'regulation', 'due process', 'equal protection'
        ]
        
        # Count legal keywords (case insensitive)
        keyword_count = sum(len(re.findall(r'\b' + keyword + r'\b', text, re.IGNORECASE)) for keyword in legal_keywords)
        
        # Count citations in this chunk
        citations_in_chunk = [cite for cite in document.get('citations', []) if cite in text]
        
        # Calculate legal importance score
        importance_score = (
            keyword_count * 0.3 +  # Legal terminology weight
            len(citations_in_chunk) * 0.4 +  # Citation density weight
            (1 if any(word in text.lower() for word in ['holding', 'rule', 'conclude', 'held']) else 0) * 0.3  # Key legal concepts
        )
        
        # Identify semantic topic (simplified classification)
        topic_keywords = {
            'procedural': ['procedure', 'jurisdiction', 'standing', 'venue', 'discovery', 'motion', 'appeal'],
            'substantive': ['damages', 'liability', 'breach', 'negligence', 'contract', 'tort', 'constitutional'],
            'evidentiary': ['evidence', 'testimony', 'witness', 'hearsay', 'relevance', 'admissible'],
            'remedial': ['remedy', 'injunction', 'damages', 'restitution', 'specific performance']
        }
        
        semantic_topic = 'general'
        max_topic_score = 0
        
        for topic, keywords in topic_keywords.items():
            topic_score = sum(len(re.findall(r'\b' + keyword + r'\b', text, re.IGNORECASE)) for keyword in keywords)
            if topic_score > max_topic_score:
                max_topic_score = topic_score
                semantic_topic = topic
        
        return {
            'legal_importance_score': importance_score,
            'semantic_topic': semantic_topic,
            'keyword_density': keyword_count / len(text.split()) if text.split() else 0,
            'citation_count': len(citations_in_chunk),
            'citations_in_chunk': citations_in_chunk,
            'chunk_confidence': min(1.0, importance_score / 3.0)  # Normalize to 0-1
        }
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document using recursive character splitting.
        
        Args:
            document: Document dictionary with opinion_text and metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        opinion_text = document.get('opinion_text', '')
        if not opinion_text or len(opinion_text.strip()) < self.min_chunk_size:
            logger.warning(f"Document {document.get('id', 'unknown')} has insufficient text for chunking")
            return []
        
        logger.info(f"Chunking document: {document.get('case_name', 'Unknown')} ({document.get('id', 'unknown')})")
        
        # Split text into overlapping chunks
        chunk_texts = self.split_text(opinion_text)
        
        if not chunk_texts:
            logger.warning(f"No chunks created for document {document.get('id', 'unknown')}")
            return []
        
        logger.info(f"  Created {len(chunk_texts)} chunks with overlap of {self.chunk_overlap} characters")
        
        # Post-process chunks and add metadata
        final_chunks = []
        quality_filtered = 0
        
        for idx, chunk_text in enumerate(chunk_texts):
            # Validate chunk quality
            chunk_dict = {'text': chunk_text}
            quality_validation = self._validate_chunk_quality(chunk_dict)
            
            # Skip low-quality chunks
            if not quality_validation['passes_quality_check']:
                logger.info(f"  Skipping chunk {idx} - quality score {quality_validation['quality_score']:.2f} below threshold {self.quality_threshold}")
                quality_filtered += 1
                continue
            
            # Calculate legal metadata
            chunk_metadata = self._calculate_chunk_metadata(chunk_dict, document)
            
            # Build final chunk object
            final_chunk = {
                'chunk_id': str(uuid.uuid4()),
                'document_id': document.get('id'),
                'docket_number': document.get('docket_number'),
                'case_name': document.get('case_name'),
                'court_id': document.get('court_id'),
                'author': document.get('author'),
                'opinion_type': document.get('opinion_type'),
                'date_filed': document.get('date_filed'),
                
                # Chunk-specific data
                'chunk_index': idx,
                'text': chunk_text,
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'token_count': int(len(chunk_text.split()) * 1.3),  # Rough token estimation (~1.3 tokens per word)
                'sentence_count': len([s for s in chunk_text.split('.') if s.strip()]),
                'chunking_method': 'recursive_character',
                'chunk_overlap': self.chunk_overlap,
                
                # Legal metadata
                **chunk_metadata,
                
                # Quality validation results
                **quality_validation,
                
                # Processing metadata
                'created_at': datetime.now().isoformat(),
                'splitter_type': 'RecursiveCharacterTextSplitter',
                'ready_for_embedding': True
            }
            
            final_chunks.append(final_chunk)
        
        logger.info(f"  Created {len(final_chunks)} valid chunks")
        if quality_filtered > 0:
            logger.info(f"  Filtered out {quality_filtered} low-quality chunks")
        return final_chunks
    
    def process_documents(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Process multiple documents from JSON file.
        
        Args:
            input_file: Path to JSON file with documents
            output_file: Output path for chunked documents
            
        Returns:
            Path to output file
        """
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_chunks.json"
        
        logger.info(f"🔄 Starting recursive character text splitting pipeline")
        logger.info(f"📂 Input: {input_file}")
        logger.info(f"💾 Output: {output_file}")
        
        # Load documents
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"📊 Loaded {len(documents)} documents")
        
        # Process each document
        all_chunks = []
        processed_count = 0
        error_count = 0
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
                processed_count += 1
                
                if processed_count % 5 == 0:
                    logger.info(f"  Processed {processed_count}/{len(documents)} documents...")
                    
            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                error_count += 1
                continue
        
        # Save chunks
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        # Generate summary
        total_chars = sum(chunk['char_count'] for chunk in all_chunks)
        total_words = sum(chunk['word_count'] for chunk in all_chunks)
        total_tokens = sum(chunk['token_count'] for chunk in all_chunks)
        avg_importance = sum(chunk['legal_importance_score'] for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        logger.info(f"\n Semantic chunking completed!")
        logger.info(f"📄 Documents processed: {processed_count}")
        logger.info(f"❌ Documents with errors: {error_count}")
        logger.info(f"🔢 Total chunks created: {len(all_chunks)}")
        logger.info(f"📝 Total characters: {total_chars:,}")
        logger.info(f"📝 Total words: {total_words:,}")
        logger.info(f"📝 Total tokens (estimated): {total_tokens:,}")
        logger.info(f"⚖️ Average legal importance: {avg_importance:.2f}")
        logger.info(f"💾 Chunks saved to: {output_file}")
        
        return output_file


def main():
    """Command line interface for recursive character text splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursive character text splitting of legal documents')
    parser.add_argument('input_file', help='Input JSON file with documents')
    parser.add_argument('--output', help='Output file for chunks')
    parser.add_argument('--chunk_size', type=int, default=1500, help='Target chunk size in characters')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap size in characters')
    parser.add_argument('--min_chunk_size', type=int, default=100, help='Minimum chunk size in characters')
    parser.add_argument('--quality_threshold', type=float, default=0.3, help='Minimum quality score for chunks')
    
    args = parser.parse_args()
    
    # Initialize chunker
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
        quality_threshold=args.quality_threshold
    )
    
    # Process documents
    output_file = chunker.process_documents(args.input_file, args.output)
    print(f"\n🎉 Recursive character text splitting complete! Output: {output_file}")


# Backward compatibility alias
SemanticChunker = RecursiveCharacterTextSplitter


if __name__ == "__main__":
    main()
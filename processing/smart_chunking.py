"""
Smart Semantic Chunking for Legal Documents using Legal BERT.

This module provides intelligent chunking of legal documents that groups
semantically similar content together, optimized for legal domain understanding.
"""

import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Semantic chunker that uses legal BERT to create logically coherent chunks
    from legal documents, grouping semantically similar content together.
    """
    
    def __init__(self, 
                 model_name: str = "nlpaueb/legal-bert-base-uncased",
                 target_chunk_size: int = 384,
                 overlap_size: int = 75,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 512,
                 clustering_threshold: float = 0.25,
                 min_cluster_size: int = 2,
                 quality_threshold: float = 0.3):
        """
        Initialize the semantic chunker with legal BERT.
        
        Args:
            model_name: Legal BERT model for semantic understanding
            target_chunk_size: Target size for chunks in tokens
            overlap_size: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to avoid tiny fragments
            max_chunk_size: Maximum chunk size to fit embedding models
        """
        self.model_name = model_name
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
        self.quality_threshold = quality_threshold
        
        # Initialize legal BERT model and tokenizer
        logger.info(f"Loading legal BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using legal-aware patterns."""
        # Enhanced sentence splitting for legal documents
        # Handle citations, legal abbreviations, and court patterns
        
        # Pre-process to protect common legal abbreviations
        abbreviations = [
            r'U\.S\.', r'F\.2d', r'F\.3d', r'S\.Ct\.', r'L\.Ed\.', r'P\.2d', r'P\.3d',
            r'A\.2d', r'A\.3d', r'N\.E\.2d', r'N\.E\.3d', r'S\.E\.2d', r'S\.E\.3d',
            r'So\.2d', r'So\.3d', r'S\.W\.2d', r'S\.W\.3d', r'N\.W\.2d', r'N\.W\.3d',
            r'Cal\.App\.', r'N\.Y\.S\.', r'etc\.', r'Inc\.', r'Corp\.', r'Ltd\.',
            r'v\.', r'vs\.', r'No\.', r'Nos\.', r'J\.', r'C\.J\.', r'e\.g\.', r'i\.e\.',
            r'cf\.', r'supra', r'infra', r'accord', r'contra'
        ]
        
        protected_text = text
        placeholders = {}
        
        # Protect abbreviations
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            matches = re.findall(abbrev, protected_text, re.IGNORECASE)
            for match in matches:
                if match not in placeholders:
                    placeholders[placeholder] = match
                    protected_text = protected_text.replace(match, placeholder)
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, protected_text)
        
        # Restore abbreviations
        restored_sentences = []
        for sentence in sentences:
            for placeholder, original in placeholders.items():
                sentence = sentence.replace(placeholder, original)
            restored_sentences.append(sentence.strip())
        
        # Filter out empty sentences and very short ones
        return [s for s in restored_sentences if len(s.strip()) > 10]
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences using legal BERT."""
        embeddings = []
        
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize and get embeddings
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                sentence_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(sentence_embedding[0])
        
        return np.array(embeddings)
    
    def _cluster_sentences(self, embeddings: np.ndarray, sentences: List[str]) -> List[List[int]]:
        """Cluster sentences based on semantic similarity."""
        if len(embeddings) < 2:
            return [[0]] if len(embeddings) == 1 else []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Use agglomerative clustering with tuned threshold for legal documents
        # Lower threshold creates more cohesive clusters
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.clustering_threshold,  # Optimized for legal domain
            linkage='average',
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group sentence indices by cluster and filter small clusters
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Filter out clusters that are too small (may be noise)
        filtered_clusters = [cluster for cluster in clusters.values() 
                           if len(cluster) >= self.min_cluster_size]
        
        # If we filtered out too many, include single-sentence clusters
        if len(filtered_clusters) < len(clusters) * 0.5:
            filtered_clusters = list(clusters.values())
        
        logger.info(f"    Created {len(filtered_clusters)} semantic clusters (filtered from {len(clusters)})")
        return filtered_clusters
    
    def _create_chunks(self, sentences: List[str], clusters: List[List[int]]) -> List[Dict[str, Any]]:
        """Create chunks from clustered sentences with size constraints."""
        chunks = []
        
        for cluster_idx, sentence_indices in enumerate(clusters):
            # Get sentences for this cluster
            cluster_sentences = [sentences[i] for i in sentence_indices]
            cluster_text = " ".join(cluster_sentences)
            
            # Check if cluster fits in one chunk
            tokens = self.tokenizer.encode(cluster_text, add_special_tokens=False)
            
            if len(tokens) <= self.target_chunk_size:
                # Cluster fits in one chunk
                chunks.append({
                    'text': cluster_text,
                    'sentence_indices': sentence_indices,
                    'semantic_cluster': cluster_idx,
                    'token_count': len(tokens),
                    'sentence_count': len(cluster_sentences)
                })
            else:
                # Split large cluster while maintaining semantic coherence
                sub_chunks = self._split_large_cluster(cluster_sentences, sentence_indices, cluster_idx)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_cluster(self, sentences: List[str], sentence_indices: List[int], cluster_id: int) -> List[Dict[str, Any]]:
        """Split large semantic clusters into appropriately sized chunks."""
        chunks = []
        current_chunk_sentences = []
        current_chunk_indices = []
        current_token_count = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            # Check if adding this sentence would exceed target size
            if current_token_count + sentence_tokens > self.target_chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    'text': chunk_text,
                    'sentence_indices': current_chunk_indices.copy(),
                    'semantic_cluster': cluster_id,
                    'token_count': current_token_count,
                    'sentence_count': len(current_chunk_sentences),
                    'is_split_cluster': True
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) >= 2 else []
                overlap_indices = current_chunk_indices[-2:] if len(current_chunk_indices) >= 2 else []
                overlap_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) for s in overlap_sentences)
                
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_indices = overlap_indices + [sentence_indices[i]]
                current_token_count = overlap_tokens + sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_indices.append(sentence_indices[i])
                current_token_count += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                'text': chunk_text,
                'sentence_indices': current_chunk_indices,
                'semantic_cluster': cluster_id,
                'token_count': current_token_count,
                'sentence_count': len(current_chunk_sentences),
                'is_split_cluster': True
            })
        
        return chunks
    
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
        Chunk a single document using semantic clustering.
        
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
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(opinion_text)
        if len(sentences) < 2:
            # Document too short, return as single chunk
            chunk_metadata = self._calculate_chunk_metadata({'text': opinion_text}, document)
            return [{
                'chunk_id': str(uuid.uuid4()),
                'document_id': document.get('id'),
                'docket_number': document.get('docket_number'),
                'case_name': document.get('case_name'),
                'chunk_index': 0,
                'text': opinion_text,
                'token_count': len(self.tokenizer.encode(opinion_text, add_special_tokens=False)),
                'sentence_count': len(sentences),
                'semantic_cluster': 0,
                'is_single_chunk_document': True,
                **chunk_metadata,
                'created_at': datetime.now().isoformat()
            }]
        
        # Step 2: Get sentence embeddings
        logger.info(f"  Processing {len(sentences)} sentences...")
        embeddings = self._get_sentence_embeddings(sentences)
        
        # Step 3: Cluster sentences semantically
        clusters = self._cluster_sentences(embeddings, sentences)
        logger.info(f"  Found {len(clusters)} semantic clusters")
        
        # Step 4: Create chunks from clusters
        raw_chunks = self._create_chunks(sentences, clusters)
        
        # Step 5: Post-process chunks and add metadata with quality validation
        final_chunks = []
        quality_filtered = 0
        
        for idx, chunk in enumerate(raw_chunks):
            if chunk['token_count'] < self.min_chunk_size:
                logger.info(f"  Skipping chunk {idx} - too small ({chunk['token_count']} tokens)")
                continue
            
            # Validate chunk quality
            quality_validation = self._validate_chunk_quality(chunk)
            
            # Skip low-quality chunks
            if not quality_validation['passes_quality_check']:
                logger.info(f"  Skipping chunk {idx} - quality score {quality_validation['quality_score']:.2f} below threshold {self.quality_threshold}")
                quality_filtered += 1
                continue
            
            # Calculate legal metadata
            chunk_metadata = self._calculate_chunk_metadata(chunk, document)
            
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
                'text': chunk['text'],
                'token_count': chunk['token_count'],
                'sentence_count': chunk['sentence_count'],
                'semantic_cluster': chunk['semantic_cluster'],
                'is_split_cluster': chunk.get('is_split_cluster', False),
                
                # Legal metadata
                **chunk_metadata,
                
                # Quality validation results
                **quality_validation,
                
                # Processing metadata
                'created_at': datetime.now().isoformat(),
                'model_used': self.model_name,
                'ready_for_embedding': True
            }
            
            final_chunks.append(final_chunk)
        
        logger.info(f"  Created {len(final_chunks)} semantic chunks")
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
        
        logger.info(f"🔄 Starting semantic chunking pipeline")
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
        total_tokens = sum(chunk['token_count'] for chunk in all_chunks)
        avg_importance = sum(chunk['legal_importance_score'] for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        logger.info(f"\n Semantic chunking completed!")
        logger.info(f"📄 Documents processed: {processed_count}")
        logger.info(f"❌ Documents with errors: {error_count}")
        logger.info(f"🔢 Total chunks created: {len(all_chunks)}")
        logger.info(f"📝 Total tokens: {total_tokens:,}")
        logger.info(f"⚖️ Average legal importance: {avg_importance:.2f}")
        logger.info(f"💾 Chunks saved to: {output_file}")
        
        return output_file


def main():
    """Command line interface for semantic chunking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic chunking of legal documents')
    parser.add_argument('input_file', help='Input JSON file with documents')
    parser.add_argument('--output', help='Output file for chunks')
    parser.add_argument('--model', default='nlpaueb/legal-bert-base-uncased', help='Legal BERT model to use')
    parser.add_argument('--chunk_size', type=int, default=384, help='Target chunk size in tokens')
    parser.add_argument('--overlap', type=int, default=75, help='Overlap size in tokens')
    parser.add_argument('--clustering_threshold', type=float, default=0.25, help='Clustering distance threshold (lower = more cohesive)')
    parser.add_argument('--min_cluster_size', type=int, default=2, help='Minimum sentences per cluster')
    parser.add_argument('--quality_threshold', type=float, default=0.3, help='Minimum quality score for chunks')
    
    args = parser.parse_args()
    
    # Initialize chunker
    chunker = SemanticChunker(
        model_name=args.model,
        target_chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        clustering_threshold=args.clustering_threshold,
        min_cluster_size=args.min_cluster_size,
        quality_threshold=args.quality_threshold
    )
    
    # Process documents
    output_file = chunker.process_documents(args.input_file, args.output)
    print(f"\n🎉 Semantic chunking complete! Output: {output_file}")


if __name__ == "__main__":
    main()
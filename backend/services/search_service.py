"""
Semantic search service for Q&A recommendations.
Handles loading FAISS index and metadata, and performing semantic search.
"""

import os
import logging
import json
from typing import Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SearchService:
    """Service class for semantic search operations."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.faiss_index = None
        self.metadata = []
        
        # Handle both dict-like and class-like config objects
        if hasattr(config, 'get'):
            self.model_name = config.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        else:
            self.model_name = getattr(config, 'SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        
        # Paths to FAISS index and metadata (container paths)
        self.faiss_index_path = '/app/embedding_data/all_crops/complete_qa_index.faiss'
        self.metadata_path = '/app/embedding_data/all_crops/metadata.json'
    
    def load_faiss_index(self) -> bool:
        """Load FAISS index from file."""
        try:
            if not os.path.exists(self.faiss_index_path):
                logger.error(f"FAISS index not found at {self.faiss_index_path}")
                return False
            
            logger.info(f"Loading FAISS index from {self.faiss_index_path}")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            logger.info(f"FAISS index loaded successfully with {self.faiss_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
    
    def load_metadata(self) -> bool:
        """Load metadata from JSON file."""
        try:
            if not os.path.exists(self.metadata_path):
                logger.error(f"Metadata file not found at {self.metadata_path}")
                return False
            
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                self.metadata = [json.loads(line) for line in f]
            
            logger.info(f"Loaded {len(self.metadata)} metadata entries")
            return True
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return False
        
    def initialize_model(self) -> bool:
        """Initialize sentence transformer model for semantic search."""
        try:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            return False
    
    def load_qa_dataset(self) -> bool:
        """Load FAISS index and metadata for Q&A search."""
        try:
            # Load FAISS index
            if not self.load_faiss_index():
                logger.error("Failed to load FAISS index")
                return False
            
            # Load metadata
            if not self.load_metadata():
                logger.error("Failed to load metadata")
                return False
            
            # Verify that index and metadata sizes match
            if self.faiss_index.ntotal != len(self.metadata):
                logger.warning(f"Index size ({self.faiss_index.ntotal}) doesn't match metadata size ({len(self.metadata)})")
            
            logger.info("✅ Successfully loaded FAISS index and metadata!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Q&A dataset: {str(e)}")
            return False
    
    def query_faiss(self, user_query: str, k: int = 3) -> List[Dict]:
        """
        Direct FAISS query method similar to the notebook example.
        
        Args:
            user_query: The query string
            k: Number of results to return
            
        Returns:
            List of results with score, question, and answer
        """
        try:
            if self.faiss_index is None or not self.metadata:
                return []
            
            # Create embedding for query
            query_embedding = self.model.encode([user_query], convert_to_numpy=True)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Collect results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    result = {
                        "score": float(dist),
                        "QUESTION [by Agri Team]": self.metadata[idx]["QUESTION [by Agri Team]"],
                        "Final_Answer": self.metadata[idx]["Final Answer"]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query_faiss: {str(e)}")
            return []
    
    def semantic_search(self, user_question: str, similarity_threshold: float = None, 
                       top_k: int = None) -> Dict:
        """
        Perform semantic search using FAISS index.
        
        Args:
            user_question: The question to search for
            similarity_threshold: Minimum similarity score for results (not used with FAISS distances)
            top_k: Number of top results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Use default values if not provided
            if top_k is None:
                if hasattr(self.config, 'get'):
                    top_k = self.config.get('DEFAULT_TOP_K', 3)
                else:
                    top_k = getattr(self.config, 'DEFAULT_TOP_K', 3)
                
            # Check if FAISS index and metadata are loaded
            if self.faiss_index is None or not self.metadata:
                return {
                    'status': 'error',
                    'message': 'FAISS index or metadata not loaded properly'
                }
                
            # Encode user question
            query_embedding = self.model.encode([user_question], convert_to_numpy=True)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Process results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.metadata):  # Ensure index is valid
                    result = {
                        "score": float(dist),
                        "question": self.metadata[idx]["QUESTION [by Agri Team]"],
                        "answer": self.metadata[idx]["Final Answer"],
                        "index": int(idx)
                    }
                    results.append(result)
            
            # Format results (FAISS returns distances, lower is better)
            formatted_results = []
            for i, result in enumerate(results):
                # Convert distance to similarity-like score (lower distance = higher similarity)
                # Note: This is approximate, actual similarity depends on the distance metric used
                similarity_score = 1.0 / (1.0 + result['score'])  # Simple conversion
                confidence = "High" if similarity_score > 0.8 else "Medium" if similarity_score > 0.6 else "Low"
                
                formatted_results.append({
                    "rank": i + 1,
                    "answer": result['answer'],
                    "matched_question": result['question'],
                    "distance": result['score'],
                    "similarity_score": similarity_score,
                    "confidence": confidence
                })
            
            return {
                "status": "success",
                "results": formatted_results,
                "query": user_question,
                "total_results": len(formatted_results)
            }
                
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error during semantic search: {str(e)}'
            }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the loaded dataset."""
        if self.faiss_index is None or not self.metadata:
            return {
                'loaded': False,
                'message': 'No dataset loaded'
            }
            
        questions = [item.get("QUESTION [by Agri Team]", "") for item in self.metadata]
        answers = [item.get("Final Answer", "") for item in self.metadata]
        
        return {
            'loaded': True,
            'total_pairs': len(self.metadata),
            'faiss_index_size': self.faiss_index.ntotal,
            'avg_question_length': sum(len(q.split()) for q in questions) / len(questions) if questions else 0,
            'avg_answer_length': sum(len(a.split()) for a in answers) / len(answers) if answers else 0,
            'sample_questions': questions[:5] if questions else []
        }
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> Dict:
        """
        Search for questions containing specific keywords.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            
        Returns:
            Dictionary containing matching questions and answers
        """
        try:
            if not self.metadata:
                return {
                    'status': 'error',
                    'message': 'No dataset loaded'
                }
                
            # Convert keywords to lowercase for case-insensitive search
            keywords_lower = [kw.lower() for kw in keywords]
            
            matches = []
            for i, item in enumerate(self.metadata):
                question = item.get("QUESTION [by Agri Team]", "")
                question_lower = question.lower()
                
                # Count how many keywords are found in the question
                keyword_matches = sum(1 for kw in keywords_lower if kw in question_lower)
                
                if keyword_matches > 0:
                    matches.append({
                        'question': question,
                        'answer': item.get("Final Answer", ""),
                        'keyword_matches': keyword_matches,
                        'index': i
                    })
            
            # Sort by number of keyword matches (descending)
            matches.sort(key=lambda x: x['keyword_matches'], reverse=True)
            
            # Return top k results
            results = matches[:top_k]
            
            return {
                'status': 'success',
                'results': results,
                'total_matches': len(matches),
                'keywords_searched': keywords
            }
            
        except Exception as e:
            logger.error(f"Error during keyword search: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error during keyword search: {str(e)}'
            }


logging.basicConfig(
    filename='search_logs.txt',  # file where logs will be written
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

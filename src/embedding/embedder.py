# src/embedding/embedder.py
from openai import OpenAI
from typing import List
from src.config.settings import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed


class Embedder:
    """OpenAI embedding client."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_embedding_model
        self.max_workers = config.max_workers
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using multithreading.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process per API call (default: 100)
        
        Returns:
            List of embedding vectors in the same order as input texts
        """
        if not texts:
            return []
        
        # If texts fit in a single batch, process directly (no threading overhead)
        if len(texts) <= batch_size:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        
        # Split texts into batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i, batch))
        
        # Process batches in parallel using ThreadPoolExecutor
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._embed_batch, batch): (batch_idx, batch)
                for batch_idx, batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    results_dict[batch_idx] = batch_embeddings
                except Exception as e:
                    # Handle errors gracefully - raise with context
                    raise RuntimeError(f"Error processing embedding batch starting at index {batch_idx}: {e}") from e
        
        # Reconstruct results in original order
        all_embeddings = []
        for batch_idx in sorted(results_dict.keys()):
            all_embeddings.extend(results_dict[batch_idx])
        
        return all_embeddings
    
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (internal method for threading).
        
        Args:
            batch: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=batch
        )
        return [item.embedding for item in response.data]
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        
        return response.data[0].embedding